"""Indexing pipeline: PDF -> parse -> chunk -> ChromaDB + BM25.

This is the one place the UI and the CLI call to add, delete or rebuild documents.

  - Each PDF is copied to data/uploads/<doc_id>/<file name>, so the index can be
    rebuilt later (e.g. after switching the embedding model).
  - doc_id is the file's hash, so the same file is never indexed twice.
  - After any change the BM25 index is rebuilt from Chroma, so both always hold
    exactly the same chunks.

Command line:
    python -m src.indexing.indexer path/to/folder      index every PDF in a folder
    python -m src.indexing.indexer report.pdf          index one PDF
    python -m src.indexing.indexer --list              show indexed documents
    python -m src.indexing.indexer --delete <doc_id>   remove a document
    python -m src.indexing.indexer --rebuild           re-embed everything
"""

import argparse
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

from src.config import Settings, get_settings
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import EmbeddingModelMismatch, VectorStore
from src.ingestion.chunker import chunk_document
from src.ingestion.image_captioner import describe_figures
from src.ingestion.pdf_parser import PDFParseError, parse_pdf
from src.ingestion.table_extractor import Summarizer
from src.utils.hashing import doc_id_for_file
from src.utils.logging import get_logger

logger = get_logger(__name__)

# progress(message, fraction from 0.0 to 1.0) - matches Streamlit's st.progress(value, text).
Progress = Callable[[str, float], None]


@dataclass
class IndexResult:
    """What happened to one file."""

    file_name: str
    status: Literal["indexed", "duplicate", "error"]
    doc_id: str = ""
    message: str = ""
    pages: int = 0
    counts: dict[str, int] = field(default_factory=dict)  # chunks per source_type


class Indexer:
    """Adds, deletes and lists documents across ChromaDB and BM25."""

    def __init__(self, settings: Optional[Settings] = None, vector_store: Optional[VectorStore] = None) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or VectorStore(self.settings)
        self.bm25 = BM25Index(self.settings.bm25_path)
        # Load the saved BM25 index; rebuild it if missing or out of sync with Chroma.
        if not self.bm25.load() or len(self.bm25) != self.vector_store.count():
            self.rebuild_bm25()

    def rebuild_bm25(self) -> None:
        """Rebuild the BM25 index from every chunk currently in Chroma."""
        self.bm25.build(self.vector_store.all_chunks())

    # ---------- adding ----------

    def index_file(
        self,
        path: str | Path,
        progress: Optional[Progress] = None,
        table_summarizer: Optional[Summarizer] = None,
        update_bm25: bool = True,
    ) -> IndexResult:
        """Index one PDF. Never raises for a bad file; the problem is in the result instead."""
        path = Path(path)
        report = progress or (lambda message, fraction: None)
        try:
            doc_id = doc_id_for_file(path)
        except OSError as exc:
            return IndexResult(path.name, "error", message=f"Cannot read '{path.name}': {exc}")

        if self.vector_store.model_mismatch:
            return IndexResult(path.name, "error", doc_id, self.vector_store.mismatch_message())
        if self.vector_store.has_document(doc_id):
            logger.info("Skipping %s: already indexed as %s", path.name, doc_id)
            return IndexResult(path.name, "duplicate", doc_id, "Already indexed, skipped.")

        if table_summarizer is None and self.settings.llm_table_summaries:
            table_summarizer = self._table_summarizer()

        try:
            # Progress bar split: parsing 0-40%, figures 40-80%, embedding 80-100%.
            parsed = parse_pdf(path, progress=lambda done, total: report(f"Parsing page {done}/{total}", 0.4 * done / total))
            figures = describe_figures(
                parsed, doc_id, self.settings,
                progress=lambda done, total: report(f"Describing figure {done}/{total}", 0.4 + 0.4 * done / total),
            )
            chunks = chunk_document(
                parsed, doc_id,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                table_summarizer=table_summarizer,
                figures=figures,
            )
            if not chunks:
                return IndexResult(path.name, "error", doc_id, "No text, tables, figures or scanned text found in this PDF.")

            # Keep a copy of the PDF so the index can be rebuilt later.
            stored = self._store_upload(path, doc_id)
            self.vector_store.add_chunks(
                chunks, progress=lambda done, total: report(f"Embedding chunk {done}/{total}", 0.8 + 0.2 * done / total)
            )
        except (PDFParseError, EmbeddingModelMismatch) as exc:
            return IndexResult(path.name, "error", doc_id, str(exc))
        except Exception as exc:  # unexpected: log the full traceback, show a short message
            logger.exception("Indexing %s failed", path.name)
            # Don't leave half a document behind.
            self.vector_store.delete_document(doc_id)
            for folder in (self.settings.uploads_dir / doc_id, self.settings.images_dir / doc_id):
                shutil.rmtree(folder, ignore_errors=True)
            return IndexResult(path.name, "error", doc_id, f"Indexing failed: {exc}")

        if update_bm25:
            self.rebuild_bm25()
        counts = dict(Counter(c.metadata["source_type"] for c in chunks))
        report("Done", 1.0)
        logger.info("Indexed %s as %s (%s), stored at %s", path.name, doc_id, counts, stored)
        return IndexResult(path.name, "indexed", doc_id, "Indexed.", parsed.page_count, counts)

    def index_folder(self, folder: str | Path, progress: Optional[Progress] = None) -> list[IndexResult]:
        """Index every PDF in a folder (not sub-folders). BM25 is rebuilt once at the end."""
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() == ".pdf")
        results = []
        for i, path in enumerate(files, start=1):
            if progress:
                progress(f"File {i}/{len(files)}: {path.name}", (i - 1) / len(files))
            results.append(self.index_file(path, update_bm25=False))
        if any(r.status == "indexed" for r in results):
            self.rebuild_bm25()
        return results

    def _table_summarizer(self) -> Optional[Summarizer]:
        """The LLM used for table summaries, or None (header-based summary) if it isn't ready."""
        from src.generation.llm import check_llm, make_summarizer

        problem = check_llm(self.settings)
        if problem:
            logger.warning("Table summaries will use the table header instead of the LLM: %s", problem)
            return None
        return make_summarizer(self.settings)

    def _store_upload(self, path: Path, doc_id: str) -> Path:
        """Copy the PDF into data/uploads/<doc_id>/ (skipped if it is already there)."""
        target = self.settings.uploads_dir / doc_id / path.name
        if path.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        return target

    # ---------- deleting, listing, rebuilding ----------

    def delete_document(self, doc_id: str) -> int:
        """Remove a document from Chroma and BM25, plus its stored PDF and images."""
        removed = self.vector_store.delete_document(doc_id)
        for folder in (self.settings.uploads_dir / doc_id, self.settings.images_dir / doc_id):
            shutil.rmtree(folder, ignore_errors=True)
        self.rebuild_bm25()
        return removed

    def list_documents(self) -> list[dict]:
        return self.vector_store.list_documents()

    def rebuild_all(self, progress: Optional[Progress] = None) -> list[IndexResult]:
        """Empty the collection and re-index every stored PDF with the current embedding model.

        This is the fix for an embedding-model mismatch.
        """
        stored_files = sorted(self.settings.uploads_dir.glob("*/*.pdf"))
        self.vector_store.reset()
        results = []
        for i, path in enumerate(stored_files, start=1):
            if progress:
                progress(f"Re-indexing {i}/{len(stored_files)}: {path.name}", (i - 1) / max(len(stored_files), 1))
            results.append(self.index_file(path, update_bm25=False))
        self.rebuild_bm25()
        return results


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _print_results(results: list[IndexResult]) -> None:
    for r in results:
        detail = f"{r.pages} pages, chunks {r.counts}" if r.status == "indexed" else r.message
        print(f"  [{r.status:>9}] {r.file_name} ({r.doc_id}) - {detail}")


def _print_documents(indexer: Indexer) -> None:
    docs = indexer.list_documents()
    print(f"\n{len(docs)} document(s) indexed with {indexer.vector_store.stored_model}")
    for d in docs:
        print(f"  {d['doc_id']}  {d['file_name']}  - content on {d['pages']} pages, {d['chunks']} chunks {d['types']}")
    print(f"Chroma chunks: {indexer.vector_store.count()} | BM25 chunks: {len(indexer.bm25)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDFs into ChromaDB + BM25.")
    parser.add_argument("path", nargs="?", help="A PDF file or a folder of PDFs")
    parser.add_argument("--list", action="store_true", help="List indexed documents")
    parser.add_argument("--delete", metavar="DOC_ID", help="Delete a document by its doc_id")
    parser.add_argument("--rebuild", action="store_true", help="Re-index all stored PDFs (after changing the embedding model)")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    indexer = Indexer()
    if indexer.vector_store.model_mismatch and not args.rebuild:
        print(f"WARNING: {indexer.vector_store.mismatch_message()}\nRun with --rebuild to fix it.")

    def show(message: str, fraction: float) -> None:
        print(f"\r  {fraction:4.0%}  {message:<50}", end="", flush=True)

    if args.delete:
        removed = indexer.delete_document(args.delete)
        print(f"Removed {removed} chunks." if removed else f"No document with doc_id '{args.delete}'.")
    elif args.rebuild:
        results = indexer.rebuild_all(progress=show)
        print()
        _print_results(results)
    elif args.path:
        target = Path(args.path)
        if target.is_dir():
            results = indexer.index_folder(target, progress=show)
        elif target.is_file():
            results = [indexer.index_file(target, progress=show)]
        else:
            print(f"ERROR: '{target}' does not exist.")
            sys.exit(1)
        print()
        _print_results(results)
    elif not args.list:
        parser.print_help()
        return
    _print_documents(indexer)


if __name__ == "__main__":
    main()
