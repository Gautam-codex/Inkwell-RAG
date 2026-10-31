"""Turn a parsed PDF into chunks (LangChain Documents) with metadata.

Rules:
  - Text and OCR text are split page by page with RecursiveCharacterTextSplitter,
    so a chunk never mixes two pages and its page number is always exact.
  - Each table is ONE chunk (never split), with a one-line summary in front.
  - Every chunk carries: doc_id, file_name, page, source_type, chunk_id
    (+ image_path for figures).
  - Each described figure is ONE chunk with source_type "figure".

Check the chunks of a PDF from the command line:
    python -m src.ingestion.chunker sample.pdf
"""

import argparse
import sys
from collections import Counter
from typing import TYPE_CHECKING, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.ingestion.pdf_parser import ParsedDocument
from src.ingestion.table_extractor import Summarizer, summarize_table
from src.utils.logging import get_logger

if TYPE_CHECKING:  # only for the type hint; avoids importing the vision code here
    from src.ingestion.image_captioner import FigureDescription

logger = get_logger(__name__)

SOURCE_TYPES = ("text", "table", "figure", "ocr")


def make_chunk_id(doc_id: str, page: int, source_type: str, index: int) -> str:
    """Build a readable, unique and repeatable ID, e.g. '3f2a...:p12:table:0'.

    The same PDF always produces the same IDs, which makes duplicates easy to spot
    when merging dense and BM25 results later.
    """
    return f"{doc_id}:p{page}:{source_type}:{index}"


def make_chunk(
    text: str,
    doc_id: str,
    file_name: str,
    page: int,
    source_type: str,
    index: int,
    image_path: Optional[str] = None,
) -> Document:
    """Create one chunk with the standard metadata."""
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"Unknown source_type '{source_type}', expected one of {SOURCE_TYPES}")
    metadata = {
        "doc_id": doc_id,
        "file_name": file_name,
        "page": page,
        "source_type": source_type,
        "chunk_id": make_chunk_id(doc_id, page, source_type, index),
    }
    # ChromaDB does not accept None values, so the key is only added when there is a path.
    if image_path:
        metadata["image_path"] = image_path
    return Document(page_content=text, metadata=metadata)


def chunk_document(
    parsed: ParsedDocument,
    doc_id: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    table_summarizer: Optional[Summarizer] = None,
    figures: Optional[list["FigureDescription"]] = None,
) -> list[Document]:
    """Split a parsed PDF into chunks ready for embedding.

    chunk_size and chunk_overlap are in characters and default to the values in .env.
    table_summarizer is an optional LLM call for table summaries.
    figures are the described figures from image_captioner.describe_figures().
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Tries to split on paragraphs first, then lines, then words, then characters,
    # so chunks end at natural boundaries whenever possible.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for page in parsed.pages:
        # Text or OCR text, split into overlapping pieces.
        source_type = "ocr" if page.is_ocr else "text"
        pieces = [p.strip() for p in splitter.split_text(page.text) if p.strip()]
        for i, piece in enumerate(pieces):
            chunks.append(make_chunk(piece, doc_id, parsed.file_name, page.page, source_type, i))

        # Each table as a single chunk: summary line + full Markdown table.
        for i, table in enumerate(page.tables):
            summary = summarize_table(table, table_summarizer)
            text = f"{summary}\n\n{table.markdown}"
            chunks.append(make_chunk(text, doc_id, parsed.file_name, page.page, "table", i))

    # Each figure description as a single chunk, pointing at its saved image.
    figure_index: Counter = Counter()  # numbers figures 0, 1, 2... on each page
    for fig in figures or []:
        i = figure_index[fig.page]
        figure_index[fig.page] += 1
        chunks.append(
            make_chunk(fig.chunk_text(), doc_id, parsed.file_name, fig.page, "figure", i, image_path=fig.image_path)
        )

    counts = Counter(c.metadata["source_type"] for c in chunks)
    logger.info("Chunked %s into %d chunks %s", parsed.file_name, len(chunks), dict(counts))
    return chunks


# ---------------------------------------------------------------------------
# Command-line check:  python -m src.ingestion.chunker sample.pdf
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse and chunk a PDF, then print chunk counts and a few example chunks."""
    from src.ingestion.pdf_parser import PDFParseError, parse_pdf
    from src.utils.hashing import doc_id_for_file

    parser = argparse.ArgumentParser(description="Show the chunks produced for a PDF.")
    parser.add_argument("pdf", help="Path to a PDF file")
    parser.add_argument("--show", type=int, default=3, help="How many chunks of each type to print (default 3)")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    sys.stdout.reconfigure(errors="replace")

    try:
        parsed = parse_pdf(args.pdf)
    except PDFParseError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    chunks = chunk_document(parsed, doc_id_for_file(args.pdf), args.chunk_size, args.chunk_overlap)
    counts = Counter(c.metadata["source_type"] for c in chunks)
    print(f"\n{parsed.file_name}: {len(chunks)} chunks {dict(counts)}\n")

    for source_type in SOURCE_TYPES:
        examples = [c for c in chunks if c.metadata["source_type"] == source_type][: args.show]
        for chunk in examples:
            print("=" * 70)
            print(chunk.metadata)
            print("-" * 70)
            print(chunk.page_content[:600] + ("..." if len(chunk.page_content) > 600 else ""))
    print("=" * 70)


if __name__ == "__main__":
    main()
