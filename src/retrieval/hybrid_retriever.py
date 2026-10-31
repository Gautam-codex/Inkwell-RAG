"""Hybrid retrieval: dense (ChromaDB) + sparse (BM25), merged with Reciprocal Rank
Fusion, then re-ranked with a cross-encoder.

    question --> dense top-20 --\
                                 >-- RRF merge (dedup by chunk_id) --> cross-encoder --> top-5
    question --> BM25  top-20 --/

Every returned chunk keeps all its scores (dense, BM25, fused rank, re-rank) so the
UI and the evaluation can show where each result came from.

Try it from the command line:
    python -m src.retrieval.hybrid_retriever "what is the maintenance interval?"
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from langchain_core.documents import Document

from src.config import Settings, get_settings
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import VectorStore
from src.retrieval.reranker import Reranker
from src.utils.logging import get_logger

logger = get_logger(__name__)

# The four set-ups compared in the evaluation (Phase 8).
Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
MODES: tuple[str, ...] = ("dense", "bm25", "hybrid", "hybrid_rerank")


class RetrievalError(Exception):
    """A readable problem that stops retrieval (e.g. index built with another model)."""


@dataclass
class RetrievedChunk:
    """A chunk plus every score it collected on the way. None = not found by that method."""

    document: Document
    dense_score: Optional[float] = None  # cosine similarity, -1..1 (higher = closer)
    dense_rank: Optional[int] = None  # 1 = best
    bm25_score: Optional[float] = None  # 0..unbounded (higher = more keyword overlap)
    bm25_rank: Optional[int] = None
    fused_score: float = 0.0  # RRF score
    fused_rank: Optional[int] = None
    rerank_score: Optional[float] = None  # cross-encoder, 0..1

    @property
    def chunk_id(self) -> str:
        return self.document.metadata["chunk_id"]

    @property
    def metadata(self) -> dict:
        return self.document.metadata


@dataclass
class RetrievalResult:
    """The final chunks plus how long each step took (milliseconds)."""

    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge several ranked lists of chunk IDs into one.

    RRF formula:   score(chunk) = sum over lists of  1 / (k + rank_in_that_list)

    - rank starts at 1, so the top result of a list adds 1/61 with k = 60.
    - A chunk found by BOTH searches gets two contributions, so agreement wins.
    - Only ranks are used, never raw scores. That matters because cosine
      similarity (0..1) and BM25 (0..20+) are on completely different scales.
    - k = 60 (from the original RRF paper) softens the gap between rank 1 and
      rank 2, so one list cannot dominate just by its top item.

    A chunk that appears in several lists is counted once (deduplicated by ID).
    Returns (chunk_id, rrf_score) pairs, best first.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    """Runs dense + BM25 search, fuses with RRF and re-ranks with a cross-encoder."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Index,
        reranker: Optional[Reranker] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker or Reranker(self.settings)

    def warm_up(self) -> None:
        """Load the embedding model and the re-ranker up front.

        Loading takes several seconds; doing it here keeps that time out of the
        per-question latency we measure and show.
        """
        _ = self.vector_store.embeddings
        if hasattr(self.reranker, "load"):
            self.reranker.load()

    # ---------- the two searches ----------

    def dense_search(self, query: str, top_k: int, doc_ids: Optional[list[str]] = None) -> list[RetrievedChunk]:
        """Embedding search in ChromaDB."""
        if self.vector_store.model_mismatch:
            raise RetrievalError(self.vector_store.mismatch_message())
        available = self.vector_store.count()
        if available == 0:
            return []
        where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        results = self.vector_store.langchain_store.similarity_search_with_score(
            query, k=min(top_k, available), filter=where
        )
        # Chroma returns a cosine DISTANCE (0 = identical); similarity = 1 - distance.
        return [
            RetrievedChunk(document=doc, dense_score=1.0 - distance, dense_rank=rank)
            for rank, (doc, distance) in enumerate(results, start=1)
        ]

    def bm25_search(self, query: str, top_k: int, doc_ids: Optional[list[str]] = None) -> list[RetrievedChunk]:
        """Keyword search in the BM25 index."""
        return [
            RetrievedChunk(document=doc, bm25_score=score, bm25_rank=rank)
            for rank, (doc, score) in enumerate(self.bm25.search(query, top_k, doc_ids), start=1)
        ]

    # ---------- fusion and re-ranking ----------

    def fuse(self, dense: list[RetrievedChunk], sparse: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Merge both lists with RRF into one list of unique chunks, keeping both scores."""
        by_id: dict[str, RetrievedChunk] = {}
        for item in dense + sparse:
            merged = by_id.setdefault(item.chunk_id, RetrievedChunk(document=item.document))
            if item.dense_rank is not None:
                merged.dense_score, merged.dense_rank = item.dense_score, item.dense_rank
            if item.bm25_rank is not None:
                merged.bm25_score, merged.bm25_rank = item.bm25_score, item.bm25_rank

        fused = reciprocal_rank_fusion(
            [[c.chunk_id for c in dense], [c.chunk_id for c in sparse]], k=self.settings.rrf_k
        )
        ordered = []
        for rank, (chunk_id, score) in enumerate(fused, start=1):
            chunk = by_id[chunk_id]
            chunk.fused_score, chunk.fused_rank = score, rank
            ordered.append(chunk)
        return ordered

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
        """Score every candidate with the cross-encoder and keep the best top_n."""
        scores = self.reranker.score(query, [c.document.page_content for c in chunks])
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = score
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_n]

    # ---------- everything together ----------

    def retrieve(
        self,
        query: str,
        mode: Mode = "hybrid_rerank",
        dense_k: Optional[int] = None,
        bm25_k: Optional[int] = None,
        top_n: Optional[int] = None,
        doc_ids: Optional[list[str]] = None,
    ) -> RetrievalResult:
        """Find the most relevant chunks for a question.

        mode chooses the pipeline (the evaluation compares all four):
          "dense" | "bm25" | "hybrid" (RRF, no re-ranker) | "hybrid_rerank" (default)
        doc_ids restricts the search to the selected documents.
        """
        if not query or not query.strip():
            raise RetrievalError("The question is empty.")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        s = self.settings
        dense_k, bm25_k, top_n = dense_k or s.dense_top_k, bm25_k or s.bm25_top_k, top_n or s.rerank_top_n
        timings: dict[str, float] = {}
        started = time.perf_counter()

        dense: list[RetrievedChunk] = []
        sparse: list[RetrievedChunk] = []
        if mode != "bm25":
            t = time.perf_counter()
            dense = self.dense_search(query, dense_k, doc_ids)
            timings["dense_ms"] = (time.perf_counter() - t) * 1000
        if mode != "dense":
            t = time.perf_counter()
            sparse = self.bm25_search(query, bm25_k, doc_ids)
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000

        # Fusion also fills fused_rank for single-list modes, so all modes look alike.
        candidates = self.fuse(dense, sparse)

        if mode == "hybrid_rerank" and candidates:
            t = time.perf_counter()
            final = self.rerank(query, candidates, top_n)
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000
        else:
            final = candidates[:top_n]

        timings["total_ms"] = (time.perf_counter() - started) * 1000
        logger.info(
            "Retrieved %d chunks (mode=%s, %d dense + %d bm25 -> %d fused) in %.0f ms",
            len(final), mode, len(dense), len(sparse), len(candidates), timings["total_ms"],
        )
        return RetrievalResult(chunks=final, timings_ms=timings)


# ---------------------------------------------------------------------------
# Command line:  python -m src.retrieval.hybrid_retriever "your question"
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> None:
    from src.indexing.indexer import Indexer

    parser = argparse.ArgumentParser(description="Search the index and show every score.")
    parser.add_argument("query", help="The question, in quotes")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("--top-n", type=int, default=None, help="How many final chunks (default from .env)")
    parser.add_argument("--doc", action="append", metavar="DOC_ID", help="Only search this document (repeatable)")
    parser.add_argument("--full", action="store_true", help="Print the full chunk text instead of a preview")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")

    indexer = Indexer()
    retriever = HybridRetriever(indexer.vector_store, indexer.bm25)
    if args.mode != "bm25":
        print("Loading models...")
        retriever.warm_up()
    try:
        result = retriever.retrieve(args.query, mode=args.mode, top_n=args.top_n, doc_ids=args.doc)
    except RetrievalError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f'\nQuery: "{args.query}"   mode: {args.mode}\n')
    if not result.chunks:
        print("No matching chunks. Is anything indexed? Try: python -m src.indexing.indexer --list")
        return
    header = f"{'#':>2} | {'rerank':>6} | {'fused#':>6} | {'RRF':>6} | {'dense':>6} | {'d#':>3} | {'bm25':>6} | {'b#':>3} | source"
    print(header)
    print("-" * len(header))
    for i, c in enumerate(result.chunks, start=1):
        m = c.metadata
        print(
            f"{i:>2} | {_fmt(c.rerank_score):>6} | {c.fused_rank or '-':>6} | {_fmt(c.fused_score, 4):>6} | "
            f"{_fmt(c.dense_score):>6} | {c.dense_rank or '-':>3} | {_fmt(c.bm25_score, 2):>6} | {c.bm25_rank or '-':>3} | "
            f"{m['file_name']} p.{m['page']} ({m['source_type']})"
        )
        text = c.document.page_content if args.full else c.document.page_content[:160].replace("\n", " ") + "..."
        print(f"     {text}\n")
    timings = ", ".join(f"{name.replace('_ms', '')} {ms:.0f} ms" for name, ms in result.timings_ms.items())
    print(f"Timings: {timings}")


if __name__ == "__main__":
    main()
