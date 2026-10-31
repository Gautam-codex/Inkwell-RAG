"""Phase 4 tests: RRF fusion, hybrid retrieval modes, document filter, re-ranking."""

import pytest
from langchain_core.documents import Document

from src.config import Settings
from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import VectorStore
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalError, reciprocal_rank_fusion
from tests.conftest import FakeEmbeddings

# ---------- RRF (pure function) ----------


def test_rrf_formula_and_order():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)  # rank 2 in list 1, rank 1 in list 2
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["d"] == pytest.approx(1 / 62)
    ranked = [cid for cid, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)]
    assert ranked[0] == "b"  # found by both lists, so it wins


def test_rrf_deduplicates():
    ranked = reciprocal_rank_fusion([["x", "y"], ["y", "x"], ["x"]])
    assert [cid for cid, _ in ranked] == ["x", "y"]


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


# ---------- Retriever ----------

CHUNKS = [
    ("d1", 1, "Pump P1 circulates coolant at 40 litres per minute."),
    ("d1", 2, "The heat exchanger transfers heat from the coolant to the air."),
    ("d1", 3, "Replace the coolant filter every six months."),
    ("d2", 1, "Quarterly revenue grew to 140 million in Q2."),
    ("d2", 2, "Operating profit was 22 million in Q2."),
    ("d2", 3, "The board approved a new dividend policy."),
]


class WordOverlapReranker:
    """Fake cross-encoder: score = share of query words found in the chunk."""

    def score(self, query, texts):
        words = set(query.lower().split())
        return [len(words & set(t.lower().replace(".", "").split())) / len(words) for t in texts]


@pytest.fixture
def retriever(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, local_embedding_model="fake")
    settings.ensure_dirs()
    store = VectorStore(settings, embeddings=FakeEmbeddings())
    docs = [
        Document(t, metadata={"doc_id": d, "file_name": f"{d}.pdf", "page": p, "source_type": "text", "chunk_id": f"{d}:p{p}"})
        for d, p, t in CHUNKS
    ]
    store.add_chunks(docs)
    bm25 = BM25Index(settings.bm25_path)
    bm25.build(store.all_chunks())
    return HybridRetriever(store, bm25, reranker=WordOverlapReranker(), settings=settings)


def test_hybrid_rerank_returns_all_scores(retriever):
    result = retriever.retrieve("coolant filter replace", top_n=3)
    assert len(result.chunks) == 3
    top = result.chunks[0]
    assert top.chunk_id == "d1:p3"
    assert top.rerank_score is not None and top.fused_rank is not None
    assert top.dense_score is not None and top.bm25_score is not None  # found by both
    assert [c.rerank_score for c in result.chunks] == sorted((c.rerank_score for c in result.chunks), reverse=True)
    assert {"dense_ms", "bm25_ms", "rerank_ms", "total_ms"} <= result.timings_ms.keys()


def test_no_duplicate_chunks(retriever):
    result = retriever.retrieve("coolant", mode="hybrid", top_n=10)
    ids = [c.chunk_id for c in result.chunks]
    assert len(ids) == len(set(ids))


def test_modes_use_the_right_searches(retriever):
    dense = retriever.retrieve("revenue Q2", mode="dense").chunks
    assert dense and all(c.bm25_score is None and c.rerank_score is None for c in dense)

    bm25 = retriever.retrieve("revenue Q2", mode="bm25").chunks
    assert bm25[0].chunk_id == "d2:p1"
    assert all(c.dense_score is None for c in bm25)

    hybrid = retriever.retrieve("revenue Q2", mode="hybrid").chunks
    assert all(c.rerank_score is None for c in hybrid)
    assert [c.fused_rank for c in hybrid] == list(range(1, len(hybrid) + 1))


def test_document_filter(retriever):
    result = retriever.retrieve("coolant revenue profit", doc_ids=["d2"], top_n=10)
    assert result.chunks and all(c.metadata["doc_id"] == "d2" for c in result.chunks)


def test_empty_question_is_rejected(retriever):
    with pytest.raises(RetrievalError):
        retriever.retrieve("   ")


def test_model_mismatch_is_reported(tmp_path, retriever):
    other = Settings(_env_file=None, data_dir=retriever.settings.data_dir, local_embedding_model="other")
    store = VectorStore(other, embeddings=FakeEmbeddings())
    with pytest.raises(RetrievalError, match="Rebuild"):
        HybridRetriever(store, retriever.bm25, reranker=WordOverlapReranker(), settings=other).retrieve("coolant")
