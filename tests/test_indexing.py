"""Phase 3 tests: BM25 build/save/load/search, Chroma indexing, duplicates, delete, model check.

A tiny fake embedding model is used so these tests run in seconds and need no download.
"""

from langchain_core.documents import Document

from src.config import Settings
from src.indexing.bm25_index import BM25Index, tokenize
from src.indexing.indexer import Indexer
from src.indexing.vector_store import VectorStore
from tests.conftest import FakeEmbeddings


def make_settings(tmp_path, model="fake-model-a") -> Settings:
    # No LLM or vision calls in these tests: they only check the indexing plumbing.
    s = Settings(
        _env_file=None, data_dir=tmp_path, local_embedding_model=model,
        llm_table_summaries=False, caption_figures=False,
    )
    s.ensure_dirs()
    return s


def make_indexer(settings) -> Indexer:
    return Indexer(settings, vector_store=VectorStore(settings, embeddings=FakeEmbeddings()))


def doc(text, doc_id="d1", n=0) -> Document:
    return Document(page_content=text, metadata={"doc_id": doc_id, "chunk_id": f"{doc_id}:{n}", "page": 1})


# ---------- BM25 ----------


def test_tokenize_keeps_codes_and_drops_stopwords():
    assert tokenize("What is the Q1-2023 revenue of llama3.2?") == ["q1-2023", "revenue", "llama3.2"]


def test_bm25_build_save_load_search(tmp_path):
    chunks = [
        doc("pump P1 circulates coolant", n=0),
        doc("the heat exchanger cools water", n=1),
        doc("quarterly revenue grew strongly", "d2", 2),
        doc("filter replacement every six months", "d2", 3),
    ]
    index = BM25Index(tmp_path / "bm25.pkl")
    index.build(chunks)

    loaded = BM25Index(tmp_path / "bm25.pkl")
    assert loaded.load() and len(loaded) == 4

    results = loaded.search("coolant pump", top_k=5)
    assert results[0][0].metadata["chunk_id"] == "d1:0"
    assert all(score > 0 for _, score in results)

    # Restricting to a document only returns chunks from that document.
    assert loaded.search("coolant pump", top_k=5, doc_ids=["d2"]) == []


def test_bm25_empty_and_missing(tmp_path):
    index = BM25Index(tmp_path / "none.pkl")
    assert index.load() is False
    index.build([])
    assert index.search("anything", top_k=5) == []


# ---------- Indexer ----------


def test_index_duplicate_and_delete(tmp_path, sample_pdf):
    settings = make_settings(tmp_path)
    indexer = make_indexer(settings)

    first = indexer.index_file(sample_pdf)
    assert first.status == "indexed"
    assert set(first.counts) == {"text", "table", "ocr"}
    total = indexer.vector_store.count()
    assert total > 0 and len(indexer.bm25) == total  # Chroma and BM25 hold the same chunks
    assert (settings.uploads_dir / first.doc_id / "sample.pdf").is_file()

    # Same file again -> skipped, nothing added.
    assert indexer.index_file(sample_pdf).status == "duplicate"
    assert indexer.vector_store.count() == total

    docs = indexer.list_documents()
    assert len(docs) == 1 and docs[0]["file_name"] == "sample.pdf"

    # Delete -> gone from Chroma, BM25 and the uploads folder.
    assert indexer.delete_document(first.doc_id) == total
    assert indexer.vector_store.count() == 0 and len(indexer.bm25) == 0
    assert not (settings.uploads_dir / first.doc_id).exists()


def test_bad_file_is_reported_not_raised(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    result = make_indexer(make_settings(tmp_path / "data")).index_file(bad)
    assert result.status == "error" and "not a valid PDF" in result.message


def test_model_change_is_detected_and_rebuild_fixes_it(tmp_path, sample_pdf):
    indexer = make_indexer(make_settings(tmp_path, model="fake-model-a"))
    indexer.index_file(sample_pdf)
    total = indexer.vector_store.count()

    # Same data folder, different embedding model in the settings.
    indexer_b = make_indexer(make_settings(tmp_path, model="fake-model-b"))
    assert indexer_b.vector_store.model_mismatch
    assert indexer_b.index_file(sample_pdf).status == "error"

    results = indexer_b.rebuild_all()
    assert [r.status for r in results] == ["indexed"]
    assert indexer_b.vector_store.stored_model == "fake-model-b"
    assert not indexer_b.vector_store.model_mismatch
    assert indexer_b.vector_store.count() == total
