"""Phase 5 tests: context labels, citation mapping/validation, streaming, question rewrite.

A fake chat model with fixed replies is used, so no Ollama is needed.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.config import Settings
from src.generation.prompts import NOT_FOUND_ANSWER
from src.generation.rag_chain import GenerationError, RAGChain, build_context, map_citations
from src.retrieval.hybrid_retriever import RetrievalResult, RetrievedChunk


def chunk(file_name: str, page: int, text: str, source_type: str = "text") -> RetrievedChunk:
    meta = {"doc_id": "d", "file_name": file_name, "page": page, "source_type": source_type,
            "chunk_id": f"{file_name}:{page}:{text[:5]}"}
    return RetrievedChunk(document=Document(text, metadata=meta), rerank_score=0.9)


CHUNKS = [
    chunk("manual.pdf", 3, "Replace the filter every 6 months."),
    chunk("report.pdf", 12, "| Quarter | Revenue |\n| Q2 | 140 |", "table"),
    chunk("manual.pdf", 3, "The filter is behind the front panel."),
]


# ---------- context and citation mapping ----------


def test_context_labels():
    context, labels = build_context(CHUNKS)
    assert list(labels) == ["S1", "S2", "S3"]
    assert context.startswith("[S1] (manual.pdf, p. 3, text)\nReplace the filter")
    assert "[S2] (report.pdf, p. 12, table)" in context


def test_labels_are_mapped_to_file_and_page():
    _, labels = build_context(CHUNKS)
    text, cites, invalid = map_citations("Replace it every 6 months [S1]. Q2 revenue was 140 [S2].", labels)
    assert text == "Replace it every 6 months [manual.pdf, p. 3]. Q2 revenue was 140 [report.pdf, p. 12]."
    assert [c.label for c in cites] == ["S1", "S2"] and invalid == []
    assert cites[1].page == 12 and cites[1].source_type == "table"


def test_invalid_labels_are_removed():
    _, labels = build_context(CHUNKS)
    text, cites, invalid = map_citations("It is behind the panel [S9]. Every 6 months [S1, S7].", labels)
    assert text == "It is behind the panel. Every 6 months [manual.pdf, p. 3]."
    assert invalid == ["S9", "S7"] and [c.label for c in cites] == ["S1"]


def test_same_page_shown_once_and_lowercase_labels():
    _, labels = build_context(CHUNKS)
    text, cites, _ = map_citations("The filter is replaced twice a year [s1; S3].", labels)
    assert text == "The filter is replaced twice a year [manual.pdf, p. 3]."
    assert [c.label for c in cites] == ["S1", "S3"]  # both chunks still listed as sources


def test_direct_citations_checked_against_context():
    _, labels = build_context(CHUNKS)
    text, cites, invalid = map_citations("A [report.pdf, p. 12]. B [report.pdf, p. 99].", labels)
    assert text == "A [report.pdf, p. 12]. B."
    assert invalid == ["report.pdf p.99"] and cites[0].label == "S2"


def test_other_brackets_are_left_alone():
    _, labels = build_context(CHUNKS)
    text, _, invalid = map_citations("See note [a] and step [2].", labels)
    assert text == "See note [a] and step [2]." and invalid == []


# ---------- the chain with a fake model and fake retriever ----------


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.queries = []

    def retrieve(self, query, mode="hybrid_rerank", top_n=None, doc_ids=None, **_):
        self.queries.append(query)
        return RetrievalResult(chunks=list(self.chunks), timings_ms={"total_ms": 5.0})


def make_chain(replies, chunks=CHUNKS, rewrite_replies=None):
    settings = Settings(_env_file=None, memory_turns=4)
    retriever = FakeRetriever(chunks)
    llm = FakeListChatModel(responses=replies)
    rewrite = FakeListChatModel(responses=rewrite_replies) if rewrite_replies else llm
    return RAGChain(retriever, settings=settings, llm=llm, rewrite_llm=rewrite), retriever


def test_streamed_text_matches_final_answer():
    chain, _ = make_chain(["The filter is replaced every 6 months [S1]. Revenue was 140 [S2][S8]."])
    streamed = []
    answer = chain.ask("How often is the filter replaced?", on_token=streamed.append)
    assert "".join(streamed) == answer.text
    assert answer.text == "The filter is replaced every 6 months [manual.pdf, p. 3]. Revenue was 140 [report.pdf, p. 12]."
    assert "[S" not in "".join(streamed)  # raw labels never reach the user
    assert answer.invalid_labels == ["S8"]
    assert {"rewrite_ms", "retrieval_ms", "generation_ms"} <= answer.timings_ms.keys()


def test_no_chunks_means_not_found_without_llm():
    chain, _ = make_chain(["this should never be used"], chunks=[])
    answer = chain.ask("Anything?")
    assert answer.text == NOT_FOUND_ANSWER and answer.not_found


def test_first_question_is_not_rewritten():
    chain, retriever = make_chain(["Answer [S1]."], rewrite_replies=["SHOULD NOT BE USED"])
    chain.ask("What is the filter interval?")
    assert retriever.queries == ["What is the filter interval?"]


def test_follow_up_is_rewritten_before_retrieval():
    chain, retriever = make_chain(["Answer [S2]."], rewrite_replies=['"What was the revenue in Q3?"\nextra'])
    history = [("What was the revenue in Q2?", "It was 140 [report.pdf, p. 12].")]
    answer = chain.ask("and in Q3?", history=history)
    assert retriever.queries == ["What was the revenue in Q3?"]
    assert answer.standalone_question == "What was the revenue in Q3?"


def test_llm_failure_gives_readable_error():
    class BrokenModel(FakeListChatModel):
        def _stream(self, *args, **kwargs):
            raise ConnectionError("Connection refused")

    chain = RAGChain(FakeRetriever(CHUNKS), settings=Settings(_env_file=None), llm=BrokenModel(responses=["x"]))
    with pytest.raises(GenerationError, match="Ollama"):
        chain.ask("Question?")


def test_openai_errors_are_explained():
    from src.generation.llm import friendly_llm_error

    no_credit = Exception("Error code: 429 - You have no credits remaining. Add credits to continue")
    assert "no credits left" in friendly_llm_error(no_credit)
    assert "rate limit" in friendly_llm_error(Exception("Error code: 429 - Rate limit reached"))
    assert "API key was rejected" in friendly_llm_error(Exception("Error code: 401 - Incorrect API key"))
