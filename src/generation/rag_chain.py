"""The RAG chain: rewrite question -> retrieve + re-rank -> generate -> map citations.

    follow-up question + last 4 turns --> standalone question
    standalone question --> HybridRetriever --> top-5 chunks labelled S1..S5
    chunks + question --> LLM (streamed token by token) --> "... [S2]"
    "[S2]" --> "[report.pdf, p. 12]"   (labels not in the context are removed)

Chat from the command line:
    python -m src.generation.rag_chain
    python -m src.generation.rag_chain --question "What is the maintenance interval?"
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.config import Settings, get_settings
from src.generation.llm import friendly_llm_error, get_llm
from src.generation.prompts import CONDENSE_PROMPT, NOT_FOUND_ANSWER, QA_PROMPT
from src.retrieval.hybrid_retriever import HybridRetriever, Mode, RetrievedChunk
from src.utils.logging import get_logger, set_console_level

logger = get_logger(__name__)

# A very long table could fill the whole context window, so each source is capped.
MAX_CHARS_PER_SOURCE = 4000

# One conversation turn = (user question, assistant answer).
Turn = tuple[str, str]


class GenerationError(Exception):
    """A readable problem with the language model (e.g. Ollama not running)."""


@dataclass
class Citation:
    """A source the answer actually cited."""

    label: str  # "S2"
    chunk: RetrievedChunk

    @property
    def file_name(self) -> str:
        return self.chunk.metadata["file_name"]

    @property
    def page(self) -> int:
        return self.chunk.metadata["page"]

    @property
    def source_type(self) -> str:
        return self.chunk.metadata["source_type"]


@dataclass
class Answer:
    """Everything the UI needs to show one answer."""

    text: str  # final answer with [file, p. N] citations
    raw_text: str  # what the model wrote, with [S1] labels
    question: str  # what the user typed
    standalone_question: str  # what was actually searched for
    citations: list[Citation] = field(default_factory=list)
    sources: list[RetrievedChunk] = field(default_factory=list)  # all chunks given to the model
    invalid_labels: list[str] = field(default_factory=list)  # cited but not in the context
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def not_found(self) -> bool:
        return self.text.strip().startswith(NOT_FOUND_ANSWER)


# ---------------------------------------------------------------------------
# Context building and citation mapping (plain functions, easy to test)
# ---------------------------------------------------------------------------


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, dict[str, RetrievedChunk]]:
    """Label each chunk S1, S2, ... and join them into the context text.

    Returns (context_text, {"S1": chunk, ...}). The label map is what lets us turn
    the model's [S1] back into a real file name and page number.
    """
    label_map: dict[str, RetrievedChunk] = {}
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"S{i}"
        label_map[label] = chunk
        m = chunk.metadata
        text = chunk.document.page_content
        if len(text) > MAX_CHARS_PER_SOURCE:
            text = text[:MAX_CHARS_PER_SOURCE] + " ..."
        parts.append(f"[{label}] ({m['file_name']}, p. {m['page']}, {m['source_type']})\n{text}")
    return "\n\n".join(parts), label_map


# [S1]  [S1, S3]  [s2; S4]  - one or more labels inside one pair of brackets
LABEL_GROUP = re.compile(r"\s*\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]", re.IGNORECASE)
# [report.pdf, p. 12] - in case the model ignores the rules and writes the page itself
DIRECT_CITATION = re.compile(r"\s*\[\s*([^\[\]]+?\.pdf)\s*,\s*p(?:age|\.)?\s*(\d+)\s*\]", re.IGNORECASE)


def map_citations(
    raw: str, label_map: dict[str, RetrievedChunk]
) -> tuple[str, list[Citation], list[str]]:
    """Replace [S1]-style labels with [file, p. N] and drop anything not in the context.

    Returns (final_text, citations_in_order_of_first_use, invalid_labels).

    - Valid labels become "[report.pdf, p. 12]". Two labels pointing at the same page
      are shown once.
    - A label that was not in the context (e.g. [S9] when only S1-S5 exist) is
      removed and logged, so no citation can point to a source the model never saw.
    - A citation the model wrote directly as [file.pdf, p. N] is kept only if that
      exact file and page were in the context.
    """
    citations: dict[str, Citation] = {}
    invalid: list[str] = []

    def replace_labels(match: re.Match) -> str:
        shown: list[str] = []
        for label in re.split(r"\s*[,;]\s*", match.group(1).upper()):
            chunk = label_map.get(label)
            if chunk is None:
                invalid.append(label)
                continue
            citations.setdefault(label, Citation(label, chunk))
            ref = f"{chunk.metadata['file_name']}, p. {chunk.metadata['page']}"
            if ref not in shown:
                shown.append(ref)
        # Keep the leading space only if something is left to show.
        return f" [{'; '.join(shown)}]" if shown else ""

    def check_direct(match: re.Match) -> str:
        file_name, page = match.group(1).strip(), int(match.group(2))
        for label, chunk in label_map.items():
            if chunk.metadata["file_name"].lower() == file_name.lower() and chunk.metadata["page"] == page:
                citations.setdefault(label, Citation(label, chunk))
                return f" [{chunk.metadata['file_name']}, p. {page}]"
        invalid.append(f"{file_name} p.{page}")
        return ""

    text = LABEL_GROUP.sub(replace_labels, raw)
    text = DIRECT_CITATION.sub(check_direct, text)
    if invalid:
        logger.warning("Removed citations not in the context: %s", invalid)
    return text, list(citations.values()), invalid


def _stream_safe_prefix(raw: str) -> str:
    """The part of a half-finished answer that is safe to map and show.

    While streaming, "[S" may have arrived without its closing "]". Everything
    from an unclosed "[" onwards is held back until the bracket closes.
    """
    last_open = raw.rfind("[")
    return raw[:last_open] if last_open > raw.rfind("]") else raw


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


class RAGChain:
    """Answers questions about the indexed documents, with verified citations."""

    def __init__(
        self,
        retriever: HybridRetriever,
        settings: Optional[Settings] = None,
        llm: Optional[BaseChatModel] = None,
        rewrite_llm: Optional[BaseChatModel] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever
        llm = llm or get_llm(self.settings, temperature=0.1)
        rewrite_llm = rewrite_llm or llm
        # LangChain "pipe" syntax: prompt -> model -> plain string.
        self.answer_chain = QA_PROMPT | llm | StrOutputParser()
        self.rewrite_chain = CONDENSE_PROMPT | rewrite_llm | StrOutputParser()

    def rewrite_question(self, question: str, history: list[Turn]) -> str:
        """Make a follow-up question self-contained using the recent conversation.

        With no history there is nothing to resolve, so no LLM call is made. If the
        rewrite fails or looks wrong, the original question is used.
        """
        recent = history[-self.settings.memory_turns :] if self.settings.memory_turns else []
        if not recent:
            return question
        # Citations and long answers add noise; the first 300 characters are enough.
        lines = []
        for user, assistant in recent:
            short = re.sub(r"\s*\[[^\[\]]*\]", "", assistant)[:300]
            lines.append(f"User: {user}\nAssistant: {short}")
        try:
            rewritten = self.rewrite_chain.invoke({"history": "\n".join(lines), "question": question})
        except Exception as exc:
            logger.warning("Question rewrite failed, using the original: %s", exc)
            return question
        rewritten = rewritten.strip().splitlines()[0].strip().strip('"') if rewritten.strip() else ""
        if not rewritten or len(rewritten) > 3 * len(question) + 100:
            return question  # empty or rambling: safer to keep the user's words
        logger.info("Rewrote %r -> %r", question, rewritten)
        return rewritten

    def ask(
        self,
        question: str,
        history: Optional[list[Turn]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        doc_ids: Optional[list[str]] = None,
        top_n: Optional[int] = None,
        mode: Mode = "hybrid_rerank",
    ) -> Answer:
        """Answer a question. on_token receives answer text as it streams in.

        Raises RetrievalError or GenerationError with a readable message.
        """
        history = history or []
        timings: dict[str, float] = {}

        t = time.perf_counter()
        standalone = self.rewrite_question(question, history)
        timings["rewrite_ms"] = (time.perf_counter() - t) * 1000

        retrieval = self.retriever.retrieve(standalone, mode=mode, top_n=top_n, doc_ids=doc_ids)
        timings["retrieval_ms"] = retrieval.timings_ms.get("total_ms", 0.0)
        chunks = retrieval.chunks

        if not chunks:  # nothing to answer from: don't ask the model to make something up
            if on_token:
                on_token(NOT_FOUND_ANSWER)
            return Answer(NOT_FOUND_ANSWER, NOT_FOUND_ANSWER, question, standalone, timings_ms=timings)

        context, label_map = build_context(chunks)
        raw, shown_len = "", 0
        t = time.perf_counter()
        try:
            for token in self.answer_chain.stream({"context": context, "question": standalone}):
                raw += token
                if on_token:
                    # Map citations in the finished part and send only the new text.
                    shown, _, _ = map_citations(_stream_safe_prefix(raw.lstrip()), label_map)
                    if len(shown) > shown_len:
                        on_token(shown[shown_len:])
                        shown_len = len(shown)
        except Exception as exc:
            logger.exception("Generation failed")
            raise GenerationError(friendly_llm_error(exc)) from exc
        timings["generation_ms"] = (time.perf_counter() - t) * 1000

        text, citations, invalid = map_citations(raw.strip(), label_map)
        if on_token and len(text) > shown_len:
            on_token(text[shown_len:])  # anything held back at the very end
        if not citations and NOT_FOUND_ANSWER not in text:
            logger.warning("The answer has no valid citations: %r", text[:200])

        return Answer(
            text=text,
            raw_text=raw,
            question=question,
            standalone_question=standalone,
            citations=citations,
            sources=chunks,
            invalid_labels=invalid,
            timings_ms=timings,
        )


# ---------------------------------------------------------------------------
# Command-line chat
# ---------------------------------------------------------------------------


def _print_answer_details(answer: Answer) -> None:
    if answer.standalone_question != answer.question:
        print(f"\n  (searched for: {answer.standalone_question})")
    if answer.citations:
        print("\n  Sources cited:")
        for c in answer.citations:
            score = c.chunk.rerank_score
            score_text = f", re-rank {score:.3f}" if score is not None else ""
            print(f"    [{c.label}] {c.file_name}, p. {c.page} ({c.source_type}{score_text})")
    if answer.invalid_labels:
        print(f"  Removed invalid citations: {answer.invalid_labels}")
    t = answer.timings_ms
    print(
        f"\n  Time: rewrite {t.get('rewrite_ms', 0):.0f} ms | retrieval {t.get('retrieval_ms', 0):.0f} ms | "
        f"generation {t.get('generation_ms', 0):.0f} ms\n"
    )


def main() -> None:
    from src.generation.llm import check_llm
    from src.indexing.indexer import Indexer
    from src.retrieval.hybrid_retriever import RetrievalError

    parser = argparse.ArgumentParser(description="Chat with your indexed PDFs.")
    parser.add_argument("--question", "-q", help="Ask one question and exit")
    parser.add_argument("--doc", action="append", metavar="DOC_ID", help="Only use this document (repeatable)")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")
    set_console_level("WARNING")  # keep INFO logs out of the streamed answer (still in app.log)

    problem = check_llm()
    if problem:
        print(f"ERROR: {problem}")
        sys.exit(1)

    indexer = Indexer()
    if indexer.vector_store.count() == 0:
        print("Nothing is indexed yet. Run: python -m src.indexing.indexer <folder of PDFs>")
        sys.exit(1)
    retriever = HybridRetriever(indexer.vector_store, indexer.bm25)
    print("Loading models...")
    retriever.warm_up()
    chain = RAGChain(retriever)

    def ask(question: str, history: list[Turn]) -> Optional[Answer]:
        print("\nAssistant: ", end="", flush=True)
        try:
            answer = chain.ask(question, history, on_token=lambda s: print(s, end="", flush=True), doc_ids=args.doc)
        except (RetrievalError, GenerationError) as exc:
            print(f"\nERROR: {exc}\n")
            return None
        print()
        _print_answer_details(answer)
        return answer

    if args.question:
        ask(args.question, [])
        return

    print("\nAsk about your documents. Commands: /clear (forget the conversation), /exit\n")
    history: list[Turn] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("/exit", "/quit"):
            break
        if question.lower() == "/clear":
            history.clear()
            print("Conversation cleared.\n")
            continue
        answer = ask(question, history)
        if answer:
            history.append((question, answer.text))
            keep = get_settings().memory_turns  # keep only the last N turns
            history[:] = history[-keep:] if keep else []


if __name__ == "__main__":
    main()
