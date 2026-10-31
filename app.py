"""Streamlit app: upload PDFs, then chat with them. Every answer cites file and page.

Run with:  streamlit run app.py

Layout
  Sidebar : provider + models, retrieval sliders, chunk size, document list
            (delete / filter), clear chat, system status
  Main    : PDF uploader with progress, chat with streamed answers, a
            "source" expander under each answer, and response times

Streamlit re-runs this whole file on every click. Heavy objects (embedding model,
re-ranker, Chroma client) are created once and kept with @st.cache_resource.
"""

import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
from pydantic import SecretStr

from src.config import Settings, get_settings
from src.generation.llm import check_llm, check_vision, list_ollama_models
from src.generation.rag_chain import Answer, GenerationError, RAGChain, Turn
from src.indexing.indexer import Indexer, IndexResult
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalError, RetrievedChunk
from src.retrieval.reranker import Reranker
from src.utils.logging import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Inkwell RAG", page_icon="📄", layout="centered")


# ---------------------------------------------------------------------------
# Cached resources (loaded once per server, shared by all browser tabs)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the embedding model and the index...")
def load_indexer() -> Indexer:
    indexer = Indexer(get_settings())
    _ = indexer.vector_store.embeddings  # load the model now, not on the first question
    return indexer


@st.cache_resource(show_spinner="Loading the re-ranker...")
def load_reranker() -> Reranker:
    reranker = Reranker(get_settings())
    reranker.load()
    return reranker


@st.cache_data(ttl=30, show_spinner=False)
def ollama_models(base_url: str) -> tuple[bool, list[str]]:
    """Pulled Ollama models, re-checked at most every 30 seconds."""
    return list_ollama_models(base_url)


@st.cache_data(ttl=300, show_spinner=False)
def tesseract_version(cmd: Optional[str]) -> Optional[str]:
    import pytesseract

    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def md_safe(text: str) -> str:
    """Stop Streamlit reading "$5 and $6" as a maths formula."""
    return text.replace("$", "\\$")


def init_state() -> None:
    st.session_state.setdefault("messages", [])  # chat history shown on screen
    st.session_state.setdefault("uploader_key", 0)  # changing it clears the uploader
    st.session_state.setdefault("index_results", [])  # results of the last upload


def chat_history(memory_turns: int) -> list[Turn]:
    """(question, answer) pairs of the last few successful turns, for follow-up questions."""
    turns, pending = [], None
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            pending = msg["content"]
        elif pending is not None and not msg.get("error"):
            turns.append((pending, msg["content"]))
            pending = None
    return turns[-memory_turns:] if memory_turns else []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def model_picker(label: str, current: str, pulled: list[str], key: str) -> str:
    """A dropdown of pulled Ollama models, or a text box if Ollama can't be reached."""
    if not pulled:
        return st.sidebar.text_input(label, current, key=key)
    options = sorted(set(pulled) | {current})
    return st.sidebar.selectbox(label, options, index=options.index(current), key=key)


def sidebar_settings(base: Settings) -> Settings:
    """Read the sidebar controls and return a copy of the settings with those values."""
    sb = st.sidebar
    sb.header("⚙️ Settings")

    provider = sb.radio(
        "Provider", ["ollama", "openai"], index=0 if base.provider == "ollama" else 1, horizontal=True,
        format_func=lambda p: "Ollama (local)" if p == "ollama" else "OpenAI (cloud)",
    )
    update: dict = {"provider": provider}
    if provider == "ollama":
        _, pulled = ollama_models(base.ollama_base_url)
        names = [n.removesuffix(":latest") for n in pulled]
        update["ollama_llm_model"] = model_picker("Chat model", base.ollama_llm_model, names, "llm_model")
        update["ollama_vision_model"] = model_picker("Vision model", base.ollama_vision_model, names, "vision_model")
    else:
        update["openai_llm_model"] = sb.text_input("Chat model", base.openai_llm_model)
        update["openai_vision_model"] = sb.text_input("Vision model", base.openai_vision_model)
        key = sb.text_input(
            "OpenAI API key", type="password",
            help="Kept only in this browser session, never saved. Leave empty to use OPENAI_API_KEY from .env.",
        )
        if key.strip():
            update["openai_api_key"] = SecretStr(key.strip())

    sb.subheader("Retrieval")
    top_k = sb.slider("Candidates per search (top-k)", 5, 50, base.dense_top_k, step=5,
                      help="How many chunks dense search and BM25 each return before fusion.")
    top_n = sb.slider("Sources given to the LLM (top-n)", 1, 10, base.rerank_top_n,
                      help="How many chunks are kept after re-ranking.")
    update.update(dense_top_k=top_k, bm25_top_k=top_k, rerank_top_n=top_n)

    sb.subheader("Chunking")
    col1, col2 = sb.columns(2)
    size = col1.number_input("Chunk size", 200, 3000, base.chunk_size, step=50)
    overlap = col2.number_input("Overlap", 0, 1000, base.chunk_overlap, step=25)
    sb.caption("Applies to documents indexed from now on.")
    update.update(chunk_size=int(size), chunk_overlap=int(overlap))

    return base.model_copy(update=update)


def sidebar_documents(indexer: Indexer) -> list[str]:
    """Document list with delete buttons. Returns the doc_ids chosen for filtering."""
    sb = st.sidebar
    docs = indexer.list_documents()
    sb.subheader(f"📚 Documents ({len(docs)})")
    if not docs:
        sb.caption("No documents yet. Upload PDFs on the main page.")
        return []

    for d in docs:
        types = ", ".join(f"{n} {t}" for t, n in sorted(d["types"].items()))
        col1, col2 = sb.columns([5, 1], vertical_alignment="center")
        col1.markdown(f"**{d['file_name']}**  \n<small>{d['chunks']} chunks: {types}</small>", unsafe_allow_html=True)
        with col2.popover("🗑️"):
            st.markdown(f"Delete **{d['file_name']}** from the index?")
            if st.button("Delete", key=f"delete_{d['doc_id']}", type="primary"):
                removed = indexer.delete_document(d["doc_id"])
                st.toast(f"Deleted {d['file_name']} ({removed} chunks)")
                st.rerun()

    names = {d["doc_id"]: d["file_name"] for d in docs}
    return sb.multiselect(
        "Search only in", options=list(names), format_func=names.get,
        placeholder="All documents", help="Leave empty to search every document.",
    )


def sidebar_status(settings: Settings, indexer: Indexer, llm_problem: Optional[str], vision_problem: Optional[str]) -> None:
    with st.sidebar.expander("🩺 System status"):
        st.markdown(("✅ " if not llm_problem else "❌ ") + f"Chat model `{settings.active_llm_model}`")
        if llm_problem:
            st.caption(llm_problem)
        st.markdown(("✅ " if not vision_problem else "⚠️ ") + f"Vision model `{settings.active_vision_model}`")
        if vision_problem:
            st.caption(vision_problem)
        version = tesseract_version(settings.tesseract_cmd)
        st.markdown(f"✅ Tesseract {version}" if version else "⚠️ Tesseract not found (scanned pages skipped)")
        st.markdown(f"✅ Embeddings `{indexer.vector_store.model_name}`")
        st.markdown(f"✅ Re-ranker `{load_reranker().model_name}`")
        st.caption(f"Chroma chunks: {indexer.vector_store.count()} · BM25 chunks: {len(indexer.bm25)}")


# ---------------------------------------------------------------------------
# Upload and indexing
# ---------------------------------------------------------------------------


def describe_result(r: IndexResult) -> str:
    c = r.counts
    parts = [f"{r.pages} pages", f"{c.get('text', 0)} text chunks", f"{c.get('table', 0)} tables",
             f"{c.get('figure', 0)} figures"]
    if c.get("ocr"):
        parts.append(f"{c['ocr']} OCR chunks")
    return ", ".join(parts)


def upload_section(
    indexer: Indexer, settings: Settings, has_docs: bool, llm_problem: Optional[str], vision_problem: Optional[str]
) -> None:
    with st.expander("➕ Add PDF documents", expanded=not has_docs):
        files = st.file_uploader(
            "PDF files", type=["pdf"], accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}", label_visibility="collapsed",
        )
        problem = settings.provider_problem()
        if problem and problem != llm_problem:  # the same message is already shown below
            st.error(problem)
        if files and st.button(f"Index {len(files)} file(s)", type="primary", disabled=bool(problem)):
            index_uploads(indexer, settings, files, llm_problem, vision_problem)

    # Results of the last upload stay visible after the page refreshes.
    for r in st.session_state.index_results:
        if r.status == "indexed":
            st.success(f"**{r.file_name}** indexed: {describe_result(r)}.")
        elif r.status == "duplicate":
            st.info(f"**{r.file_name}** is already indexed, skipped.")
        else:
            st.error(f"**{r.file_name}**: {r.message}")


def index_uploads(
    indexer: Indexer, settings: Settings, files, llm_problem: Optional[str], vision_problem: Optional[str]
) -> None:
    """Save each upload to a temp folder, index it with a progress bar, then refresh."""
    if llm_problem and settings.llm_table_summaries:
        st.warning("The chat model is not available, so tables get a simple header-based summary.")
    if settings.caption_figures and vision_problem:
        st.warning("The vision model is not available, so figures are indexed by their printed caption only.")

    indexer.settings = settings  # use the sidebar's chunk size and provider for this upload
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for upload in files:
            # Keep the original file name: it is what citations show.
            path = Path(tmp) / Path(upload.name).name
            path.write_bytes(upload.getvalue())
            bar = st.progress(0.0, text=f"{upload.name}: starting...")
            try:
                result = indexer.index_file(
                    path, progress=lambda msg, frac, name=upload.name: bar.progress(min(frac, 1.0), text=f"{name}: {msg}")
                )
            except Exception as exc:  # index_file reports problems itself; this is a last resort
                logger.exception("Unexpected error indexing %s", upload.name)
                result = IndexResult(upload.name, "error", message=f"Unexpected error: {exc}")
            bar.empty()
            results.append(result)

    st.session_state.index_results = results
    st.session_state.uploader_key += 1  # clear the uploader
    st.rerun()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def fmt(value: Optional[float], digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def show_source(label: str, chunk: RetrievedChunk, settings: Settings) -> None:
    """One expander per cited source: file, page, type, scores, text and figure image."""
    m = chunk.metadata
    icon = {"table": "📊", "figure": "🖼️", "ocr": "🔎"}.get(m["source_type"], "📄")
    title = f"{icon} [{label}] {m['file_name']} · p. {m['page']} · {m['source_type']} · re-rank {fmt(chunk.rerank_score)}"
    with st.expander(title):
        st.caption(
            f"Re-rank score {fmt(chunk.rerank_score)} · dense similarity {fmt(chunk.dense_score)} "
            f"(rank {chunk.dense_rank or '–'}) · BM25 {fmt(chunk.bm25_score, 2)} (rank {chunk.bm25_rank or '–'}) "
            f"· fused rank {chunk.fused_rank or '–'}"
        )
        image_path = m.get("image_path")
        if image_path and settings.data_file(image_path).is_file():
            st.image(str(settings.data_file(image_path)))
        if m["source_type"] in ("table", "figure"):
            st.markdown(md_safe(chunk.document.page_content))  # tables render as real tables
        else:
            st.text(chunk.document.page_content)


def show_answer_details(answer: Answer, settings: Settings) -> None:
    if answer.standalone_question != answer.question:
        st.caption(f"🔁 Searched for: *{md_safe(answer.standalone_question)}*")
    for citation in answer.citations:
        show_source(citation.label, citation.chunk, settings)
    t = answer.timings_ms
    timing = f"⏱️ Retrieval {t.get('retrieval_ms', 0) / 1000:.2f} s · Generation {t.get('generation_ms', 0) / 1000:.2f} s"
    if t.get("rewrite_ms", 0) > 1:
        timing += f" · Question rewrite {t['rewrite_ms'] / 1000:.2f} s"
    st.caption(timing)


def show_history(settings: Settings) -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("error"):
                st.error(msg["content"])
                continue
            st.markdown(md_safe(msg["content"]))
            if msg.get("answer"):
                show_answer_details(msg["answer"], settings)


def answer_question(question: str, retriever: HybridRetriever, settings: Settings, doc_ids: list[str]) -> None:
    """Show the question, stream the answer, then show sources and timings."""
    history = chat_history(settings.memory_turns)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md_safe(question))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("*Searching your documents...*")
        streamed: list[str] = []

        def on_token(text: str) -> None:
            streamed.append(text)
            placeholder.markdown(md_safe("".join(streamed)) + " ▌")

        try:
            chain = RAGChain(retriever, settings)
            answer = chain.ask(question, history, on_token=on_token, doc_ids=doc_ids or None)
        except (RetrievalError, GenerationError, ValueError) as exc:
            placeholder.empty()
            st.error(str(exc))
            st.session_state.messages.append({"role": "assistant", "content": str(exc), "error": True})
            return
        except Exception as exc:
            logger.exception("Unexpected error answering %r", question)
            placeholder.empty()
            message = f"Something went wrong: {exc}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message, "error": True})
            return

        placeholder.markdown(md_safe(answer.text))
        show_answer_details(answer, settings)
        st.session_state.messages.append({"role": "assistant", "content": answer.text, "answer": answer})


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    init_state()
    base = get_settings()

    st.title("📄 Inkwell RAG")
    st.caption("Ask questions about your PDFs: text, tables, figures and scanned pages. Every answer cites its page.")

    try:
        indexer = load_indexer()
        reranker = load_reranker()
    except Exception as exc:
        logger.exception("Startup failed")
        st.error(f"Could not load the models: {exc}. The first run needs internet to download them.")
        st.stop()

    # Documents added from the command line while the app runs: refresh the BM25 index.
    if len(indexer.bm25) != indexer.vector_store.count():
        indexer.rebuild_bm25()

    settings = sidebar_settings(base)
    doc_ids = sidebar_documents(indexer)
    st.sidebar.divider()
    if st.sidebar.button("🧹 Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    # Check the models once per page refresh (each check is a call to Ollama).
    llm_problem = check_llm(settings) or settings.provider_problem()
    vision_problem = check_vision(settings) if settings.caption_figures else "figure captions are turned off"
    sidebar_status(settings, indexer, llm_problem, vision_problem)

    # The index was built with another embedding model: searching it would fail.
    store = indexer.vector_store
    if store.model_mismatch:
        st.warning(store.mismatch_message())
        if st.button("Rebuild the index now", type="primary"):
            bar = st.progress(0.0, text="Rebuilding...")
            results = indexer.rebuild_all(progress=lambda msg, frac: bar.progress(min(frac, 1.0), text=msg))
            st.session_state.index_results = results
            st.rerun()
        st.stop()

    has_docs = store.count() > 0
    upload_section(indexer, settings, has_docs, llm_problem, vision_problem)

    if llm_problem:
        st.error(llm_problem)

    show_history(settings)

    if not has_docs:
        hint = "Upload a PDF to start chatting"
    elif llm_problem:
        hint = "Fix the problem above to start chatting"
    else:
        hint = "Ask a question about your documents..."
    question = st.chat_input(hint, disabled=not has_docs or bool(llm_problem))

    if question is not None:
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            retriever = HybridRetriever(store, indexer.bm25, reranker=reranker, settings=settings)
            answer_question(question.strip(), retriever, settings, doc_ids)


main()
