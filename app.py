"""Inkwell: upload PDFs, then chat with them. Every answer cites file and page.

Run with:  streamlit run app.py

Layout
  Sidebar : provider (Ollama / OpenAI), models, answer mode (fast / accurate re-ranker),
            advanced settings (top-k, top-n, chunk size), document library
            (delete / search only in), clear chat, system status
  Main    : header with live status, library stats, uploader with progress, chat with
            streamed answers, citation chips, one card per cited source, response times

Speed
  - The page appears immediately; the embedding model, the re-ranker and the Ollama
    chat model are loaded in a background thread (they take ~20 s together).
  - Models are created once per server with @st.cache_resource.
  - Ollama health checks are cached for 15 s instead of running on every click.

Look: theme colours/fonts in .streamlit/config.toml, extra CSS in assets/style.css.
"""

import html
import re
import tempfile
import threading
from pathlib import Path
from typing import Optional

import streamlit as st
from pydantic import SecretStr

from src.config import Settings, get_settings
from src.generation.llm import check_llm, check_vision, list_ollama_models, preload_ollama_model
from src.generation.rag_chain import Answer, GenerationError, RAGChain, Turn
from src.indexing.indexer import Indexer, IndexResult
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalError, RetrievedChunk
from src.retrieval.reranker import Reranker
from src.utils.logging import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Inkwell · Ask your documents", page_icon="🖋️", layout="centered")

ASSETS = Path(__file__).resolve().parent / "assets"
ASSISTANT_AVATAR = ":material/history_edu:"  # a quill: the "ink" in Inkwell

# Example questions shown for the bundled sample PDFs (evaluation/sample_docs).
SAMPLE_QUESTIONS = {
    "hydroflow_pump_manual.pdf": ["What is the maximum flow rate of the HF-200 pump?",
                                  "What should I do if the pump does not deliver water?"],
    "northwind_annual_report.pdf": ["Which business segment had the highest revenue in 2024?",
                                    "What dividend does the Board propose?"],
    "grid_storage_whitepaper.pdf": ["What does the capacity retention chart show?",
                                    "Which battery chemistry has the lowest cell cost?"],
}
GENERIC_QUESTIONS = ["What is this document about?", "Summarise the main conclusions.",
                     "What do the tables show?"]

LOGO_SVG = """<svg width="52" height="52" viewBox="0 0 48 48" aria-hidden="true">
<defs><linearGradient id="inkg" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="#3B6CB5"/><stop offset="1" stop-color="#16233F"/></linearGradient></defs>
<path d="M24 3C24 3 9 20 9 30.5a15 15 0 0 0 30 0C39 20 24 3 24 3z" fill="url(#inkg)"/>
<path d="M17 31.5a7 7 0 0 0 7 7" stroke="#fff" stroke-opacity=".6" stroke-width="2.6" fill="none" stroke-linecap="round"/>
</svg>"""


# ---------------------------------------------------------------------------
# Cached resources and background loading
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Opening your library...")
def load_indexer() -> Indexer:
    """Chroma + BM25. The embedding model itself is loaded in the background (see Warmup)."""
    return Indexer(get_settings())


@st.cache_resource(show_spinner=False)
def get_reranker(model_name: str) -> Reranker:
    """One re-ranker per model name ("accurate" and "fast" can both be cached)."""
    return Reranker(get_settings().model_copy(update={"reranker_model": model_name}))


class Warmup:
    """Loads the slow models in a background thread so the page can show at once.

    Only plain attributes are set here: Streamlit functions must not be called from
    another thread. The page reads `steps` to show progress.
    """

    def __init__(self, indexer: Indexer, reranker: Reranker, settings: Settings) -> None:
        self.steps = {"Embeddings": False, "Re-ranker": False, "Chat model": settings.provider != "ollama"}
        self.error: Optional[str] = None
        threading.Thread(target=self._run, args=(indexer, reranker, settings), daemon=True).start()

    def _run(self, indexer: Indexer, reranker: Reranker, settings: Settings) -> None:
        try:
            _ = indexer.vector_store.embeddings
            self.steps["Embeddings"] = True
            reranker.load()
            self.steps["Re-ranker"] = True
            if settings.provider == "ollama":
                preload_ollama_model(settings)  # failure is fine: the health check reports it
        except Exception as exc:  # e.g. no internet on the very first run
            logger.exception("Background model loading failed")
            self.error = str(exc)
        finally:
            self.steps = {name: True for name in self.steps}

    @property
    def ready(self) -> bool:
        return all(self.steps.values())

    @property
    def progress(self) -> str:
        return f"{sum(self.steps.values())}/{len(self.steps)}"


@st.cache_resource(show_spinner=False)
def start_warmup() -> Warmup:
    settings = get_settings()
    return Warmup(load_indexer(), get_reranker(settings.reranker_model), settings)


@st.cache_data(ttl=15, show_spinner=False)
def model_health(_settings: Settings, cache_key: tuple) -> tuple[Optional[str], Optional[str]]:
    """(chat model problem, vision model problem), re-checked at most every 15 s.

    cache_key holds the settings that matter; _settings itself is not hashed.
    """
    llm = check_llm(_settings) or _settings.provider_problem()
    vision = check_vision(_settings) if _settings.caption_figures else "figure descriptions are turned off"
    return llm, vision


@st.cache_data(ttl=30, show_spinner=False)
def ollama_models(base_url: str) -> tuple[bool, list[str]]:
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
# HTML helpers
# ---------------------------------------------------------------------------


def load_css() -> None:
    st.markdown(f"<style>{(ASSETS / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def md_safe(text: str) -> str:
    """Stop Streamlit reading "$5 and $6" as a maths formula."""
    return text.replace("$", "\\$")


# One citation group as written by the RAG chain: [file.pdf, p. 3] or [a.pdf, p. 2; b.pdf, p. 5]
CITATION_GROUP = re.compile(r"\[((?:[^\[\]]+?\.pdf, p\. \d+)(?:; [^\[\]]+?\.pdf, p\. \d+)*)\]", re.IGNORECASE)


def _short(name: str, limit: int = 26) -> str:
    stem = name[:-4] if name.lower().endswith(".pdf") else name
    return stem if len(stem) <= limit else stem[: limit - 1] + "…"


def answer_html(text: str) -> str:
    """Answer text with each [file.pdf, p. N] turned into a small citation chip.

    The model's text is HTML-escaped first, so it cannot inject markup.
    """
    safe = md_safe(html.escape(text, quote=False))

    def chips(match: re.Match) -> str:
        out = []
        for ref in match.group(1).split("; "):
            file_name, page = ref.rsplit(", p. ", 1)
            out.append(f'<span class="cite" title="{file_name}, page {page}">{_short(file_name)} · p.{page}</span>')
        return "".join(out)

    return CITATION_GROUP.sub(chips, safe)


def pill(text: str, state: str) -> str:
    return f'<span class="ink-pill"><span class="ink-dot {state}"></span>{html.escape(text)}</span>'


def render_hero(settings: Settings, llm_problem: Optional[str], warmup: Warmup, n_docs: int) -> None:
    if settings.provider == "ollama":
        provider = pill(f"Offline · {settings.ollama_llm_model}", "bad" if llm_problem else "ok")
    else:
        provider = pill(f"Cloud · {settings.openai_llm_model}", "bad" if llm_problem else "ok")
    models = pill("Models ready", "ok") if warmup.ready else pill(f"Warming up models {warmup.progress}", "warm")
    library = pill(f"{n_docs} document{'s' if n_docs != 1 else ''}", "ok" if n_docs else "warm")
    st.markdown(
        f"""<div class="ink-hero">{LOGO_SVG}<div>
        <h1>Ink<em>well</em></h1>
        <p class="ink-tagline">Ask your PDFs anything: text, tables, charts and scanned pages.
        Every answer is signed with its page.</p>
        <div class="ink-pills">{provider}{models}{library}</div></div></div>""",
        unsafe_allow_html=True,
    )


def render_stats(docs: list[dict]) -> None:
    pages = sum(d["pages"] for d in docs)
    tables = sum(d["types"].get("table", 0) for d in docs)
    figures = sum(d["types"].get("figure", 0) for d in docs)
    cells = [(len(docs), "Documents"), (pages, "Pages"), (tables, "Tables"), (figures, "Figures")]
    st.markdown(
        '<div class="ink-stats">'
        + "".join(f'<div class="ink-stat"><b>{n}</b><span>{label}</span></div>' for n, label in cells)
        + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_state() -> None:
    st.session_state.setdefault("messages", [])  # chat history shown on screen
    st.session_state.setdefault("uploader_key", 0)  # changing it clears the uploader
    st.session_state.setdefault("index_results", [])  # results of the last upload
    st.session_state.setdefault("pending_question", None)  # a clicked example question


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


def side_label(text: str) -> None:
    st.sidebar.markdown(f'<div class="side-label">{text}</div>', unsafe_allow_html=True)


def model_picker(label: str, current: str, pulled: list[str], key: str) -> str:
    """A dropdown of pulled Ollama models, or a text box if Ollama can't be reached."""
    if not pulled:
        return st.sidebar.text_input(label, current, key=key)
    options = sorted(set(pulled) | {current})
    return st.sidebar.selectbox(label, options, index=options.index(current), key=key)


def sidebar_settings(base: Settings) -> Settings:
    """Read the sidebar controls and return a copy of the settings with those values."""
    sb = st.sidebar
    sb.markdown(f'<div class="side-brand">🖋️<div>Inkwell<small>MULTIMODAL RAG</small></div></div>',
                unsafe_allow_html=True)

    side_label("Model")
    provider = sb.segmented_control(
        "Provider", ["ollama", "openai"], default=base.provider, key="provider",
        format_func=lambda p: "Local · Ollama" if p == "ollama" else "Cloud · OpenAI",
        label_visibility="collapsed",
    ) or base.provider  # clicking the selected option again would clear it
    update: dict = {"provider": provider}
    if provider == "ollama":
        _, pulled = ollama_models(base.ollama_base_url)
        names = [n.removesuffix(":latest") for n in pulled]
        update["ollama_llm_model"] = model_picker("Chat model", base.ollama_llm_model, names, "llm_model")
        update["ollama_vision_model"] = model_picker("Vision model", base.ollama_vision_model, names, "vision_model")
    else:
        update["openai_llm_model"] = sb.text_input("Chat model", base.openai_llm_model)
        update["openai_vision_model"] = sb.text_input("Vision model", base.openai_vision_model)
        key = sb.text_input("OpenAI API key", type="password",
                            help="Kept only in this browser session, never saved. Leave empty to use OPENAI_API_KEY from .env.")
        if key.strip():
            update["openai_api_key"] = SecretStr(key.strip())

    side_label("Answer mode")
    accurate, fast = base.reranker_model, base.reranker_fallback_model
    mode = sb.segmented_control(
        "Answer mode", ["accurate", "fast"], default="accurate" if accurate != fast else "fast", key="answer_mode",
        format_func=lambda m: "Accurate" if m == "accurate" else "Fast",
        label_visibility="collapsed",
        help="Accurate uses the bge-reranker-base cross-encoder. Fast uses the smaller MiniLM "
             "re-ranker: several seconds quicker per question on a laptop CPU.",
    ) or "accurate"
    update["reranker_model"] = accurate if mode == "accurate" else fast

    with sb.expander("Advanced settings", icon=":material/tune:"):
        top_k = st.slider("Candidates per search (top-k)", 5, 50, base.dense_top_k, step=5,
                          help="How many chunks dense search and BM25 each return before fusion.")
        top_n = st.slider("Sources given to the LLM (top-n)", 1, 10, base.rerank_top_n,
                          help="How many chunks are kept after re-ranking.")
        col1, col2 = st.columns(2)
        size = col1.number_input("Chunk size", 200, 3000, base.chunk_size, step=50)
        overlap = col2.number_input("Overlap", 0, 1000, base.chunk_overlap, step=25)
        st.caption("Chunk settings apply to documents indexed from now on.")
    update.update(dense_top_k=top_k, bm25_top_k=top_k, rerank_top_n=top_n,
                  chunk_size=int(size), chunk_overlap=int(overlap))
    return base.model_copy(update=update)


def sidebar_documents(docs: list[dict], indexer: Indexer) -> list[str]:
    """Library with delete buttons. Returns the doc_ids chosen for filtering."""
    side_label(f"Library · {len(docs)}")
    if not docs:
        st.sidebar.caption("No documents yet.")
        return []
    for d in docs:
        types = " · ".join(f"{n} {t}" for t, n in sorted(d["types"].items()))
        col1, col2 = st.sidebar.columns([6, 1], vertical_alignment="center")
        col1.markdown(f'<div class="side-doc">{html.escape(d["file_name"])}<br><small>{types}</small></div>',
                      unsafe_allow_html=True)
        with col2.popover("", icon=":material/delete:"):
            st.markdown(f"Remove **{d['file_name']}** from the library?")
            if st.button("Delete", key=f"delete_{d['doc_id']}", type="primary"):
                removed = indexer.delete_document(d["doc_id"])
                st.toast(f"Deleted {d['file_name']} ({removed} chunks)", icon=":material/delete:")
                st.rerun()
    names = {d["doc_id"]: d["file_name"] for d in docs}
    return st.sidebar.multiselect("Search only in", options=list(names), format_func=names.get,
                                  placeholder="All documents")


def sidebar_status(settings: Settings, indexer: Indexer, reranker: Reranker,
                   llm_problem: Optional[str], vision_problem: Optional[str]) -> None:
    with st.sidebar.expander("System status", icon=":material/monitor_heart:"):
        st.markdown(("✅ " if not llm_problem else "❌ ") + f"Chat model `{settings.active_llm_model}`")
        if llm_problem:
            st.caption(llm_problem)
        st.markdown(("✅ " if not vision_problem else "⚠️ ") + f"Vision model `{settings.active_vision_model}`")
        if vision_problem:
            st.caption(vision_problem)
        version = tesseract_version(settings.tesseract_cmd)
        st.markdown(f"✅ Tesseract {version}" if version else "⚠️ Tesseract not found (scanned pages skipped)")
        st.markdown(f"✅ Embeddings `{indexer.vector_store.model_name}`")
        loaded = reranker.model_name if reranker.is_loaded else f"{settings.reranker_model} (loads on first question)"
        st.markdown(f"✅ Re-ranker `{loaded}`")
        st.caption(f"Chroma chunks: {indexer.vector_store.count()} · BM25 chunks: {len(indexer.bm25)}")


# ---------------------------------------------------------------------------
# Upload and indexing
# ---------------------------------------------------------------------------


def describe_result(r: IndexResult) -> str:
    c = r.counts
    parts = [f"{r.pages} pages", f"{c.get('text', 0)} text chunks", f"{c.get('table', 0)} tables",
             f"{c.get('figure', 0)} figures"]
    if c.get("ocr"):
        parts.append(f"{c['ocr']} scanned-page chunks")
    return ", ".join(parts)


def upload_section(indexer: Indexer, settings: Settings, has_docs: bool,
                   llm_problem: Optional[str], vision_problem: Optional[str]) -> None:
    with st.expander("Add PDF documents", expanded=not has_docs, icon=":material/upload_file:"):
        files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True,
                                 key=f"uploader_{st.session_state.uploader_key}", label_visibility="collapsed")
        problem = settings.provider_problem()
        if problem and problem != llm_problem:  # the same message is already shown below
            st.error(problem)
        if files and st.button(f"Index {len(files)} file{'s' if len(files) > 1 else ''}", type="primary",
                               icon=":material/auto_awesome:", disabled=bool(problem)):
            index_uploads(indexer, settings, files, llm_problem, vision_problem)

    for r in st.session_state.index_results:  # results of the last upload stay visible
        if r.status == "indexed":
            st.success(f"**{r.file_name}** indexed: {describe_result(r)}.", icon=":material/check_circle:")
        elif r.status == "duplicate":
            st.info(f"**{r.file_name}** is already in the library, skipped.", icon=":material/content_copy:")
        else:
            st.error(f"**{r.file_name}**: {r.message}", icon=":material/error:")


def index_uploads(indexer: Indexer, settings: Settings, files,
                  llm_problem: Optional[str], vision_problem: Optional[str]) -> None:
    """Save each upload to a temp folder, index it with a progress bar, then refresh."""
    if llm_problem and settings.llm_table_summaries:
        st.warning("The chat model is not available, so tables get a simple header-based summary.")
    if settings.caption_figures and vision_problem:
        st.warning("The vision model is not available, so figures are indexed by their printed caption only.")

    indexer.settings = settings  # use the sidebar's chunk size and provider for this upload
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for upload in files:
            path = Path(tmp) / Path(upload.name).name  # keep the original name: citations show it
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

TYPE_ICONS = {"text": ":material/article:", "table": ":material/table_chart:",
              "figure": ":material/insert_chart:", "ocr": ":material/document_scanner:"}
TYPE_NAMES = {"text": "text", "table": "table", "figure": "figure", "ocr": "scanned page"}


def fmt(value: Optional[float], digits: int = 3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def show_source(label: str, chunk: RetrievedChunk, settings: Settings) -> None:
    """One card per cited source: file, page, type, scores, text and figure image."""
    m = chunk.metadata
    relevance = f" · relevance {chunk.rerank_score:.0%}" if chunk.rerank_score is not None else ""
    title = f"**{label}** · {_short(m['file_name'], 34)} · page {m['page']} · {TYPE_NAMES[m['source_type']]}{relevance}"
    with st.expander(title, icon=TYPE_ICONS.get(m["source_type"], ":material/article:")):
        st.markdown(
            '<div class="ink-scores">'
            f"<span>re-rank <b>{fmt(chunk.rerank_score)}</b></span>"
            f"<span>dense <b>{fmt(chunk.dense_score)}</b> (#{chunk.dense_rank or '–'})</span>"
            f"<span>BM25 <b>{fmt(chunk.bm25_score, 2)}</b> (#{chunk.bm25_rank or '–'})</span>"
            f"<span>fused rank <b>#{chunk.fused_rank or '–'}</b></span>"
            f"<span>{html.escape(m['file_name'])}</span></div>",
            unsafe_allow_html=True,
        )
        image_path = m.get("image_path")
        if image_path and settings.data_file(image_path).is_file():
            st.image(str(settings.data_file(image_path)))
        if m["source_type"] in ("table", "figure"):
            st.markdown(md_safe(chunk.document.page_content))  # tables render as real tables
        else:
            st.text(chunk.document.page_content)


def show_answer_details(answer: Answer, settings: Settings) -> None:
    if answer.citations:
        st.markdown('<div class="ink-label">Sources</div>', unsafe_allow_html=True)
        for citation in answer.citations:
            show_source(citation.label, citation.chunk, settings)
    t = answer.timings_ms
    chips = [f"⚡ Retrieval {t.get('retrieval_ms', 0) / 1000:.1f} s",
             f"✍️ Writing {t.get('generation_ms', 0) / 1000:.1f} s"]
    if answer.standalone_question != answer.question:
        chips.append(f"🔁 Searched for: {html.escape(answer.standalone_question)}")
    st.markdown('<div class="ink-meta">' + "".join(f"<span>{c}</span>" for c in chips) + "</div>",
                unsafe_allow_html=True)


def show_history(settings: Settings) -> None:
    for msg in st.session_state.messages:
        avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else "user"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg.get("error"):
                st.error(msg["content"], icon=":material/error:")
                continue
            if msg["role"] == "assistant":
                st.markdown(answer_html(msg["content"]), unsafe_allow_html=True)
            else:
                st.markdown(md_safe(html.escape(msg["content"], quote=False)))
            if msg.get("answer"):
                show_answer_details(msg["answer"], settings)


def answer_question(question: str, retriever: HybridRetriever, settings: Settings, doc_ids: list[str]) -> None:
    """Show the question, stream the answer with citation chips, then sources and timings."""
    history = chat_history(settings.memory_turns)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="user"):
        st.markdown(md_safe(html.escape(question, quote=False)))

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        placeholder = st.empty()
        placeholder.markdown("*Reading your documents…*")
        streamed: list[str] = []

        def on_token(text: str) -> None:
            streamed.append(text)
            placeholder.markdown(answer_html("".join(streamed)) + " ▌", unsafe_allow_html=True)

        def fail(message: str) -> None:
            placeholder.empty()
            st.error(message, icon=":material/error:")
            st.session_state.messages.append({"role": "assistant", "content": message, "error": True})

        try:
            chain = RAGChain(retriever, settings)
            answer = chain.ask(question, history, on_token=on_token, doc_ids=doc_ids or None)
        except (RetrievalError, GenerationError, ValueError) as exc:
            return fail(str(exc))
        except Exception as exc:
            logger.exception("Unexpected error answering %r", question)
            return fail(f"Something went wrong: {exc}")

        placeholder.markdown(answer_html(answer.text), unsafe_allow_html=True)
        show_answer_details(answer, settings)
        st.session_state.messages.append({"role": "assistant", "content": answer.text, "answer": answer})


def _pick_suggestion() -> None:
    """Clicking an example question asks it (the click is turned into a pending question)."""
    st.session_state.pending_question = st.session_state.suggestion
    st.session_state.suggestion = None


def empty_state(docs: list[dict]) -> None:
    if not docs:
        st.markdown(
            """<div class="ink-card"><h3>Start your library</h3><div class="ink-steps">
            <div class="ink-step"><b>1</b><br>Open <i>Add PDF documents</i> and drop in reports, manuals or papers.</div>
            <div class="ink-step"><b>2</b><br>Inkwell reads text, tables, charts and scanned pages, and remembers each page.</div>
            <div class="ink-step"><b>3</b><br>Ask questions. Every answer cites the document and page it came from.</div>
            </div></div>""",
            unsafe_allow_html=True,
        )
        return
    questions = [q for d in docs for q in SAMPLE_QUESTIONS.get(d["file_name"], [])][:6] or GENERIC_QUESTIONS
    st.markdown('<div class="ink-label">Try asking</div>', unsafe_allow_html=True)
    st.pills("Try asking", questions, key="suggestion", on_change=_pick_suggestion, label_visibility="collapsed")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


@st.fragment(run_every=1.5)
def watch_warmup(warmup: Warmup) -> None:
    """While models load in the background, check every 1.5 s and refresh the page when done."""
    if warmup.ready:
        st.rerun(scope="app")


def main() -> None:
    init_state()
    load_css()
    base = get_settings()

    try:
        indexer = load_indexer()
        warmup = start_warmup()
    except Exception as exc:
        logger.exception("Startup failed")
        st.error(f"Could not open the library: {exc}")
        st.stop()

    # Documents added from the command line while the app runs: refresh the BM25 index.
    if len(indexer.bm25) != indexer.vector_store.count():
        indexer.rebuild_bm25()

    settings = sidebar_settings(base)
    docs = indexer.list_documents()
    doc_ids = sidebar_documents(docs, indexer)
    st.sidebar.divider()
    if st.sidebar.button("Clear chat", icon=":material/mop:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    health_key = (settings.provider, settings.active_llm_model, settings.active_vision_model,
                  settings.openai_api_key is not None, settings.caption_figures)
    llm_problem, vision_problem = model_health(settings, health_key)
    reranker = get_reranker(settings.reranker_model)
    sidebar_status(settings, indexer, reranker, llm_problem, vision_problem)

    render_hero(settings, llm_problem, warmup, len(docs))
    if not warmup.ready:
        watch_warmup(warmup)
    if warmup.error:
        st.error(f"Could not load the models: {warmup.error}. The first run needs internet to download them.")

    store = indexer.vector_store
    if store.model_mismatch:  # index built with another embedding model: searching it would fail
        st.warning(store.mismatch_message())
        if st.button("Rebuild the index now", type="primary"):
            bar = st.progress(0.0, text="Rebuilding...")
            st.session_state.index_results = indexer.rebuild_all(
                progress=lambda msg, frac: bar.progress(min(frac, 1.0), text=msg))
            st.rerun()
        st.stop()

    if docs:
        render_stats(docs)
    upload_section(indexer, settings, bool(docs), llm_problem, vision_problem)
    if llm_problem:
        st.error(llm_problem, icon=":material/power_off:")

    if not st.session_state.messages:
        empty_state(docs)
    show_history(settings)

    if not docs:
        hint = "Add a PDF to start asking questions"
    elif llm_problem:
        hint = "Fix the problem above to start asking"
    else:
        hint = "Ask about your documents…"
    question = st.chat_input(hint, disabled=not docs or bool(llm_problem))
    if question is None and st.session_state.pending_question and docs and not llm_problem:
        question, st.session_state.pending_question = st.session_state.pending_question, None

    if question is not None:
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            retriever = HybridRetriever(store, indexer.bm25, reranker=reranker, settings=settings)
            answer_question(question.strip(), retriever, settings, doc_ids)


main()
