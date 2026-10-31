"""Streamlit entry point.

Phase 1: a hello-world page that also checks the local setup
(Ollama running, models pulled, Tesseract installed). The chat UI comes in Phase 7.

Run with:  streamlit run app.py
"""

import json
import sys
import urllib.request

import streamlit as st

from src.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def check_ollama(base_url: str) -> tuple[bool, list[str]]:
    """Return (is_running, list_of_pulled_model_names) using Ollama's /api/tags endpoint."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
            data = json.load(response)
        return True, [m["name"] for m in data.get("models", [])]
    except Exception as exc:  # connection refused, timeout, bad JSON...
        logger.warning("Ollama not reachable at %s: %s", base_url, exc)
        return False, []


def check_tesseract() -> tuple[bool, str]:
    """Return (is_installed, version_or_error_message)."""
    import pytesseract

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    try:
        return True, str(pytesseract.get_tesseract_version())
    except Exception as exc:
        return False, str(exc)


def model_pulled(wanted: str, pulled: list[str]) -> bool:
    """Ollama names models 'llama3.2:latest'; treat 'llama3.2' as ':latest'."""
    full_name = wanted if ":" in wanted else f"{wanted}:latest"
    return full_name in pulled


st.set_page_config(page_title="Multimodal RAG", page_icon="📄", layout="wide")
st.title("📄 Autonomous Multimodal RAG Platform")
st.caption("Phase 1: setup check. Upload and chat arrive in later phases.")

problem = settings.provider_problem()
if problem:
    st.error(problem)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Configuration")
    st.markdown(
        f"""
- **Python:** {sys.version.split()[0]}
- **Provider:** `{settings.provider}`
- **LLM:** `{settings.active_llm_model}`
- **Vision model:** `{settings.active_vision_model}`
- **Embeddings:** `{settings.active_embedding_model}`
- **Re-ranker:** `{settings.reranker_model}`
- **Chunk size / overlap:** {settings.chunk_size} / {settings.chunk_overlap}
- **Data folder:** `{settings.data_path}`
"""
    )

with col2:
    st.subheader("Local services")

    if settings.provider == "ollama":
        running, pulled = check_ollama(settings.ollama_base_url)
        if running:
            st.success(f"Ollama is running at {settings.ollama_base_url}")
            for name in (settings.ollama_llm_model, settings.ollama_vision_model):
                if model_pulled(name, pulled):
                    st.success(f"Model `{name}` is pulled")
                else:
                    st.warning(f"Model `{name}` not found. Run: `ollama pull {name}`")
        else:
            st.error("Ollama is not running. Start the Ollama app, then refresh this page.")
    else:
        st.info("Provider is OpenAI, so Ollama isn't needed.")

    ok, info = check_tesseract()
    if ok:
        st.success(f"Tesseract OCR found (version {info})")
    else:
        st.warning("Tesseract not found. Install it, or set TESSERACT_CMD in .env.")

logger.info("Setup page rendered (provider=%s)", settings.provider)
