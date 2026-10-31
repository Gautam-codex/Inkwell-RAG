"""LLM factory: Ollama (local, default) or OpenAI, chosen by PROVIDER in .env.

Both are returned as LangChain chat models, so the rest of the code works the same
whichever provider is selected.
"""

import json
import urllib.request
from typing import Callable, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Context window for Ollama, in tokens. Ollama's default (2048-4096) can be too small
# for 5 sources + a table + the instructions. 8192 uses roughly 1 GB extra RAM.
OLLAMA_NUM_CTX = 8192


def get_llm(settings: Optional[Settings] = None, temperature: float = 0.1) -> BaseChatModel:
    """Return the chat model for the selected provider.

    A low temperature keeps answers factual and close to the sources.
    """
    settings = settings or get_settings()
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=OLLAMA_NUM_CTX,
        )

    from langchain_openai import ChatOpenAI

    if settings.openai_api_key is None:
        raise ValueError("PROVIDER=openai needs OPENAI_API_KEY in .env or the sidebar.")
    return ChatOpenAI(model=settings.openai_llm_model, api_key=settings.openai_api_key, temperature=temperature)


def get_vision_llm(settings: Optional[Settings] = None, max_tokens: int = 350) -> BaseChatModel:
    """Return the vision model (llama3.2-vision / llava on Ollama, gpt-4o-mini on OpenAI).

    max_tokens caps the length of each description, which also caps the time per figure.
    """
    settings = settings or get_settings()
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_vision_model,
            base_url=settings.ollama_base_url,
            temperature=0.0,
            num_predict=max_tokens,
        )

    from langchain_openai import ChatOpenAI

    if settings.openai_api_key is None:
        raise ValueError("PROVIDER=openai needs OPENAI_API_KEY in .env or the sidebar.")
    return ChatOpenAI(
        model=settings.openai_vision_model, api_key=settings.openai_api_key, temperature=0.0, max_tokens=max_tokens
    )


def check_vision(settings: Optional[Settings] = None) -> Optional[str]:
    """Return a readable problem with the vision model, or None if it looks ready."""
    settings = settings or get_settings()
    if settings.provider == "openai":
        return settings.provider_problem()
    running, pulled = list_ollama_models(settings.ollama_base_url)
    if not running:
        return f"Ollama is not running at {settings.ollama_base_url}."
    if not model_pulled(settings.ollama_vision_model, pulled):
        return f"Vision model '{settings.ollama_vision_model}' is not pulled. Run: ollama pull {settings.ollama_vision_model}"
    return None


def make_summarizer(settings: Optional[Settings] = None) -> Callable[[str], str]:
    """Return a simple prompt -> text function (used for table summaries at ingestion)."""
    llm = get_llm(settings, temperature=0.0)
    return lambda prompt: str(llm.invoke(prompt).content)


# ---------------------------------------------------------------------------
# Health checks and readable errors
# ---------------------------------------------------------------------------


def list_ollama_models(base_url: str) -> tuple[bool, list[str]]:
    """Return (is_running, names_of_pulled_models) using Ollama's /api/tags endpoint."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
            data = json.load(response)
        return True, [m["name"] for m in data.get("models", [])]
    except Exception as exc:  # connection refused, timeout, bad JSON...
        logger.warning("Ollama not reachable at %s: %s", base_url, exc)
        return False, []


def model_pulled(wanted: str, pulled: list[str]) -> bool:
    """Ollama names models 'llama3.2:latest'; treat 'llama3.2' as ':latest'."""
    full_name = wanted if ":" in wanted else f"{wanted}:latest"
    return full_name in pulled


def check_llm(settings: Optional[Settings] = None) -> Optional[str]:
    """Return a readable problem with the selected LLM, or None if it looks ready."""
    settings = settings or get_settings()
    if settings.provider == "openai":
        return settings.provider_problem()
    running, pulled = list_ollama_models(settings.ollama_base_url)
    if not running:
        return f"Ollama is not running at {settings.ollama_base_url}. Start the Ollama app (or run `ollama serve`)."
    if not model_pulled(settings.ollama_llm_model, pulled):
        return f"Model '{settings.ollama_llm_model}' is not pulled. Run: ollama pull {settings.ollama_llm_model}"
    return None


def friendly_llm_error(exc: Exception) -> str:
    """Turn a provider exception into a message a user can act on."""
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "connect" in lowered or "refused" in lowered:
        return "Cannot reach the language model. Is Ollama running? Start the Ollama app and try again."
    if "not found" in lowered and "model" in lowered:
        return "The selected model is not available. For Ollama run: ollama pull <model name>"
    if "401" in lowered or "api key" in lowered or "authentication" in lowered:
        return "The OpenAI API key was rejected. Check OPENAI_API_KEY."
    if "429" in lowered or "rate limit" in lowered or "quota" in lowered:
        return "The OpenAI rate limit or quota was reached. Wait a moment or check your billing."
    return f"The language model failed: {exc}"
