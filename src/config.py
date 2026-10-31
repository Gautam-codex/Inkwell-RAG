"""Central settings for the whole app.

Values are read from (highest priority first):
  1. real environment variables
  2. the .env file in the project folder
  3. the defaults written below

Usage:
    from src.config import get_settings
    settings = get_settings()
    print(settings.active_llm_model)
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder that contains app.py (one level above src/).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All configurable values. Each field maps to an env var of the same name in UPPER CASE."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",  # absolute path, so it works from any working directory
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # "OPENAI_API_KEY=" in .env counts as "not set"
        extra="ignore",
    )

    # ---------- Provider (single local/cloud switch for LLM + vision) ----------
    provider: Literal["ollama", "openai"] = "ollama"

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2"
    ollama_vision_model: str = "llama3.2-vision"

    # OpenAI (cloud, optional)
    openai_api_key: Optional[SecretStr] = None  # SecretStr hides the key in logs/printing
    openai_llm_model: str = "gpt-4o-mini"
    openai_vision_model: str = "gpt-4o-mini"

    # ---------- Embeddings ----------
    # Kept separate from `provider` because switching embeddings forces a full re-index.
    embedding_provider: Literal["local", "openai"] = "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"

    # ---------- Re-ranker ----------
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_fallback_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---------- Chunking ----------
    chunk_size: int = Field(800, gt=0)
    chunk_overlap: int = Field(150, ge=0)

    # ---------- Retrieval ----------
    dense_top_k: int = Field(20, gt=0)
    bm25_top_k: int = Field(20, gt=0)
    rrf_k: int = Field(60, gt=0)
    rerank_top_n: int = Field(5, gt=0)

    # ---------- Ingestion ----------
    min_image_size: int = Field(100, gt=0)
    ocr_min_text_chars: int = Field(50, ge=0)
    ocr_image_coverage: float = Field(0.6, gt=0, le=1)
    tesseract_cmd: Optional[str] = None

    # ---------- Conversation ----------
    memory_turns: int = Field(4, ge=0)

    # ---------- Storage and logging ----------
    data_dir: Path = Path("data")
    chroma_collection: str = "documents"
    log_level: str = "INFO"

    # ---------- Derived values ----------
    @property
    def data_path(self) -> Path:
        """Absolute data folder. A relative DATA_DIR is resolved against the project root."""
        return self.data_dir if self.data_dir.is_absolute() else PROJECT_ROOT / self.data_dir

    @property
    def uploads_dir(self) -> Path:
        return self.data_path / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_path / "chroma"

    @property
    def images_dir(self) -> Path:
        return self.data_path / "extracted_images"

    @property
    def bm25_path(self) -> Path:
        return self.data_path / "bm25_index.pkl"

    @property
    def logs_dir(self) -> Path:
        return self.data_path / "logs"

    @property
    def active_llm_model(self) -> str:
        """Chat model name for the selected provider."""
        return self.ollama_llm_model if self.provider == "ollama" else self.openai_llm_model

    @property
    def active_vision_model(self) -> str:
        """Vision model name for the selected provider."""
        return self.ollama_vision_model if self.provider == "ollama" else self.openai_vision_model

    @property
    def active_embedding_model(self) -> str:
        """Embedding model name. Stored with the Chroma collection to detect model changes."""
        if self.embedding_provider == "local":
            return self.local_embedding_model
        return self.openai_embedding_model

    def provider_problem(self) -> Optional[str]:
        """Return a readable message if the chosen provider can't work, else None.

        Returning a message instead of raising lets the UI show the problem without crashing.
        """
        needs_openai = self.provider == "openai" or self.embedding_provider == "openai"
        key_missing = self.openai_api_key is None or not self.openai_api_key.get_secret_value().strip()
        if needs_openai and key_missing:
            return "OpenAI is selected but OPENAI_API_KEY is empty. Add it to .env or the sidebar."
        if self.chunk_overlap >= self.chunk_size:
            return "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        return None

    def ensure_dirs(self) -> None:
        """Create the data folders if they don't exist yet (data/ is not in git)."""
        for folder in (self.uploads_dir, self.chroma_dir, self.images_dir, self.logs_dir):
            folder.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once and reuse them (call get_settings.cache_clear() to reload)."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
