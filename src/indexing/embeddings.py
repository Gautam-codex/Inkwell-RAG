"""Embedding factory: BGE (local, offline) or OpenAI.

An embedding model turns text into a list of numbers (a vector) so that texts with
similar meaning end up close together. ChromaDB stores these vectors and finds the
chunks closest to the question.

We wrap sentence-transformers ourselves (about 30 lines) instead of adding another
package, and this also lets us add BGE's recommended query instruction.
"""

import os
from functools import lru_cache
from typing import Optional

from langchain_core.embeddings import Embeddings

from src.config import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Windows without Developer Mode can't make symlinks; Hugging Face then warns on every
# load even though caching still works. The warning is harmless, so silence it.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# BGE models retrieve better when the QUESTION (not the documents) starts with this
# instruction. It comes from the BAAI model card for the bge-*-en-v1.5 models.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Embed this many chunks at a time. Larger batches are faster but use more RAM.
EMBED_BATCH_SIZE = 32


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible wrapper around a local sentence-transformers model."""

    def __init__(self, model_name: str, batch_size: int = EMBED_BATCH_SIZE) -> None:
        from sentence_transformers import SentenceTransformer  # slow import, so done lazily

        self.model_name = model_name
        self.batch_size = batch_size
        try:
            # Use the copy already downloaded to ~/.cache/huggingface, with no internet.
            self.model = SentenceTransformer(model_name, local_files_only=True)
            logger.info("Loaded embedding model %s from the local cache", model_name)
        except Exception:
            logger.info("Downloading embedding model %s (first run only)...", model_name)
            self.model = SentenceTransformer(model_name)
        self.query_instruction = BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks. normalize_embeddings=True makes cosine similarity a simple dot product."""
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a question, with the BGE instruction in front."""
        return self.embed_documents([self.query_instruction + text])[0]


@lru_cache(maxsize=2)
def _load_local(model_name: str) -> SentenceTransformerEmbeddings:
    """Load each local model only once per process (loading takes several seconds)."""
    return SentenceTransformerEmbeddings(model_name)


def get_embeddings(settings: Optional[Settings] = None) -> Embeddings:
    """Return the embedding model selected by EMBEDDING_PROVIDER in .env."""
    settings = settings or get_settings()
    if settings.embedding_provider == "local":
        return _load_local(settings.local_embedding_model)

    from langchain_openai import OpenAIEmbeddings

    if settings.openai_api_key is None:
        raise ValueError("EMBEDDING_PROVIDER=openai needs OPENAI_API_KEY in .env")
    return OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
