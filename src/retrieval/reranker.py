"""Cross-encoder re-ranking.

Why a second step? The embedding model (a "bi-encoder") turns the question and each
chunk into vectors SEPARATELY, then compares the vectors. That is fast enough to
search thousands of chunks, but it never looks at question and chunk together.

A cross-encoder reads the question and one chunk TOGETHER as a single input
("[question] [SEP] [chunk]") and outputs one relevance score. Because every word of
the question can attend to every word of the chunk, it judges relevance much
better, but it is too slow to run on every chunk. So we use it only on the ~40
candidates that hybrid search found, and keep the best few.

bge-reranker-base outputs a raw number (a logit); sentence-transformers passes it
through a sigmoid, so the score we get is between 0 (irrelevant) and 1 (relevant).
"""

import os
from typing import Optional

from src.config import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Number of (question, chunk) pairs scored at once. Lower it if RAM is tight.
RERANK_BATCH_SIZE = 16
# Chunks longer than this many tokens are cut off; 512 is the model's limit.
MAX_LENGTH = 512


def _load_cross_encoder(model_name: str):
    """Load from the local cache (offline); download only if it is not there yet."""
    from sentence_transformers import CrossEncoder  # slow import, so done lazily

    try:
        return CrossEncoder(model_name, max_length=MAX_LENGTH, local_files_only=True)
    except Exception:
        logger.info("Downloading re-ranker %s (first run only)...", model_name)
        return CrossEncoder(model_name, max_length=MAX_LENGTH)


class Reranker:
    """Scores (question, chunk) pairs with a cross-encoder. The model loads on first use."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.model_name: Optional[str] = None
        self._model = None

    def load(self) -> None:
        """Load the model now (otherwise it loads on the first query)."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        primary, fallback = self.settings.reranker_model, self.settings.reranker_fallback_model
        try:
            self._model = _load_cross_encoder(primary)
            self.model_name = primary
        except Exception as exc:  # e.g. MemoryError on a small laptop, or no internet on first run
            logger.warning("Could not load re-ranker %s (%s); using smaller %s", primary, exc, fallback)
            self._model = _load_cross_encoder(fallback)
            self.model_name = fallback
        logger.info("Re-ranker ready: %s", self.model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score (0 to 1) per text, in the same order."""
        if not texts:
            return []
        self._ensure_loaded()
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False)
        return [float(s) for s in scores]
