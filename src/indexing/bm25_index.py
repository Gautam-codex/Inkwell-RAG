"""BM25 keyword index over the same chunks that are stored in ChromaDB.

Dense (embedding) search is good at meaning; BM25 is good at exact words: part
numbers, names, acronyms, figures like "Q3 2023". Phase 4 combines both.

BM25 in one line: a chunk scores higher when it contains the query's words often,
when those words are rare across all chunks, and when the chunk is not very long.

The index is rebuilt from Chroma after every add/delete, so the two never disagree,
and saved with pickle so it loads instantly next time.
"""

import pickle
import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Bump this if the saved format changes, so an old file is rebuilt instead of misread.
FORMAT_VERSION = 1

# Very common words that carry no meaning for search.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "what", "which", "who", "with", "how", "does", "do", "did",
}

# Words, plus numbers and codes that contain dots or dashes: "3.5", "q1-2023", "llama3.2".
TOKEN_PATTERN = re.compile(r"\w+(?:[.\-]\w+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case the text, split it into words and drop stopwords."""
    return [t for t in TOKEN_PATTERN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Build, save, load and search a BM25 index."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.chunks: list[Document] = []
        self.bm25: Optional[BM25Okapi] = None

    def __len__(self) -> int:
        return len(self.chunks)

    def build(self, chunks: list[Document]) -> None:
        """Build the index from chunks and save it to disk."""
        self.chunks = list(chunks)
        corpus = [tokenize(c.page_content) for c in self.chunks]
        # rank_bm25 crashes on an empty corpus, so an empty index just has no model.
        self.bm25 = BM25Okapi(corpus) if corpus else None
        self.save()
        logger.info("BM25 index built over %d chunks", len(self.chunks))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({"version": FORMAT_VERSION, "chunks": self.chunks, "bm25": self.bm25}, f)

    def load(self) -> bool:
        """Load the saved index. Returns False if there is none (or it is outdated)."""
        if not self.path.is_file():
            return False
        try:
            # pickle is safe here because we only ever load a file this app wrote itself.
            with open(self.path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            logger.warning("Could not read BM25 index %s, it will be rebuilt: %s", self.path, exc)
            return False
        if data.get("version") != FORMAT_VERSION:
            return False
        self.chunks, self.bm25 = data["chunks"], data["bm25"]
        return True

    def search(self, query: str, top_k: int, doc_ids: Optional[list[str]] = None) -> list[tuple[Document, float]]:
        """Return up to top_k (chunk, bm25_score) pairs, best first.

        doc_ids restricts the search to chosen documents. Chunks with score 0
        (no query word in common) are left out.
        """
        tokens = tokenize(query)
        if self.bm25 is None or not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked:
            if scores[i] <= 0:
                break  # the rest score 0 too
            if doc_ids and self.chunks[i].metadata["doc_id"] not in doc_ids:
                continue
            results.append((self.chunks[i], float(scores[i])))
            if len(results) == top_k:
                break
        return results
