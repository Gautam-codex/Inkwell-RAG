"""File hashing, used to give each PDF a stable ID and to skip re-indexing duplicates.

The same bytes always give the same hash, so if a user uploads report.pdf twice
(even under a different name) we can tell it is already indexed.
"""

import hashlib
from pathlib import Path

# Short IDs are easier to read in logs and chunk IDs. 16 hex characters = 64 bits,
# so two different PDFs getting the same ID is practically impossible.
DOC_ID_LENGTH = 16


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of some bytes (e.g. a Streamlit upload)."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MB blocks to keep memory low."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def doc_id_for_file(path: str | Path) -> str:
    """Return the document ID for a file: the first 16 characters of its SHA-256."""
    return sha256_file(path)[:DOC_ID_LENGTH]
