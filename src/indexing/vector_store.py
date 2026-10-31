"""ChromaDB wrapper: one persistent collection holding every chunk and its vector.

The name of the embedding model is saved in the collection's metadata. Vectors from
different models can't be compared (they even have different lengths), so if the
model in .env changes we refuse to add to the old collection and ask for a rebuild.
"""

from typing import Callable, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import Settings, get_settings
from src.indexing.embeddings import get_embeddings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Add chunks to Chroma this many at a time, so progress can be shown and RAM stays low.
ADD_BATCH_SIZE = 64


class EmbeddingModelMismatch(Exception):
    """The collection was built with a different embedding model than the one in .env."""


class VectorStore:
    """Persistent ChromaDB collection + the embedding model used to fill it."""

    def __init__(self, settings: Optional[Settings] = None, embeddings: Optional[Embeddings] = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = self.settings.active_embedding_model
        self._embeddings = embeddings  # loaded on first use, see the property below

        # anonymized_telemetry=False keeps Chroma fully offline.
        self.client = chromadb.PersistentClient(
            path=str(self.settings.chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self._open_collection()

    # ---------- collection and model check ----------

    def _open_collection(self):
        """Open the collection, creating it (with our model name stored) if it is new."""
        return self.client.get_or_create_collection(
            name=self.settings.chroma_collection,
            # "hnsw:space": "cosine" -> distance = 1 - cosine similarity (0 = identical).
            metadata={"hnsw:space": "cosine", "embedding_model": self.model_name},
        )

    @property
    def stored_model(self) -> Optional[str]:
        """The embedding model the collection was built with."""
        return (self.collection.metadata or {}).get("embedding_model")

    @property
    def model_mismatch(self) -> bool:
        """True if .env now names a different embedding model than the stored one."""
        return self.stored_model is not None and self.stored_model != self.model_name

    def mismatch_message(self) -> str:
        return (
            f"The index was built with '{self.stored_model}' but .env selects '{self.model_name}'. "
            "Rebuild the index (re-embeds every document), or switch the model back."
        )

    def reset(self) -> None:
        """Delete the whole collection and create an empty one for the current model."""
        logger.warning("Deleting collection '%s' (was built with %s)", self.settings.chroma_collection, self.stored_model)
        self.client.delete_collection(self.settings.chroma_collection)
        self.collection = self._open_collection()

    # ---------- LangChain access ----------

    @property
    def embeddings(self) -> Embeddings:
        """Load the embedding model only when it is first needed (it takes a few seconds)."""
        if self._embeddings is None:
            self._embeddings = get_embeddings(self.settings)
        return self._embeddings

    @property
    def langchain_store(self) -> Chroma:
        """The same collection as a LangChain vector store (used for searching in Phase 4)."""
        return Chroma(
            client=self.client,
            collection_name=self.settings.chroma_collection,
            embedding_function=self.embeddings,
        )

    # ---------- add / delete / read ----------

    def add_chunks(self, chunks: list[Document], progress: Optional[Callable[[int, int], None]] = None) -> None:
        """Embed and store chunks in batches. progress(done, total) is called after each batch."""
        if self.model_mismatch:
            raise EmbeddingModelMismatch(self.mismatch_message())

        total = len(chunks)
        for start in range(0, total, ADD_BATCH_SIZE):
            batch = chunks[start : start + ADD_BATCH_SIZE]
            texts = [c.page_content for c in batch]
            self.collection.upsert(  # upsert = insert, or overwrite if the ID already exists
                ids=[c.metadata["chunk_id"] for c in batch],
                documents=texts,
                metadatas=[c.metadata for c in batch],
                embeddings=self.embeddings.embed_documents(texts),
            )
            if progress:
                progress(min(start + ADD_BATCH_SIZE, total), total)
        logger.info("Added %d chunks to Chroma", total)

    def has_document(self, doc_id: str) -> bool:
        """True if any chunk of this document is already stored."""
        return len(self.collection.get(where={"doc_id": doc_id}, limit=1)["ids"]) > 0

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks of a document. Returns how many were removed."""
        ids = self.collection.get(where={"doc_id": doc_id})["ids"]
        if ids:
            self.collection.delete(ids=ids)
        logger.info("Deleted %d chunks of document %s from Chroma", len(ids), doc_id)
        return len(ids)

    def count(self) -> int:
        return self.collection.count()

    def all_chunks(self) -> list[Document]:
        """Every stored chunk as a Document (used to build the BM25 index)."""
        data = self.collection.get(include=["documents", "metadatas"])
        return [Document(page_content=text, metadata=meta) for text, meta in zip(data["documents"], data["metadatas"])]

    def list_documents(self) -> list[dict]:
        """One summary row per document: doc_id, file_name, pages, chunk counts per type."""
        docs: dict[str, dict] = {}
        for meta in self.collection.get(include=["metadatas"])["metadatas"]:
            row = docs.setdefault(
                meta["doc_id"],
                {"doc_id": meta["doc_id"], "file_name": meta["file_name"], "pages": set(), "chunks": 0, "types": {}},
            )
            row["pages"].add(meta["page"])
            row["chunks"] += 1
            row["types"][meta["source_type"]] = row["types"].get(meta["source_type"], 0) + 1
        for row in docs.values():
            row["pages"] = len(row["pages"])  # pages that produced at least one chunk
        return sorted(docs.values(), key=lambda r: r["file_name"].lower())
