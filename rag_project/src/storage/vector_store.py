"""
vector_store.py
---------------
Vector store that ingests text documents, stores their embeddings, and
exposes a ``search`` method for similarity retrieval.

Backend priority:
  1. FAISS ``IndexFlatIP`` — production-grade, highly optimised
  2. Pure NumPy fallback — exact cosine search via matrix multiply

Similarity metric
~~~~~~~~~~~~~~~~~
We use cosine similarity implemented via inner-product search on
L2-normalised vectors.

    cosine_sim(A, B) = A · B  when ||A|| = ||B|| = 1

Production migration note
~~~~~~~~~~~~~~~~~~~~~~~~~
For production at scale, swap to Vertex AI Matching Engine.
The VectorStore interface is storage-agnostic to make this easy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _try_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


@dataclass
class Document:
    """
    A chunk of text with its metadata and (optionally) its embedding.

    Attributes
    ----------
    id : str     Stable identifier (e.g. "doc_001").
    title : str  Human-readable title.
    text : str   The full chunk text.
    embedding : np.ndarray | None  Dense vector set during ingestion.
    """
    id: str
    title: str
    text: str
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class SearchResult:
    """A single retrieved document with its similarity score."""
    rank: int
    score: float  # cosine similarity ∈ [-1, 1]; higher = more similar
    document: Document

    def __str__(self) -> str:
        return (
            f"[Rank {self.rank}]  score={self.score:.4f}  id={self.document.id!r}\n"
            f"  title : {self.document.title}\n"
            f"  text  : {self.document.text[:120]}…"
        )


class _FaissIndex:
    """FAISS-backed exact inner-product index."""

    def __init__(self, dim: int) -> None:
        faiss = _try_faiss()
        self._index = faiss.IndexFlatIP(dim)
        logger.info("Index backend: FAISS IndexFlatIP")

    def add(self, vectors: np.ndarray) -> None:
        self._index.add(vectors.astype(np.float32))

    def search(self, query: np.ndarray, k: int):
        scores, indices = self._index.search(query.reshape(1, -1).astype(np.float32), k)
        return scores[0].tolist(), indices[0].tolist()

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def reset(self) -> None:
        self._index.reset()


class _NumpyIndex:
    """Pure NumPy exact cosine (inner-product) index."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._matrix: Optional[np.ndarray] = None
        logger.info("Index backend: NumPy (FAISS not available)")

    def add(self, vectors: np.ndarray) -> None:
        v = vectors.astype(np.float32)
        if self._matrix is None:
            self._matrix = v
        else:
            self._matrix = np.vstack([self._matrix, v])

    def search(self, query: np.ndarray, k: int):
        if self._matrix is None:
            return [], []
        scores = (self._matrix @ query.astype(np.float32)).tolist()
        # Get top-k indices by descending score
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [scores[i] for i in indices], indices

    @property
    def ntotal(self) -> int:
        return 0 if self._matrix is None else self._matrix.shape[0]

    def reset(self) -> None:
        self._matrix = None


def _build_index(dim: int):
    if _try_faiss() is not None:
        return _FaissIndex(dim)
    return _NumpyIndex(dim)


class VectorStore:
    """
    Manages a vector index over a collection of Document objects.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the embedding vectors to be stored.

    Example
    -------
    >>> from src.embedding.embedder import Embedder
    >>> embedder = Embedder()
    >>> store = VectorStore(embedding_dim=embedder.embedding_dim)
    >>> store.ingest([{"id": "d1", "title": "T", "text": "hello world"}], embedder)
    >>> results = store.search(embedder.embed_single("hello"), top_k=1)
    """

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        self._documents: List[Document] = []
        self._index = _build_index(embedding_dim)
        logger.info("VectorStore initialised — dim=%d", embedding_dim)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, raw_docs: List[dict], embedder) -> None:
        """
        Embed and index a list of raw document dicts.

        Each dict must have keys: "id", "title", "text".
        """
        required_keys = {"id", "title", "text"}
        for doc in raw_docs:
            missing = required_keys - doc.keys()
            if missing:
                raise ValueError(f"Document missing keys: {missing!r}  ->  {doc}")

        texts = [d["text"] for d in raw_docs]
        logger.info("Ingesting %d documents…", len(texts))
        vectors: np.ndarray = embedder.embed(texts)

        for i, raw in enumerate(raw_docs):
            doc = Document(id=raw["id"], title=raw["title"], text=raw["text"], embedding=vectors[i])
            self._documents.append(doc)

        self._index.add(vectors.astype(np.float32))
        logger.info("Index now contains %d vectors.", self._index.ntotal)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        """
        Return the top-k most similar documents to query_vector.

        Parameters
        ----------
        query_vector : np.ndarray
            1-D float32 array of shape (embedding_dim,). Must be L2-normalised.
        top_k : int
            Number of results to return.
        """
        if self._index.ntotal == 0:
            raise ValueError("VectorStore is empty — call ingest() first.")
        if query_vector.shape != (self.embedding_dim,):
            raise RuntimeError(
                f"Expected query shape ({self.embedding_dim},), got {query_vector.shape}"
            )

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, k)

        results: List[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx == -1:
                continue
            results.append(SearchResult(rank=rank, score=float(score), document=self._documents[idx]))

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._index.ntotal

    def reset(self) -> None:
        self._documents.clear()
        self._index.reset()
        logger.info("VectorStore reset.")

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"VectorStore(dim={self.embedding_dim}, size={self.size})"
