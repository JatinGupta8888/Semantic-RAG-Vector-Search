"""
retriever.py  —  Strategy A: Raw Vector Search
-----------------------------------------------
Embeds the user query as-is and performs a direct FAISS nearest-neighbour
search.  No query rewriting or augmentation is applied.

This is the baseline strategy against which Strategy B is benchmarked.

Pipeline
~~~~~~~~
  User Query  ──embed──►  Query Vector  ──FAISS search──►  Top-K SearchResults
"""

from __future__ import annotations

import logging
from typing import List

from src.embedding.embedder import Embedder
from src.storage.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Strategy A — direct embedding retrieval.

    Parameters
    ----------
    embedder : Embedder
        Embedding model used to encode the query.
    vector_store : VectorStore
        Populated vector store to search against.

    Example
    -------
    >>> retriever = Retriever(embedder, store)
    >>> results = retriever.retrieve("How does the system handle peak load?", top_k=3)
    """

    STRATEGY_NAME = "Strategy A — Raw Vector Search"

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Retrieve the top-k most relevant documents for ``query``.

        Parameters
        ----------
        query : str
            Raw user query.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[SearchResult]
            Ranked by descending cosine similarity.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")

        logger.info("[Strategy A] Query: %r", query)
        query_vector = self.embedder.embed_single(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        logger.info("[Strategy A] Retrieved %d results.", len(results))
        return results

    def __repr__(self) -> str:
        return (
            f"Retriever(embedder={self.embedder!r}, "
            f"store_size={len(self.vector_store)})"
        )
