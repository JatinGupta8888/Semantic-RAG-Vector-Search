"""
test_retriever.py
-----------------
Unit tests for Strategy A: src/retrieval/retriever.py

Tests cover:
  - Basic retrieval returns expected types
  - Result ranking is correct
  - Empty query raises ValueError
  - Relevance: on-topic query retrieves the most relevant document
"""

from __future__ import annotations

import pytest

from src.embedding.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.storage.vector_store import SearchResult, VectorStore


class TestRetrieverInit:
    def test_strategy_name(self) -> None:
        assert "Strategy A" in Retriever.STRATEGY_NAME

    def test_repr(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        r = Retriever(embedder, populated_store)
        assert "Retriever" in repr(r)


class TestRetrieve:
    def test_returns_list_of_search_results(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        retriever = Retriever(embedder, populated_store)
        results = retriever.retrieve("how does traffic distribution work?", top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_default_top_k_is_three(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        retriever = Retriever(embedder, populated_store)
        results = retriever.retrieve("semantic search embeddings")
        assert len(results) == 3

    def test_top_k_one(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        retriever = Retriever(embedder, populated_store)
        results = retriever.retrieve("circuit breaker pattern", top_k=1)
        assert len(results) == 1

    def test_empty_query_raises(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        retriever = Retriever(embedder, populated_store)
        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("   ", top_k=1)

    def test_scores_descending(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        retriever = Retriever(embedder, populated_store)
        results = retriever.retrieve("vector similarity search", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_doc_ranked_first(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        """
        A query about load balancing should surface t001 at rank 1.
        """
        retriever = Retriever(embedder, populated_store)
        results = retriever.retrieve(
            "load balancer distributes requests to prevent bottlenecks", top_k=3
        )
        assert results[0].document.id == "t001"

    def test_result_documents_have_text(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        retriever = Retriever(embedder, populated_store)
        results = retriever.retrieve("peak load handling", top_k=2)
        for r in results:
            assert r.document.text
            assert r.document.id
            assert r.document.title
