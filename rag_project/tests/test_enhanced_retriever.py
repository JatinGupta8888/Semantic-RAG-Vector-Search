"""
test_enhanced_retriever.py
--------------------------
Unit tests for Strategy B: src/retrieval/enhanced_retriever.py

Tests cover:
  - Basic retrieval returns expected types
  - expanded_query is populated after retrieve()
  - Expanded query is longer/richer than the original
  - MockGenerativeModel can be patched via unittest.mock
  - Empty query raises ValueError
  - expand_query() returns non-empty string
  - GCP SDK is fully mocked — no real API calls
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.embedding.embedder import Embedder
from src.embedding.mock_vertex import MockGenerativeModel
from src.retrieval.enhanced_retriever import EnhancedRetriever
from src.storage.vector_store import SearchResult, VectorStore


class TestEnhancedRetrieverInit:
    def test_strategy_name(self) -> None:
        assert "Strategy B" in EnhancedRetriever.STRATEGY_NAME

    def test_default_model_created(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        r = EnhancedRetriever(embedder, populated_store)
        assert r.generative_model is not None

    def test_custom_model_accepted(
        self,
        embedder: Embedder,
        populated_store: VectorStore,
        mock_generative_model: MockGenerativeModel,
    ) -> None:
        r = EnhancedRetriever(embedder, populated_store, mock_generative_model)
        assert r.generative_model is mock_generative_model


class TestRetrieve:
    def test_returns_list_of_search_results(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        r = EnhancedRetriever(embedder, populated_store)
        results = r.retrieve("How does the system handle peak load?", top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(res, SearchResult) for res in results)

    def test_expanded_query_populated(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        r = EnhancedRetriever(embedder, populated_store)
        r.retrieve("peak traffic spike")
        assert r.last_expanded_query != ""

    def test_expanded_query_longer_than_original(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        query = "peak load"
        r = EnhancedRetriever(embedder, populated_store)
        r.retrieve(query)
        assert len(r.last_expanded_query) > len(query)

    def test_scores_descending(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        r = EnhancedRetriever(embedder, populated_store)
        results = r.retrieve("vector semantic similarity search", top_k=3)
        scores = [res.score for res in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_raises(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        r = EnhancedRetriever(embedder, populated_store)
        with pytest.raises(ValueError, match="empty"):
            r.retrieve("   ", top_k=1)


class TestExpandQuery:
    def test_returns_non_empty_string(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        r = EnhancedRetriever(embedder, populated_store)
        expanded = r.expand_query("How does the system handle peak load?")
        assert isinstance(expanded, str)
        assert len(expanded) > 0

    def test_peak_load_query_contains_relevant_terms(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        r = EnhancedRetriever(embedder, populated_store)
        expanded = r.expand_query("How does the system handle peak load?")
        # The mock expander should include load-related terms
        relevant_terms = ["load balanc", "autoscal", "circuit", "elasticity", "scal"]
        assert any(t in expanded.lower() for t in relevant_terms), (
            f"Expected at least one relevant term in expanded query: {expanded!r}"
        )


class TestMockingGCPSDK:
    """
    Demonstrate that the GCP SDK can be fully patched out.
    These tests show the pattern used in CI where no GCP credentials exist.
    """

    def test_generative_model_can_be_patched(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        """
        Patch MockGenerativeModel with a MagicMock and assert it's called.
        In real CI this would patch 'vertexai.generative_models.GenerativeModel'.
        """
        fake_response = MagicMock()
        fake_response.text = "expanded: load balancer traffic distribution autoscaling"

        fake_model = MagicMock(spec=MockGenerativeModel)
        fake_model.generate_content.return_value = fake_response

        retriever = EnhancedRetriever(embedder, populated_store, fake_model)
        results = retriever.retrieve("How does the system handle peak load?", top_k=3)

        # Model was called exactly once
        fake_model.generate_content.assert_called_once()

        # The expanded query came from our mock
        assert retriever.last_expanded_query == fake_response.text
        assert len(results) == 3

    def test_text_embedding_model_can_be_patched(
        self, embedder: Embedder, populated_store: VectorStore
    ) -> None:
        """
        Show that MockTextEmbeddingModel can replace the real SDK.
        Patches the import path for the SDK class.
        """
        with patch(
            "src.embedding.mock_vertex.MockTextEmbeddingModel.get_embeddings"
        ) as mock_get_emb:
            import numpy as np

            # Return a valid normalised vector
            v = np.random.default_rng(0).standard_normal(embedder.embedding_dim).astype(np.float32)
            v /= np.linalg.norm(v)

            class FakeValues:
                values = v.tolist()

            mock_get_emb.return_value = [FakeValues()]
            from src.embedding.mock_vertex import MockTextEmbeddingModel

            model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
            result = model.get_embeddings(["test"])
            assert len(result) == 1
            mock_get_emb.assert_called_once_with(["test"])
