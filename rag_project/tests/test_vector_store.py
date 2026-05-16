"""
test_vector_store.py
--------------------
Unit tests for src/storage/vector_store.py

Tests cover:
  - Ingestion (single and batch)
  - Index size tracking
  - Search returns correct types and ranking order
  - Empty store raises ValueError
  - Wrong-shape query raises RuntimeError
  - Missing document keys raise ValueError
  - Reset clears state
  - Top-k clamped to corpus size
"""
from __future__ import annotations

import numpy as np
import pytest

from src.embedding.embedder import Embedder
from src.storage.vector_store import Document, SearchResult, VectorStore
from tests.conftest import MINI_CORPUS


class TestVectorStoreInit:
    def test_empty_on_creation(self, empty_store: VectorStore) -> None:
        assert len(empty_store) == 0
        assert empty_store.size == 0

    def test_repr(self, empty_store: VectorStore) -> None:
        r = repr(empty_store)
        assert "VectorStore" in r
        assert "0" in r


class TestIngestion:
    def test_ingest_increases_size(self, embedder: Embedder, empty_store: VectorStore) -> None:
        empty_store.ingest(MINI_CORPUS, embedder)
        assert len(empty_store) == len(MINI_CORPUS)

    def test_ingest_missing_key_raises(self, embedder: Embedder, empty_store: VectorStore) -> None:
        bad_doc = [{"id": "x", "text": "oops"}]  # missing "title"
        with pytest.raises(ValueError, match="title"):
            empty_store.ingest(bad_doc, embedder)

    def test_populated_store_fixture_size(self, populated_store: VectorStore) -> None:
        assert len(populated_store) == len(MINI_CORPUS)


class TestSearch:
    def test_returns_list_of_search_results(self, populated_store: VectorStore, embedder: Embedder) -> None:
        q = embedder.embed_single("traffic distribution")
        results = populated_store.search(q, top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_top_k_respected(self, populated_store: VectorStore, embedder: Embedder) -> None:
        q = embedder.embed_single("semantic search")
        results = populated_store.search(q, top_k=2)
        assert len(results) == 2

    def test_scores_descending(self, populated_store: VectorStore, embedder: Embedder) -> None:
        q = embedder.embed_single("load balancer")
        results = populated_store.search(q, top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_ascending(self, populated_store: VectorStore, embedder: Embedder) -> None:
        q = embedder.embed_single("vector embeddings")
        results = populated_store.search(q, top_k=3)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_scores_bounded(self, populated_store: VectorStore, embedder: Embedder) -> None:
        """Cosine similarity of normalised vectors is in [-1, 1]."""
        q = embedder.embed_single("circuit breaker resilience")
        results = populated_store.search(q, top_k=3)
        for r in results:
            assert -1.01 <= r.score <= 1.01

    def test_top_result_id_is_valid(self, populated_store: VectorStore, embedder: Embedder) -> None:
        """Top result ID must be from the known corpus."""
        valid_ids = {d["id"] for d in MINI_CORPUS}
        q = embedder.embed_single("load balancer distributes traffic across servers")
        results = populated_store.search(q, top_k=1)
        assert results[0].document.id in valid_ids

    def test_top_result_has_non_empty_text(self, populated_store: VectorStore, embedder: Embedder) -> None:
        q = embedder.embed_single("load balancer")
        results = populated_store.search(q, top_k=1)
        assert results[0].document.text.strip()

    def test_empty_store_raises(self, empty_store: VectorStore, random_vector: np.ndarray) -> None:
        with pytest.raises(ValueError, match="empty"):
            empty_store.search(random_vector, top_k=1)

    def test_wrong_shape_raises(self, populated_store: VectorStore, embedder: Embedder) -> None:
        bad_vector = np.zeros(embedder.embedding_dim + 10, dtype=np.float32)
        with pytest.raises(RuntimeError):
            populated_store.search(bad_vector, top_k=1)

    def test_top_k_clamped_to_corpus_size(self, populated_store: VectorStore, embedder: Embedder) -> None:
        q = embedder.embed_single("resilience pattern")
        results = populated_store.search(q, top_k=999)
        assert len(results) == len(MINI_CORPUS)


class TestReset:
    def test_reset_clears_documents(self, embedder: Embedder, empty_store: VectorStore) -> None:
        empty_store.ingest(MINI_CORPUS, embedder)
        assert len(empty_store) > 0
        empty_store.reset()
        assert len(empty_store) == 0

    def test_can_ingest_after_reset(self, embedder: Embedder, empty_store: VectorStore) -> None:
        empty_store.ingest(MINI_CORPUS, embedder)
        empty_store.reset()
        empty_store.ingest(MINI_CORPUS[:1], embedder)
        assert len(empty_store) == 1
