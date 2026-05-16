"""
test_embedder.py
----------------
Unit tests for src/embedding/embedder.py

Tests cover:
  - Output shape and dtype
  - L2 normalisation
  - Single-text embedding
  - Batch embedding
  - Empty-input guard
  - Semantic similarity ordering (backend-agnostic)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.embedding.embedder import Embedder


class TestEmbedderInit:
    def test_default_model_loads(self, embedder: Embedder) -> None:
        assert embedder is not None

    def test_embedding_dim_positive(self, embedder: Embedder) -> None:
        # Backend-agnostic: dimension must be > 0
        assert embedder.embedding_dim > 0

    def test_repr_contains_model_name(self, embedder: Embedder) -> None:
        r = repr(embedder)
        assert "Embedder" in r
        # dim should appear somewhere in the repr
        assert str(embedder.embedding_dim) in r


class TestEmbedBatch:
    def test_output_shape(self, embedder: Embedder) -> None:
        texts = ["hello world", "distributed systems", "vector search"]
        vectors = embedder.embed(texts)
        assert vectors.shape == (3, embedder.embedding_dim)

    def test_output_dtype_float32(self, embedder: Embedder) -> None:
        vectors = embedder.embed(["test"])
        assert vectors.dtype == np.float32

    def test_l2_normalised(self, embedder: Embedder) -> None:
        vectors = embedder.embed(["peak load balancing", "semantic embeddings"])
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)

    def test_empty_list_raises(self, embedder: Embedder) -> None:
        with pytest.raises(ValueError, match="empty"):
            embedder.embed([])

    def test_single_text_batch(self, embedder: Embedder) -> None:
        vectors = embedder.embed(["hello"])
        assert vectors.shape == (1, embedder.embedding_dim)


class TestEmbedSingle:
    def test_output_shape_1d(self, embedder: Embedder) -> None:
        v = embedder.embed_single("hello")
        assert v.shape == (embedder.embedding_dim,)

    def test_l2_normalised(self, embedder: Embedder) -> None:
        v = embedder.embed_single("peak traffic handling")
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-5

    def test_consistent_with_batch(self, embedder: Embedder) -> None:
        text = "load balancer distributes requests"
        v_single = embedder.embed_single(text)
        v_batch = embedder.embed([text])[0]
        np.testing.assert_allclose(v_single, v_batch, atol=1e-6)


class TestSemanticOrdering:
    """
    Verify that the embed → cosine-score pipeline is internally consistent.
    We use same-text = score ~1.0 as the key sanity check; ordering tests
    are intentionally soft because the TF-IDF backend (used when
    sentence-transformers is unavailable) has weaker semantic ordering than
    a neural model.
    """

    def test_identical_texts_score_1(self, embedder: Embedder) -> None:
        text = "peak load handling via autoscaling"
        v1 = embedder.embed_single(text)
        v2 = embedder.embed_single(text)
        score = float(np.dot(v1, v2))
        assert abs(score - 1.0) < 1e-4

    def test_cosine_scores_bounded(self, embedder: Embedder) -> None:
        texts = [
            "load balancer traffic distribution",
            "vector similarity search",
            "chocolate cake recipe",
        ]
        vecs = embedder.embed(texts)
        # All cosine similarities of unit vectors lie in [-1, 1]
        for i in range(len(texts)):
            for j in range(len(texts)):
                score = float(np.dot(vecs[i], vecs[j]))
                assert -1.01 <= score <= 1.01

    def test_self_similarity_highest(self, embedder: Embedder) -> None:
        """Each text should be most similar to itself."""
        texts = [
            "load balancer distributes traffic",
            "vector embeddings semantic search",
        ]
        vecs = embedder.embed(texts)
        for i, v in enumerate(vecs):
            self_score = float(np.dot(v, v))
            for j, u in enumerate(vecs):
                if i != j:
                    cross_score = float(np.dot(v, u))
                    assert self_score >= cross_score - 1e-5
