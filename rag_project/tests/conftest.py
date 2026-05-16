"""
conftest.py
-----------
Shared pytest fixtures for the entire test suite.

All fixtures use a small 3-document corpus so tests are fast and deterministic.
The ``embedder`` fixture loads the real SentenceTransformer model once per
session to avoid reloading weights for every test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embedding.embedder import Embedder
from src.embedding.mock_vertex import MockGenerativeModel, MockTextEmbeddingModel
from src.storage.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Tiny corpus used across all tests
# ---------------------------------------------------------------------------

MINI_CORPUS = [
    {
        "id": "t001",
        "title": "Load Balancing",
        "text": (
            "Load balancers distribute incoming requests across multiple servers "
            "to prevent bottlenecks and ensure high availability during peak traffic."
        ),
    },
    {
        "id": "t002",
        "title": "Vector Embeddings",
        "text": (
            "Vector embeddings represent text as dense numerical vectors, enabling "
            "semantic similarity search via cosine or dot-product metrics."
        ),
    },
    {
        "id": "t003",
        "title": "Circuit Breakers",
        "text": (
            "Circuit breakers prevent cascading failures by short-circuiting calls "
            "to degraded services and returning fallback responses immediately."
        ),
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """Real Embedder — loaded once per test session."""
    return Embedder()


@pytest.fixture(scope="session")
def mock_text_embedding_model() -> MockTextEmbeddingModel:
    return MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")


@pytest.fixture(scope="session")
def mock_generative_model() -> MockGenerativeModel:
    return MockGenerativeModel("gemini-1.5-pro")


@pytest.fixture
def populated_store(embedder: Embedder) -> VectorStore:
    """A VectorStore pre-loaded with MINI_CORPUS."""
    store = VectorStore(embedding_dim=embedder.embedding_dim)
    store.ingest(MINI_CORPUS, embedder)
    return store


@pytest.fixture
def empty_store(embedder: Embedder) -> VectorStore:
    """A VectorStore with no documents."""
    return VectorStore(embedding_dim=embedder.embedding_dim)


@pytest.fixture
def random_vector(embedder: Embedder) -> np.ndarray:
    """A random L2-normalised vector matching the embedder dimension."""
    rng = np.random.default_rng(42)
    v = rng.standard_normal(embedder.embedding_dim).astype(np.float32)
    return v / np.linalg.norm(v)
