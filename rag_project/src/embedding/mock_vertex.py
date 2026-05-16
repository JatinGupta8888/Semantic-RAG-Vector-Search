"""
mock_vertex.py
--------------
Drop-in mocks for Google Cloud Vertex AI SDK classes.

These mocks are used in two ways:
  1. Directly in application code when GCP credentials are absent (local dev).
  2. Patched over the real SDK in pytest tests via ``unittest.mock.patch``.

Mocked classes
~~~~~~~~~~~~~~
- ``MockTextEmbeddingModel``  ←→  ``vertexai.language_models.TextEmbeddingModel``
- ``MockGenerativeModel``     ←→  ``vertexai.generative_models.GenerativeModel``

Design goals
~~~~~~~~~~~~
- No GCP credentials or network calls required.
- Behaviour is deterministic and realistic enough to validate the pipeline.
- The public method signatures exactly mirror the real SDK.
"""

from __future__ import annotations

import logging
import re
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: lazily import the local Embedder to avoid circular deps
# ---------------------------------------------------------------------------


def _get_local_embedder():
    """Return a module-level singleton Embedder to avoid reloading the model."""
    global _SHARED_EMBEDDER  # noqa: PLW0603
    if "_SHARED_EMBEDDER" not in globals():
        from src.embedding.embedder import Embedder

        _SHARED_EMBEDDER = Embedder()
    return _SHARED_EMBEDDER


# ---------------------------------------------------------------------------
# TextEmbeddingModel mock
# ---------------------------------------------------------------------------


class _MockEmbeddingValues:
    """Mimics the object returned by the real TextEmbeddingModel."""

    def __init__(self, vector: List[float]) -> None:
        self.values: List[float] = vector


class MockTextEmbeddingModel:
    """
    Mock of ``vertexai.language_models.TextEmbeddingModel``.

    Delegates to the local ``sentence-transformers`` model so that the
    embedding semantics are real, even though GCP is not involved.

    Usage (mirrors real SDK)
    ------------------------
    >>> model = MockTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    >>> results = model.get_embeddings(["hello world"])
    >>> vector = results[0].values   # list[float]
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        logger.debug("MockTextEmbeddingModel initialised with model=%r", model_name)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "MockTextEmbeddingModel":
        """Factory method — mirrors the real SDK."""
        return cls(model_name)

    def get_embeddings(self, texts: List[str]) -> List[_MockEmbeddingValues]:
        """
        Return a list of embedding value objects, one per input text.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        list[_MockEmbeddingValues]
            Each object has a ``.values`` attribute (``list[float]``).
        """
        embedder = _get_local_embedder()
        vectors: np.ndarray = embedder.embed(texts)
        return [_MockEmbeddingValues(v.tolist()) for v in vectors]

    def __repr__(self) -> str:
        return f"MockTextEmbeddingModel(model={self._model_name!r})"


# ---------------------------------------------------------------------------
# GenerativeModel mock  (used for query expansion in Strategy B)
# ---------------------------------------------------------------------------


class _MockGenerationResponse:
    """Mimics ``vertexai.generative_models.GenerationResponse``."""

    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f"_MockGenerationResponse(text={self._text[:60]!r}…)"


class MockGenerativeModel:
    """
    Mock of ``vertexai.generative_models.GenerativeModel``.

    Implements a rule-based query rewriter that enriches the user query with
    domain-relevant terminology — simulating what Gemini Pro would produce.

    The expansion rules are intentionally simple and transparent so that the
    benchmark output is deterministic and easy to interpret.

    Usage (mirrors real SDK)
    ------------------------
    >>> model = MockGenerativeModel("gemini-1.5-pro")
    >>> response = model.generate_content("How does the system handle peak load?")
    >>> expanded = response.text
    """

    # Domain-specific expansion rules.
    # Each entry: (regex pattern, expansion suffix)
    _EXPANSION_RULES: list[tuple[str, str]] = [
        (
            r"peak load|high traffic|traffic spike|heavy load",
            (
                "load balancing strategies, horizontal autoscaling, elasticity, "
                "connection pooling, circuit breakers, graceful degradation under stress"
            ),
        ),
        (
            r"retriev|search|find|lookup",
            (
                "semantic search, vector similarity, approximate nearest neighbour, "
                "RAG pipeline, embedding-based retrieval, FAISS index"
            ),
        ),
        (
            r"scal|autoscal|replicas|horizontal",
            (
                "Kubernetes HPA, horizontal pod autoscaler, CPU utilisation thresholds, "
                "scale-out policy, scale-in stabilisation, elasticity"
            ),
        ),
        (
            r"cache|redis|memcache",
            (
                "cache-aside pattern, TTL expiry, write-through, thundering herd, "
                "cache hit rate, distributed caching"
            ),
        ),
        (
            r"embed|vector|semantic",
            (
                "dense vector representations, cosine similarity, high-dimensional space, "
                "sentence-transformers, textembedding-gecko, FAISS, ANN search"
            ),
        ),
        (
            r"fail|circuit.?breaker|fallback|graceful",
            (
                "circuit breaker pattern, open/closed/half-open states, "
                "fallback responses, cascading failures, microservice resilience"
            ),
        ),
        (
            r"monitor|observ|alert|SLO|latency",
            (
                "distributed tracing, p99 latency, error budget, SLO compliance, "
                "metrics and logs, incident response"
            ),
        ),
        (
            r"database|shard|postgres|replica",
            (
                "horizontal sharding, shard key selection, connection pooling, "
                "PgBouncer, read replicas, write contention"
            ),
        ),
    ]

    _SYSTEM_PROMPT = (
        "You are a query expansion assistant. Given a user query, "
        "rewrite it into a richer, embedding-friendly formulation that "
        "includes related technical concepts to improve semantic retrieval. "
        "Return only the expanded query, no preamble."
    )

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        logger.debug("MockGenerativeModel initialised with model=%r", model_name)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "MockGenerativeModel":
        return cls(model_name)

    def generate_content(self, prompt: str) -> _MockGenerationResponse:
        """
        Rewrite / expand ``prompt`` into an embedding-friendly query.

        Parameters
        ----------
        prompt : str
            The raw user query (or a prompt containing it).

        Returns
        -------
        _MockGenerationResponse
            ``.text`` contains the expanded query string.
        """
        expanded = self._rule_based_expand(prompt)
        logger.debug("Query expansion: %r → %r", prompt[:80], expanded[:120])
        return _MockGenerationResponse(expanded)

    # ------------------------------------------------------------------
    # Internal expansion logic
    # ------------------------------------------------------------------

    def _rule_based_expand(self, query: str) -> str:
        """
        Apply all matching expansion rules and concatenate their suffixes.

        If no rule matches, append a generic software-engineering expansion
        so the output is always richer than the input.
        """
        query_lower = query.lower()
        suffixes: list[str] = []

        for pattern, suffix in self._EXPANSION_RULES:
            if re.search(pattern, query_lower):
                suffixes.append(suffix)

        if suffixes:
            expansion = "; ".join(suffixes)
            expanded = f"{query.rstrip('?. ')} — including {expansion}"
        else:
            # Generic fallback
            expanded = (
                f"{query.rstrip('?. ')} — covering system design, distributed "
                "architecture, scalability, reliability, and performance optimisation"
            )

        return expanded

    def __repr__(self) -> str:
        return f"MockGenerativeModel(model={self._model_name!r})"
