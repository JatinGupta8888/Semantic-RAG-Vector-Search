"""
enhanced_retriever.py  —  Strategy B: AI-Enhanced Retrieval
------------------------------------------------------------
Applies generative query expansion before embedding, closing the lexical
gap between how users phrase questions and how answers are phrased in the
corpus.

Pipeline
~~~~~~~~
  User Query
      │
      ▼
  MockGenerativeModel.generate_content()   (or real Gemini in prod)
      │   ← enriches with related technical terms
      ▼
  Expanded Query  ──embed──►  Query Vector  ──FAISS search──►  Top-K Results

Why this works
~~~~~~~~~~~~~~
Sentence-transformer models produce embeddings that reflect the semantic
centroid of the input text.  An expanded query that explicitly mentions
related concepts shifts the centroid toward the cluster of relevant
documents, improving recall — especially for short or ambiguous queries.

Production note
~~~~~~~~~~~~~~~
Replace ``MockGenerativeModel`` with:

    from vertexai.generative_models import GenerativeModel
    model = GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(expansion_prompt)
    expanded_query = response.text
"""

from __future__ import annotations

import logging
from typing import List

from src.embedding.embedder import Embedder
from src.embedding.mock_vertex import MockGenerativeModel
from src.storage.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

# Prompt template sent to the generative model for query expansion.
_EXPANSION_PROMPT_TEMPLATE = (
    "You are a query expansion assistant helping a semantic search system. "
    "Rewrite the following user query into a richer formulation that includes "
    "related technical terms, synonyms, and adjacent concepts. "
    "Return only the expanded query — no explanation, no preamble.\n\n"
    "Original query: {query}"
)


class EnhancedRetriever:
    """
    Strategy B — generative query expansion + vector search.

    Parameters
    ----------
    embedder : Embedder
        Embedding model used to encode the (expanded) query.
    vector_store : VectorStore
        Populated vector store to search against.
    generative_model : MockGenerativeModel | None
        Generative model for query expansion.  If ``None``, a new
        ``MockGenerativeModel`` is instantiated automatically.

    Example
    -------
    >>> retriever = EnhancedRetriever(embedder, store)
    >>> results = retriever.retrieve("How does the system handle peak load?", top_k=3)
    >>> print(retriever.last_expanded_query)
    """

    STRATEGY_NAME = "Strategy B — AI-Enhanced Retrieval (Query Expansion)"

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        generative_model: MockGenerativeModel | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.generative_model = generative_model or MockGenerativeModel(
            "gemini-1.5-pro"
        )
        # Stores the last expanded query for inspection / benchmarking
        self.last_expanded_query: str = ""

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Expand ``query`` via the generative model, then retrieve top-k results.

        Parameters
        ----------
        query : str
            Raw user query.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[SearchResult]
            Ranked by descending cosine similarity against the expanded query.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")

        logger.info("[Strategy B] Original query: %r", query)

        # --- Step 1: Query Expansion ---
        prompt = query
        response = self.generative_model.generate_content(prompt)
        expanded_query = response.text.strip()
        self.last_expanded_query = expanded_query
        logger.info("[Strategy B] Expanded query: %r", expanded_query)

        # --- Step 2: Embed expanded query ---
        query_vector = self.embedder.embed_single(expanded_query)

        # --- Step 3: FAISS search ---
        results = self.vector_store.search(query_vector, top_k=top_k)
        logger.info("[Strategy B] Retrieved %d results.", len(results))
        return results

    def expand_query(self, query: str) -> str:
        """
        Public helper — return the expanded query without performing retrieval.
        Useful for debugging and unit tests.
        """
        prompt = query
        response = self.generative_model.generate_content(prompt)
        return response.text.strip()

    def __repr__(self) -> str:
        return (
            f"EnhancedRetriever(embedder={self.embedder!r}, "
            f"store_size={len(self.vector_store)}, "
            f"model={self.generative_model!r})"
        )
