"""
embedder.py
-----------
Local embedding engine that simulates Vertex AI textembedding-gecko behaviour.

Backend priority (auto-detected at runtime):
  1. sentence-transformers  all-MiniLM-L6-v2  (best semantic quality)
  2. sklearn TF-IDF sparse vectors, L2-normalised (reliable fallback)
  3. Pure NumPy hashing  (last resort, always available)

In production swap internals for:
    from vertexai.language_models import TextEmbeddingModel
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    embeddings = model.get_embeddings(texts)
The public interface is intentionally identical.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import List

import numpy as np

logger = logging.getLogger(__name__)
_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


# ── helpers ──────────────────────────────────────────────────────────────────

def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _has_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


# ── backends ─────────────────────────────────────────────────────────────────

class _SentenceTransformerBackend:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("Backend: sentence-transformers  dim=%d", self.embedding_dim)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True,
                                  show_progress_bar=False).astype(np.float32)


class _SklearnBackend:
    """
    TF-IDF vectoriser producing sparse float32 vectors.

    We do NOT apply SVD because TruncatedSVD output dimensionality is
    bounded by min(n_samples, n_features, n_components), which makes the
    dimension unstable at query time (query = 1 sample).  Instead we keep
    the raw TF-IDF vectors and hash-bucket them into a fixed 512-dim dense
    space using random projections — guaranteeing a constant dimension.
    """
    DIM = 512

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode",
            max_features=4096,   # cap vocabulary for stable projection
        )
        self.embedding_dim: int = self.DIM
        self._fitted = False
        self._fit_texts: List[str] = []
        # Fixed random projection matrix (seed=42 for reproducibility)
        rng = np.random.default_rng(42)
        self._proj: np.ndarray | None = None  # built lazily once vocab is known
        logger.info("Backend: sklearn TF-IDF + random projection  dim=%d", self.embedding_dim)

    def _ensure_fitted(self, texts: List[str]) -> None:
        new = [t for t in texts if t not in self._fit_texts]
        if not self._fitted or new:
            combined = list(dict.fromkeys(self._fit_texts + list(texts)))
            self._vectorizer.fit(combined)
            vocab_size = len(self._vectorizer.vocabulary_)
            rng = np.random.default_rng(42)
            # Random Gaussian projection matrix (vocab_size × DIM)
            self._proj = (rng.standard_normal((vocab_size, self.DIM)) / (vocab_size ** 0.5)).astype(np.float32)
            self._fitted = True
            self._fit_texts = combined

    def encode(self, texts: List[str]) -> np.ndarray:
        self._ensure_fitted(texts)
        tfidf = self._vectorizer.transform(texts)          # sparse (N, vocab)
        dense = (tfidf @ self._proj).toarray() if hasattr(tfidf @ self._proj, 'toarray') \
            else np.asarray(tfidf @ self._proj)            # (N, DIM)
        return dense.astype(np.float32)


class _HashingBackend:
    """Pure-NumPy fallback using term-hash random projections."""
    DIM = 256

    def __init__(self) -> None:
        self.embedding_dim = self.DIM
        logger.info("Backend: NumPy hashing  dim=%d", self.embedding_dim)

    def _term_vec(self, term: str) -> np.ndarray:
        seed = int(hashlib.md5(term.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
        return np.random.default_rng(seed).standard_normal(self.DIM).astype(np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            terms = re.findall(r"[a-z]+", text.lower())
            if terms:
                vecs = np.stack([self._term_vec(t) for t in terms])
                result[i] = vecs.mean(axis=0)
        return result


def _build_backend(model_name: str):
    if _has_sentence_transformers():
        logger.info("Using sentence-transformers backend.")
        return _SentenceTransformerBackend(model_name)
    if _has_sklearn():
        logger.info("sentence-transformers not found — using sklearn TF-IDF + projection backend.")
        return _SklearnBackend()
    logger.warning("Falling back to NumPy hashing backend.")
    return _HashingBackend()


# ── public class ─────────────────────────────────────────────────────────────

class Embedder:
    """
    Generates dense vector embeddings for text.

    Auto-selects best available backend:
    sentence-transformers → sklearn TF-IDF+projection → NumPy hashing.

    Parameters
    ----------
    model_name : str
        Model name for sentence-transformers (ignored by other backends).
    normalize : bool
        If True (default), L2-normalise every output vector.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME, normalize: bool = True) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self._backend = _build_backend(model_name)
        self.embedding_dim: int = self._backend.embedding_dim

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts. Returns shape (N, embedding_dim), dtype float32."""
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")
        vectors = self._backend.encode(texts).astype(np.float32)
        if self.normalize:
            vectors = self._l2_normalize(vectors)
        return vectors

    def embed_single(self, text: str) -> np.ndarray:
        """Embed one text. Returns 1-D vector of shape (embedding_dim,)."""
        return self.embed([text])[0]

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def __repr__(self) -> str:
        return f"Embedder(model={self.model_name!r}, dim={self.embedding_dim}, normalize={self.normalize})"
