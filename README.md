# 🔍 Semantic RAG & Vector Search Engine

> **Senior Gen AI Assessment** — Context-Aware Retrieval Engine  
> GCP Stack Focus | Embeddings · Vector Databases · Retrieval Logic · Benchmarking

---

## 📁 Project Structure

```
rag_project/
├── README.md                         ← You are here
├── requirements.txt                  ← All dependencies
├── setup.py                          ← Package setup
├── .env.example                      ← Environment variable template
│
├── src/
│   ├── __init__.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── embedder.py               ← Local SentenceTransformer embedder (mocks gecko)
│   │   └── mock_vertex.py            ← Mock VertexAI TextEmbeddingModel & GenerativeModel
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   └── vector_store.py           ← FAISS-backed vector store
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py              ← Strategy A: raw vector search
│   │   └── enhanced_retriever.py     ← Strategy B: AI query-expansion + search
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── benchmarker.py            ← Runs A vs B comparison, outputs JSON + table
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   ← Shared pytest fixtures
│   ├── test_embedder.py              ← Unit tests for embedding logic
│   ├── test_vector_store.py          ← Unit tests for FAISS storage
│   ├── test_retriever.py             ← Unit tests for Strategy A retrieval
│   ├── test_enhanced_retriever.py    ← Unit tests for Strategy B retrieval (mocked GCP)
│   └── test_benchmarker.py           ← Integration tests for full benchmark pipeline
│
├── data/
│   └── corpus.py                     ← 10 technical paragraphs used as the corpus
│
└── docs/
    ├── retrieval_benchmark.md        ← 📊 Strategy A vs B comparison report (generated)
    └── architecture.md               ← Design decisions & Vertex AI migration guide
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
# Clone the repo
git clone <your-repo-url>
cd rag_project

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Run the Benchmark (Strategy A vs B)

```bash
python -m src.evaluation.benchmarker
```

This will:
1. Ingest 10 technical paragraphs into the FAISS vector store
2. Run 3 complex queries through **Strategy A** (raw vector search)
3. Run the same queries through **Strategy B** (AI query-expansion first)
4. Print a side-by-side comparison table to the console
5. Write `docs/retrieval_benchmark.md` with full results

### 3. Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# A specific suite
pytest tests/test_benchmarker.py -v
```

---

## ⚙️ How It Works

### Ingestion Pipeline

```
Raw Text Corpus
      │
      ▼
 Embedder (SentenceTransformer all-MiniLM-L6-v2)
      │   simulates Vertex AI textembedding-gecko
      ▼
 FAISS Index (cosine similarity via inner product on L2-normalised vectors)
      │
      ▼
 VectorStore (in-memory, serialisable to disk)
```

### Strategy A — Raw Vector Search

```
User Query  ──embed──►  Query Vector  ──FAISS search──►  Top-K Chunks
```

### Strategy B — AI-Enhanced Retrieval (Query Expansion)

```
User Query
    │
    ▼
MockGenerativeModel.expand_query()
    │   rewrites query into a richer, embedding-friendly formulation
    ▼
Expanded Query  ──embed──►  Query Vector  ──FAISS search──►  Top-K Chunks
```

The `MockGenerativeModel` simulates what you would call via  
`vertexai.generative_models.GenerativeModel("gemini-pro")` in production.

---

## 🧪 Mocking Strategy

All GCP SDK calls are fully mocked — **no credentials required**:

| Real GCP Class | Mock Location | Behaviour |
|---|---|---|
| `vertexai.language_models.TextEmbeddingModel` | `src/embedding/mock_vertex.py` | Delegates to local SentenceTransformer |
| `vertexai.generative_models.GenerativeModel` | `src/embedding/mock_vertex.py` | Uses a rule-based query rewriter |

In tests, `unittest.mock.patch` is used to swap in the mocks at the import boundary, keeping production code clean.

---

## 📐 Similarity Metric: Cosine vs Euclidean

**We use Cosine Similarity.**

| Metric | Formula | Behaviour |
|---|---|---|
| **Cosine** | `1 - (A·B / \|A\|\|B\|)` | Measures *angle* between vectors — invariant to magnitude |
| **Euclidean** | `sqrt(Σ(aᵢ-bᵢ)²)` | Measures *distance* in absolute space |

**Why Cosine for text embeddings:**

- Sentence-transformer models (and gecko) produce embeddings whose *direction* encodes semantic meaning; magnitude is not meaningful.
- A short sentence and a long paragraph about the same topic should score similarly — cosine achieves this, euclidean penalises the length difference.
- Cosine is efficiently implemented in FAISS via `IndexFlatIP` on L2-normalised vectors (inner-product of unit vectors = cosine similarity).

---

## ☁️ Migrating to Vertex AI Vector Search (Matching Engine)

See [`docs/architecture.md`](docs/architecture.md) for the full migration guide. In brief:

| Layer | Local (this repo) | Production (Vertex AI) |
|---|---|---|
| **Embedding** | `sentence-transformers` | `textembedding-gecko@003` via `TextEmbeddingModel.get_embeddings()` |
| **Index** | `faiss.IndexFlatIP` (in-memory) | Vertex AI Matching Engine — Approximate Nearest Neighbour at scale |
| **Query Expansion** | `MockGenerativeModel` | `gemini-1.5-pro` via `GenerativeModel.generate_content()` |
| **Deployment** | Local Python process | Cloud Run / Vertex AI Pipelines |

---

## 📊 Benchmark Results

After running `python -m src.evaluation.benchmarker`, see [`docs/retrieval_benchmark.md`](docs/retrieval_benchmark.md).

---

## 🛠️ Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Local embedding model (simulates gecko) |
| `faiss-cpu` | High-performance vector similarity search |
| `numpy` | Vector operations |
| `tabulate` | Pretty-print comparison tables |
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-mock` | Convenient mock fixtures |
| `colorama` | Coloured terminal output |

---

## 📝 Notes

- No GCP account or API key is needed — everything runs locally and offline.
- The FAISS index is re-built in memory on each run (deterministic, fast for this corpus size).
- All randomness is seeded for reproducible results.

---

## 🔧 Environment Notes

### Embedding Backend Auto-detection

This project auto-selects the best available embedding backend:

| Priority | Library | Quality | Install |
|---|---|---|---|
| 1 | `sentence-transformers` | Best (neural) | `pip install sentence-transformers` |
| 2 | `sklearn` TF-IDF + random projection | Good (lexical) | Already in `requirements.txt` |
| 3 | NumPy hashing | Basic | Always available |

The pipeline produces identical results regardless of backend — only retrieval quality changes.

### Running Tests Without `pytest`

If `pytest` cannot be installed (e.g. no internet access), use the built-in runner:

```bash
python run_tests.py
```

This discovers all `test_*.py` files, resolves fixtures, and reports pass/fail — no external dependencies needed. When `pytest` IS available, just use it normally:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```
