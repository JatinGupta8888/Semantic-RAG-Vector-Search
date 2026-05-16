# Architecture & Design Decisions

## 1. Similarity Metric: Cosine vs Euclidean

### Summary

We use **cosine similarity** throughout this project.

### Why not Euclidean?

| Aspect | Cosine | Euclidean |
|---|---|---|
| **What it measures** | Angle between vectors | Absolute distance in vector space |
| **Sensitive to magnitude?** | No | Yes |
| **Good for text?** | ✅ Yes | ⚠️ Partially |
| **FAISS support** | `IndexFlatIP` (on unit vectors) | `IndexFlatL2` |

Sentence-transformer models (and Vertex AI's `textembedding-gecko`) encode **semantic meaning in the direction** of the vector, not its magnitude. A 10-word sentence and a 200-word paragraph about the same topic will produce vectors pointing in similar directions but with different magnitudes. Cosine similarity correctly scores them as similar; Euclidean distance penalises the magnitude difference.

### Implementation

We implement cosine similarity efficiently in FAISS:

```python
# 1. L2-normalise every vector at ingest and query time
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# 2. Use IndexFlatIP (inner product)
index = faiss.IndexFlatIP(embedding_dim)

# For unit vectors: inner_product(A, B) == cosine_similarity(A, B)
```

This approach matches the performance of a dedicated cosine index while reusing FAISS's highly optimised inner-product kernel.

---

## 2. Embedding Model Choice

| | Local (this repo) | Production |
|---|---|---|
| **Model** | `all-MiniLM-L6-v2` | `textembedding-gecko@003` |
| **Dimensions** | 384 | 768 |
| **Library** | `sentence-transformers` | `google-cloud-aiplatform` |
| **Credentials needed?** | No | Yes (ADC / Service Account) |

`all-MiniLM-L6-v2` was chosen because:
- It runs offline with no API keys.
- Its semantic quality is sufficient for benchmarking retrieval strategies.
- The 384-dimension output is a good proxy for gecko's 768-dimension output; FAISS handles both trivially.

Swapping to gecko requires only changing the `Embedder` class internals — the rest of the pipeline is model-agnostic.

---

## 3. FAISS Index Choice

We use `IndexFlatIP` — an **exact** inner-product search index.

For this corpus size (10 documents), exact search is instantaneous and eliminates recall concerns.  At scale, the migration path is:

| Corpus Size | Recommended Index | Recall | Query Time |
|---|---|---|---|
| < 100K vectors | `IndexFlatIP` (exact) | 100% | O(n) |
| 100K – 10M | `IndexHNSWFlat` | ~98% | O(log n) |
| > 10M | `IndexIVF_PQ` | ~95% | O(1) approx. |
| Cloud-scale | Vertex AI Matching Engine | ~98% (ANN) | < 10ms (managed) |

---

## 4. Query Expansion Strategy

Strategy B uses a `MockGenerativeModel` that rewrites the user query using rule-based expansion.  In production this is replaced by Gemini Pro, which produces semantically richer expansions.

**Why query expansion works for RAG:**

The user asks: `"How does the system handle peak load?"`

The corpus document says: `"load balancers distribute incoming requests... autoscaling provisions instances in response to real-time demand..."`

There is a **lexical gap** — the query does not contain the word "autoscaling" or "circuit breaker". A raw vector embedding of the short query sits in a slightly different region of the embedding space than the longer, richer corpus document.

Expansion bridges this gap:

```
Original: "How does the system handle peak load?"
Expanded: "How does the system handle peak load? — including load balancing 
           strategies, horizontal autoscaling, elasticity, connection pooling, 
           circuit breakers, graceful degradation under stress"
```

The expanded query's embedding vector sits much closer to the cluster of relevant documents.

---

## 5. Migrating to Vertex AI Vector Search (Matching Engine)

### Step-by-step migration

#### Phase 1 — Swap the embedding model

```python
# Before (local)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts)

# After (Vertex AI)
import vertexai
from vertexai.language_models import TextEmbeddingModel
vertexai.init(project="your-project", location="us-central1")
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
embeddings = model.get_embeddings(texts)
vectors = [e.values for e in embeddings]
```

The `Embedder` class in this repo is designed so this swap requires changing only the constructor internals. The `embed()` and `embed_single()` public methods stay identical.

#### Phase 2 — Swap the vector store

```python
# Before (FAISS, local)
import faiss
index = faiss.IndexFlatIP(768)
index.add(vectors)

# After (Vertex AI Matching Engine)
from google.cloud import aiplatform
aiplatform.init(project="your-project", location="us-central1")

# Create an index (done once)
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="rag-index",
    dimensions=768,
    approximate_neighbors_count=10,
    distance_measure_type="DOT_PRODUCT_DISTANCE",  # = cosine for unit vectors
)

# Deploy the index to an endpoint
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="rag-endpoint",
    public_endpoint_enabled=True,
)
endpoint.deploy_index(index=index, deployed_index_id="rag-deployed-index")

# Query
response = endpoint.find_neighbors(
    deployed_index_id="rag-deployed-index",
    queries=[query_vector.tolist()],
    num_neighbors=3,
)
```

#### Phase 3 — Swap the generative model

```python
# Before (mock)
from src.embedding.mock_vertex import MockGenerativeModel
model = MockGenerativeModel("gemini-1.5-pro")

# After (real Gemini)
from vertexai.generative_models import GenerativeModel
model = GenerativeModel("gemini-1.5-pro")
response = model.generate_content(expansion_prompt)
expanded_query = response.text
```

### Cloud architecture diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Client (Cloud Run / Vertex AI Pipeline)                     │
│                                                              │
│   User Query                                                 │
│       │                                                      │
│       ▼                                                      │
│   Gemini Pro (Query Expansion)                               │
│       │                                                      │
│       ▼                                                      │
│   textembedding-gecko@003 (Embedding)                        │
│       │                                                      │
│       ▼                                                      │
│   Vertex AI Matching Engine (ANN Search)                     │
│       │                                                      │
│       ▼                                                      │
│   Top-K Chunks → Gemini Pro (Answer Generation)              │
│       │                                                      │
│       ▼                                                      │
│   Response to User                                           │
└─────────────────────────────────────────────────────────────┘
```

### Infrastructure considerations

| Concern | Local | Vertex AI |
|---|---|---|
| **Latency** | < 1 ms FAISS | < 10 ms Matching Engine |
| **Scale** | Single process | Billions of vectors |
| **Availability** | None | 99.9% SLA |
| **Cost** | Free | Pay-per-query + index hosting |
| **Auth** | Not required | ADC / Service Account required |
| **Monitoring** | Logging only | Cloud Monitoring + Trace |

---

## 6. Chunking Strategy (not in scope, but noted)

For production RAG with large documents:

- **Chunk size**: 256–512 tokens with 10–15% overlap.
- **Chunking method**: Recursive character splitter (respects sentence and paragraph boundaries).
- **Metadata**: Store source document ID, page number, and section title alongside each chunk for citation.
- **Re-ranking**: Add a cross-encoder re-ranking step after the top-50 ANN results to re-score the top-10 before injecting into the LLM prompt.

---

_Last updated: 2025_
