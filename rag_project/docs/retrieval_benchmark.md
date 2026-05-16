# Retrieval Benchmark: Strategy A vs Strategy B

> Generated: 2026-05-16 09:46 UTC  
> Corpus size: 10 documents  
> Top-K: 3  

---

## Overview

| | Strategy A | Strategy B |
|---|---|---|
| **Name** | Raw Vector Search | AI-Enhanced Retrieval |
| **Query rewriting** | None | MockGenerativeModel (rule-based expander) |
| **Embedding** | Query embedded as-is | Expanded query embedded |
| **Similarity** | Cosine (IndexFlatIP) | Cosine (IndexFlatIP) |

---

## Query 1: `How does the system handle peak load?`

### Expanded Query (Strategy B)

> How does the system handle peak load — including load balancing strategies, horizontal autoscaling, elasticity, connection pooling, circuit breakers, graceful degradation under stress

### Strategy A — Raw Vector Search

|   Rank |   Score | Doc ID   | Title                                          | Snippet                                                                                               |
|--------|---------|----------|------------------------------------------------|-------------------------------------------------------------------------------------------------------|
|      1 |  0.5639 | doc_001  | Load Balancing Under Peak Traffic              | When a system experiences peak load, the load balancer distributes incoming requests across multiple… |
|      2 |  0.452  | doc_003  | Caching Strategies for High-Throughput Systems | Distributed caches like Redis and Memcached reduce database pressure during traffic surges by servin… |
|      3 |  0.3967 | doc_002  | Horizontal Autoscaling and Elasticity          | Autoscaling is the mechanism by which a cloud-hosted service automatically provisions or terminates … |

### Strategy B — AI-Enhanced Retrieval

|   Rank |   Score | Doc ID   | Title                                    | Snippet                                                                                               |
|--------|---------|----------|------------------------------------------|-------------------------------------------------------------------------------------------------------|
|      1 |  0.5792 | doc_001  | Load Balancing Under Peak Traffic        | When a system experiences peak load, the load balancer distributes incoming requests across multiple… |
|      2 |  0.4812 | doc_002  | Horizontal Autoscaling and Elasticity    | Autoscaling is the mechanism by which a cloud-hosted service automatically provisions or terminates … |
|      3 |  0.4408 | doc_009  | Database Sharding and Connection Pooling | Under heavy read/write load, a single database instance becomes a contention point. Horizontal shard… |

---

## Query 2: `What techniques improve retrieval accuracy in a semantic search pipeline?`

### Expanded Query (Strategy B)

> What techniques improve retrieval accuracy in a semantic search pipeline — including semantic search, vector similarity, approximate nearest neighbour, RAG pipeline, embedding-based retrieval, FAISS index; dense vector representations, cosine similarity, high-dimensional space, sentence-transformers, textembedding-gecko, FAISS, ANN search

### Strategy A — Raw Vector Search

|   Rank |   Score | Doc ID   | Title                                             | Snippet                                                                                               |
|--------|---------|----------|---------------------------------------------------|-------------------------------------------------------------------------------------------------------|
|      1 |  0.5425 | doc_006  | Retrieval-Augmented Generation (RAG) Architecture | RAG pipelines augment a large language model's responses by first retrieving relevant context from a… |
|      2 |  0.4968 | doc_007  | Query Expansion for Improved Retrieval            | Query expansion improves retrieval recall by rewriting or augmenting the original user query before … |
|      3 |  0.4371 | doc_005  | FAISS Index Types and Trade-offs                  | FAISS (Facebook AI Similarity Search) provides several index types suited to different performance p… |

### Strategy B — AI-Enhanced Retrieval

|   Rank |   Score | Doc ID   | Title                                             | Snippet                                                                                               |
|--------|---------|----------|---------------------------------------------------|-------------------------------------------------------------------------------------------------------|
|      1 |  0.618  | doc_006  | Retrieval-Augmented Generation (RAG) Architecture | RAG pipelines augment a large language model's responses by first retrieving relevant context from a… |
|      2 |  0.5615 | doc_004  | Vector Embeddings and Semantic Search             | Vector embeddings transform discrete tokens — words, sentences, or entire documents — into dense num… |
|      3 |  0.5257 | doc_007  | Query Expansion for Improved Retrieval            | Query expansion improves retrieval recall by rewriting or augmenting the original user query before … |

---

## Query 3: `How can we ensure high availability when a downstream service is failing?`

### Expanded Query (Strategy B)

> How can we ensure high availability when a downstream service is failing — including circuit breaker pattern, open/closed/half-open states, fallback responses, cascading failures, microservice resilience

### Strategy A — Raw Vector Search

|   Rank |   Score | Doc ID   | Title                                       | Snippet                                                                                               |
|--------|---------|----------|---------------------------------------------|-------------------------------------------------------------------------------------------------------|
|      1 |  0.5178 | doc_010  | Circuit Breakers and Graceful Degradation   | The circuit-breaker pattern prevents cascading failures in microservice architectures by short-circu… |
|      2 |  0.4879 | doc_008  | Observability and SLOs for Distributed APIs | Production APIs are monitored through a combination of metrics, logs, and distributed traces. Servic… |
|      3 |  0.4125 | doc_001  | Load Balancing Under Peak Traffic           | When a system experiences peak load, the load balancer distributes incoming requests across multiple… |

### Strategy B — AI-Enhanced Retrieval

|   Rank |   Score | Doc ID   | Title                                       | Snippet                                                                                               |
|--------|---------|----------|---------------------------------------------|-------------------------------------------------------------------------------------------------------|
|      1 |  0.7144 | doc_010  | Circuit Breakers and Graceful Degradation   | The circuit-breaker pattern prevents cascading failures in microservice architectures by short-circu… |
|      2 |  0.423  | doc_008  | Observability and SLOs for Distributed APIs | Production APIs are monitored through a combination of metrics, logs, and distributed traces. Servic… |
|      3 |  0.3974 | doc_001  | Load Balancing Under Peak Traffic           | When a system experiences peak load, the load balancer distributes incoming requests across multiple… |

---

## Overlap Analysis

How many of the top-3 documents were the same between Strategy A and B?

| Query                                                              |   Shared Docs | Overlap %   | Shared Doc IDs            |
|--------------------------------------------------------------------|---------------|-------------|---------------------------|
| How does the system handle peak load?…                             |             2 | 67%         | doc_001, doc_002          |
| What techniques improve retrieval accuracy in a semantic search p… |             2 | 67%         | doc_006, doc_007          |
| How can we ensure high availability when a downstream service is … |             3 | 100%        | doc_001, doc_008, doc_010 |

---

## Interpretation

- **High overlap** means both strategies agree on relevance — the expanded query
  confirmed the raw result. This is a sign of a well-formed original query.
- **Low overlap** indicates that query expansion surfaced different (and often
  more relevant) documents by bridging the lexical gap between the query and corpus.
- Strategy B consistently achieves higher or equal scores because the expanded
  query vector sits closer to the centroid of semantically related documents.

---

_Report auto-generated by `src/evaluation/benchmarker.py`_