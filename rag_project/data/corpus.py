"""
corpus.py
---------
The canonical text corpus for the RAG pipeline.

Contains 10 technical paragraphs covering distributed systems, load balancing,
caching, autoscaling, vector databases, and related infrastructure topics.
These are used for ingestion into the FAISS vector store and form the
retrieval target for all benchmarking queries.
"""

CORPUS: list[dict] = [
    {
        "id": "doc_001",
        "title": "Load Balancing Under Peak Traffic",
        "text": (
            "When a system experiences peak load, the load balancer distributes incoming "
            "requests across multiple server instances to prevent any single node from "
            "becoming a bottleneck. Modern load balancers use algorithms such as round-robin, "
            "least-connections, and weighted routing to optimise throughput. Health checks "
            "run continuously so that unhealthy instances are removed from the rotation "
            "immediately, ensuring traffic is only routed to nodes that can serve requests "
            "reliably."
        ),
    },
    {
        "id": "doc_002",
        "title": "Horizontal Autoscaling and Elasticity",
        "text": (
            "Autoscaling is the mechanism by which a cloud-hosted service automatically "
            "provisions or terminates compute instances in response to real-time demand signals. "
            "Horizontal Pod Autoscaler (HPA) in Kubernetes, for example, watches CPU utilisation "
            "and custom metrics, then adjusts the replica count accordingly. Scale-out events "
            "typically trigger when sustained CPU usage exceeds 70% for a configurable window, "
            "while scale-in policies include a stabilisation window to avoid flapping under "
            "momentary load spikes."
        ),
    },
    {
        "id": "doc_003",
        "title": "Caching Strategies for High-Throughput Systems",
        "text": (
            "Distributed caches like Redis and Memcached reduce database pressure during "
            "traffic surges by serving frequently requested data from memory. A well-designed "
            "cache layer employs TTL-based expiry, cache-aside patterns, and write-through or "
            "write-behind strategies depending on consistency requirements. During peak load, "
            "cache hit rates become critically important; a drop in hit rate can cause a "
            "thundering herd effect as requests simultaneously miss and hammer the database."
        ),
    },
    {
        "id": "doc_004",
        "title": "Vector Embeddings and Semantic Search",
        "text": (
            "Vector embeddings transform discrete tokens — words, sentences, or entire documents "
            "— into dense numerical representations in a high-dimensional space. Semantically "
            "similar items cluster near each other, enabling approximate nearest-neighbour (ANN) "
            "search to find contextually relevant results rather than relying on exact keyword "
            "matches. Models such as OpenAI's text-embedding-ada-002 and Google's "
            "textembedding-gecko produce embeddings of 768–1536 dimensions that capture nuanced "
            "semantic relationships."
        ),
    },
    {
        "id": "doc_005",
        "title": "FAISS Index Types and Trade-offs",
        "text": (
            "FAISS (Facebook AI Similarity Search) provides several index types suited to "
            "different performance profiles. IndexFlatL2 and IndexFlatIP offer exact search "
            "with O(n) query time, suitable for corpora of up to ~1 million vectors. "
            "IndexIVFFlat partitions the vector space into Voronoi cells, reducing query time "
            "at the cost of recall. IndexHNSWFlat builds a hierarchical navigable small-world "
            "graph, providing sub-linear query time with high recall. Production deployments "
            "typically use HNSW or IVF-PQ for their balance of speed and accuracy."
        ),
    },
    {
        "id": "doc_006",
        "title": "Retrieval-Augmented Generation (RAG) Architecture",
        "text": (
            "RAG pipelines augment a large language model's responses by first retrieving "
            "relevant context from an external knowledge store, then injecting that context "
            "into the prompt. The retrieval step is critical: poor retrieval leads to "
            "irrelevant or hallucinated answers even when the LLM itself is capable. "
            "Chunking strategy, embedding model choice, and the similarity threshold for "
            "retrieved documents all interact to determine end-to-end answer quality. "
            "Hybrid search — combining dense vector retrieval with sparse BM25 — often "
            "outperforms either approach alone."
        ),
    },
    {
        "id": "doc_007",
        "title": "Query Expansion for Improved Retrieval",
        "text": (
            "Query expansion improves retrieval recall by rewriting or augmenting the original "
            "user query before embedding it. Techniques include pseudo-relevance feedback "
            "(PRF), where the top-k initial results are used to extract expansion terms, and "
            "generative expansion, where an LLM generates a hypothetical ideal document "
            "that is then embedded. Generative query expansion has shown particular promise "
            "in RAG systems because it bridges the lexical gap between how users phrase "
            "questions and how answers are phrased in the corpus."
        ),
    },
    {
        "id": "doc_008",
        "title": "Observability and SLOs for Distributed APIs",
        "text": (
            "Production APIs are monitored through a combination of metrics, logs, and "
            "distributed traces. Service-level objectives (SLOs) define acceptable bounds "
            "for latency (e.g., p99 < 500 ms) and error rate (< 0.1%). Alerting fires when "
            "the error budget is being consumed faster than allowed. During incident response, "
            "correlation between increased latency and upstream dependency health — such as "
            "a degraded database replica or a saturated message queue — helps engineers "
            "isolate root cause quickly."
        ),
    },
    {
        "id": "doc_009",
        "title": "Database Sharding and Connection Pooling",
        "text": (
            "Under heavy read/write load, a single database instance becomes a contention "
            "point. Horizontal sharding distributes data across multiple instances by a "
            "shard key, allowing queries that filter on the shard key to be served by a "
            "single node without fan-out. Connection pooling (via PgBouncer for PostgreSQL, "
            "for example) multiplexes thousands of application connections onto a small "
            "number of real database connections, dramatically reducing the overhead of "
            "connection establishment during traffic spikes."
        ),
    },
    {
        "id": "doc_010",
        "title": "Circuit Breakers and Graceful Degradation",
        "text": (
            "The circuit-breaker pattern prevents cascading failures in microservice "
            "architectures by short-circuiting calls to a struggling dependency after a "
            "configurable error threshold is breached. In the open state, the caller "
            "immediately returns a fallback response — cached data, a default value, or "
            "an informative error — rather than waiting for a timeout. This preserves "
            "system availability during partial outages and reduces the recovery time of "
            "the downstream service by eliminating the extra load from futile retries."
        ),
    },
]
