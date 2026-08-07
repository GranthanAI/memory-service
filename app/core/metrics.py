"""
app/core/metrics.py

Central Prometheus metric registry for the Memory Service.
All counters, histograms, and gauges defined here as module-level singletons.

Metric naming convention: memory_<component>_<name>_<unit>
Matches the HLD §9 Metrics Surface table exactly.
"""

from prometheus_client import Counter, Gauge, Histogram

# ─── Cache Metrics ────────────────────────────────────────────────────────────

REDIS_HIT = Counter(
    "memory_redis_hit_total",
    "Total Redis cache hit count across all cache key types (snapshot, summary, messages)",
)

REDIS_MISS = Counter(
    "memory_redis_miss_total",
    "Total Redis cache miss count — triggers Cassandra read-through fallback",
)

LOCK_WAIT = Histogram(
    "memory_redis_lock_wait_seconds",
    "Time spent waiting to acquire a Redis distributed lock (UUID ownership token)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# ─── Vector Search Metrics ────────────────────────────────────────────────────

MILVUS_QPS = Counter(
    "memory_milvus_queries_total",
    "Total Milvus HNSW similarity search query count",
)

# ─── Context Assembly Metrics ─────────────────────────────────────────────────

CTX_BUILD = Histogram(
    "memory_context_build_seconds",
    "End-to-end context assembly latency (retrieval + ranking + serialization)",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ─── Queue Depth Gauges ───────────────────────────────────────────────────────

SUMMARY_Q = Gauge(
    "memory_summary_queue_size",
    "Number of outbox_jobs rows pending for summary worker consumption",
)

FACT_Q = Gauge(
    "memory_fact_queue_size",
    "Number of outbox_jobs rows pending for fact extraction worker consumption",
)

EMBEDDING_Q = Gauge(
    "memory_embedding_queue_size",
    "Number of outbox_jobs rows pending for embedding worker consumption",
)

DLQ_SIZE = Gauge(
    "memory_dlq_size",
    "Number of jobs currently in the Dead Letter Queue (retry_jobs with status=FAILED)",
)

OUTBOX_PEND = Gauge(
    "memory_outbox_pending_total",
    "Total pending outbox_jobs rows across all topics",
)

RETRY_PEND = Gauge(
    "memory_retry_pending_total",
    "Total pending rows in retry_jobs table awaiting next_retry timestamp",
)

# ─── gRPC Error Metrics ───────────────────────────────────────────────────────

GRPC_ERRORS = Counter(
    "memory_grpc_channel_errors_total",
    "Total gRPC channel-level errors (circuit breaker trips, TRANSIENT_FAILURE, timeouts)",
)

# ─── LLM Metrics ──────────────────────────────────────────────────────────────

LLM_REQUESTS = Counter(
    "memory_llm_requests_total",
    "Total LLM generation requests, labeled by provider, model, action, and status",
    ["provider", "model", "action", "status"],
)

LLM_LATENCY = Histogram(
    "memory_llm_latency_seconds",
    "LLM call completion latency",
    ["provider", "model", "action"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

LLM_TOKENS = Counter(
    "memory_llm_tokens_total",
    "Total tokens consumed by LLM operations, labeled by type (prompt, completion)",
    ["type"],
)
