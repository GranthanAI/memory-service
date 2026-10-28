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

# ─── Automatic conversation-memory compaction and context assembly (bounded labels only) ───────────────────────

COMPACTION_REQUESTS = Counter(
    "memory_compaction_requests_total",
    "Summary (compaction) requests scheduled, by what triggered them",
    ["reason"],      # count | pressure | stale_read | deduplicated
)
COMPACTION_RESULTS = Counter(
    "memory_compaction_results_total",
    "Outcome of one summary update",
    ["outcome"],     # committed | nothing_new | refined | consolidated | fallback_trimmed | failed | lost_race
                     # | anchor_lost (a pinned fact / open item vanished) | unsupported_entity (an invented name or number)
)
BRANCH_SUMMARY_RESULTS = Counter(
    "memory_branch_summary_total",
    "Outcome of ensuring a parent summary exists at a new branch point",
    ["outcome"],     # not_a_branch | present | requested | unavailable | skipped | error
)
IDLE_COMPACTION_RESULTS = Counter(
    "memory_idle_compaction_total",
    "Outcome of the idle-conversation sweep",
    ["outcome"],     # requested | nothing_new | busy | no_snapshot | error
)
CONVERSATION_DELETE_RESULTS = Counter(
    "memory_conversation_delete_total",
    "Outcome of purging a deleted conversation",
    ["outcome"],     # purged | skipped | ignored | error
)
MEMORY_UPDATE_RESULTS = Counter(
    "memory_update_ingest_results_total",
    "Outcome of ingesting one memory.update.requested event",
    ["outcome"],     # ingested | retry_scheduled | dead_lettered | rewound | answer_replaced | explicit_memory_queued
)
COMPACTION_SECONDS = Histogram(
    "memory_compaction_seconds",
    "Duration of one summary update (all model calls included)",
    buckets=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
)
COMPACTION_SOURCE_MESSAGES = Histogram(
    "memory_compaction_source_messages",
    "Messages folded into the summary by one update",
    buckets=(1, 5, 10, 20, 40, 80, 200),
)
SUMMARY_TOKENS = Histogram(
    "memory_summary_tokens",
    "Token count of a committed summary",
    buckets=(100, 250, 500, 750, 1000, 1500, 2000),
)
SUMMARY_STALE = Counter(
    "memory_summary_stale_total",
    "Context builds where many messages were not yet covered by the summary",
)
CONTEXT_COVERED_SKIPPED = Counter(
    "memory_context_covered_messages_skipped_total",
    "Window messages not re-sent because the summary already covers them",
)
CONTEXT_GAP_CLOSED = Counter(
    "memory_context_gap_closed_total",
    "Context builds that read messages back from durable storage because neither the summary nor the window had them",
)
HISTORY_SEARCH = Counter(
    "memory_history_search_total",
    "Searches of older original messages for the current question",
    ["outcome"],     # hit | miss | skipped | error
)
HISTORY_SEARCH_SECONDS = Histogram(
    "memory_history_search_seconds",
    "Duration of one older-message search",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
