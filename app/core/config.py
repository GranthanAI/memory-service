from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SystemSettings(BaseSettings):
    """
    Validates and stores all service execution configurations.
    Loads environment variables from .env file automatically.
    Settings are grouped by concern and match the HLD v4.1 / LLD v3.1 architecture.
    """

    model_config = SettingsConfigDict(hide_input_in_errors=True, 
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "graphgpt-memory-service"
    APP_ENV: str = "production"
    ENVIRONMENT: str = "development"   # "production" turns on the fail-fast secret checks below
    DEBUG: bool = False
    HTTP_HOST: str = "0.0.0.0"
    HTTP_PORT: int = 8000

    # ─── Kafka ────────────────────────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_GROUP_ID: str = "memory-service-consumers"
    KAFKA_SESSION_TIMEOUT_MS: int = 30000
    KAFKA_MAX_POLL_INTERVAL_MS: int = 300000

    # Internal async task topics
    KAFKA_SUMMARY_TOPIC: str = "memory.summary.request"
    KAFKA_FACT_TOPIC: str = "memory.fact.request"
    KAFKA_EMBEDDING_TOPIC: str = "memory.embedding.request"
    KAFKA_DELETE_TOPIC: str = "memory.delete.request"
    KAFKA_DLQ_TOPIC: str = "memory.dlq"

    # Inbound ingestion topic: published by llm-service after every response
    # (KafkaPublisher.publish_memory_update). Feeds EventDispatcher to append
    # the user/assistant turn to conversation_recent_messages and advance the
    # summary/fact/embedding pipeline.
    KAFKA_MEMORY_UPDATE_TOPIC: str = "memory.update.requested"

    # Number of ingested messages (user + assistant turns counted individually)
    # per conversation after which EventDispatcher marks the snapshot
    # SUMMARY_PENDING and stages a memory.summary.request job. Previously
    # hardcoded to 20 inside EventDispatcher's constructor default; exposed
    # here so it can be tuned per-environment (e.g. lowered for faster
    # end-to-end verification) without editing code.
    SUMMARY_THRESHOLD: int = 20

    # ─── Cassandra (Primary Source of Truth) ─────────────────────────────────
    # Comma-separated for multi-node clusters: "10.0.0.1,10.0.0.2,10.0.0.3"
    CASSANDRA_HOSTS: str = "localhost"
    CASSANDRA_PORT: int = 9042
    CASSANDRA_KEYSPACE: str = "graphgpt_memory"
    CASSANDRA_TIMEOUT_SECONDS: float = 5.0

    # ─── Redis (Hot Cache Only) ───────────────────────────────────────────────
    # Redis stores nothing that cannot be rebuilt from Cassandra.
    REDIS_URL: str = "redis://localhost:6379/0"
    SNAPSHOT_TTL_SECONDS: int = 2592000        # 30 days
    SHORT_TERM_MESSAGE_LIMIT: int = 20         # Max recent messages in Redis list
    IDEMPOTENCY_TTL_SECONDS: int = 604800      # 7 days — covers Kafka replay window
    REDIS_LOCK_TTL_SECONDS: int = 5            # Initial lock TTL (extended by watchdog)
    REDIS_LOCK_WATCHDOG_INTERVAL: float = 2.0  # Watchdog heartbeat interval (seconds)

    # ─── Milvus (Vector Index) ────────────────────────────────────────────────
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    VECTOR_DIMENSION: int = 384
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_VERSION: str = "v1.0.0"
    MILVUS_BULK_INSERT_BATCH_SIZE: int = 100   # Vectors per Milvus insert batch
    EMBEDDING_CLIENT_TYPE: str = "grpc"         # "grpc" | "mock"

    # ─── gRPC — LLM Service Pool ─────────────────────────────────────────────
    LLM_SERVICE_HOST: str = "localhost"
    LLM_SERVICE_PORT: int = 50051
    GRPC_POOL_SIZE: int = 5                              # Number of persistent gRPC channels
    GRPC_TIMEOUT_SECONDS: float = 5.0                   # Per-call timeout
    GRPC_HEALTH_CHECK_INTERVAL_SECONDS: float = 30.0    # Dead channel replacement interval

    # ─── Graph Service ────────────────────────────────────────────────────────
    GRAPH_SERVICE_URL: str = "http://localhost:8001"
    GRAPH_SERVICE_TIMEOUT_MS: int = 200   # Graceful fallback if Graph Service is slow

    # ─── Outbox Worker ────────────────────────────────────────────────────────
    # Poll interval is configurable to tune throughput vs. latency trade-off.
    OUTBOX_POLL_INTERVAL_MS: int = 1000          # How often to poll outbox_jobs table (ms)
    OUTBOX_BATCH_SIZE: int = 50                  # Max jobs claimed per poll cycle
    OUTBOX_STALE_PROCESSING_MINUTES: int = 5     # Threshold for reaper to reclaim stuck jobs

    # ─── Circuit Breaker ──────────────────────────────────────────────────────
    CB_FAILURE_THRESHOLD: int = 5                    # Failures before OPEN state
    CB_RECOVERY_TIMEOUT_SECONDS: float = 60.0        # Time in OPEN before HALF_OPEN probe
    CB_HALF_OPEN_LIMIT: int = 2                      # Max probes allowed in HALF_OPEN state

    # ─── Retrieval Scoring ────────────────────────────────────────────────────
    # Final score = (similarity × W_SIM) + (recency × W_REC) + (importance × W_IMP)
    # Weights must sum to 1.0.
    # ─── Service-to-Service Security ──────────────────────────────────────────
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY: str = "graphgpt-memory-secret"
    JWT_SECRET_KEY: str = "graphgpt-jwt-secret"
    JWT_ALGORITHM: str = "HS256"

    # ─── End-user Authentication (public /memory routes) ──────────────────────
    # User access tokens are issued by auth-service: this MUST equal auth-service's
    # JWT_SECRET_KEY (it is NOT the service-to-service JWT_SECRET_KEY above). Empty
    # disables the public routes (every call is rejected).
    USER_JWT_SECRET_KEY: str = ""
    USER_JWT_ISSUER: str = "https://auth.granthan.com"

    # ─── Startup Validation ───────────────────────────────────────────────────
    STRICT_STARTUP_VALIDATION: bool = False

    RETRIEVAL_WEIGHT_SIMILARITY: float = 0.5
    RETRIEVAL_WEIGHT_RECENCY: float = 0.2
    RETRIEVAL_WEIGHT_IMPORTANCE: float = 0.3
    RETRIEVAL_DECAY_RATE: float = 0.05         # Exponential decay rate for recency scoring
    RETRIEVAL_MIN_SIMILARITY: float = 0.3      # Topical facts below this cosine similarity are not injected
    RETRIEVAL_FACT_CHAR_BUDGET: int = 1500     # Max characters of memory facts injected into a prompt
    RETRIEVAL_TOP_K_FACTS: int = 10            # Max facts to retrieve from Milvus per query
    FACT_MERGE_SIMILARITY_THRESHOLD: float = 0.85   # Above this → supersede, below → insert new

    # ─── Internal LLM Engine ──────────────────────────────────────────────────
    LLM_PROVIDER: str = "groq"
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "openai/gpt-oss-20b"
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 3
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024

    # ─── Automatic conversation-memory compaction (no user-facing setting) ────────────────────────────────────────
    # A summary is requested when ANY of: SUMMARY_THRESHOLD messages arrived since the last one (the original rule), or
    # at least COMPACTION_MIN_NEW_MESSAGES new messages carry COMPACTION_TRIGGER_TOKENS tokens (long messages build up
    # context pressure much faster than 20 short ones). One request per conversation is in flight at a time.
    COMPACTION_TRIGGER_TOKENS: int = 6000
    COMPACTION_MIN_NEW_MESSAGES: int = 6
    COMPACTION_INFLIGHT_TTL_SECONDS: int = 120
    COMPACTION_PENDING_TTL_SECONDS: int = 604800
    # A summary is validated in TOKENS with the real tokenizer. The default equals the previous fixed limit
    # (6000 characters ~ 1500 tokens), so nothing changes unless an operator changes it.
    SUMMARY_MAX_TOKENS: int = 1500
    SUMMARY_MAX_REFINEMENTS: int = 1                    # extra, shorter attempts when a summary is over the limit
    SUMMARY_CONSOLIDATE_RATIO: float = 0.85             # at/above this share of the limit the summary is consolidated
    SUMMARY_CONSOLIDATE_TARGET_RATIO: float = 0.65
    SUMMARY_PROMPT_VERSION: str = "2"
    # Context assembly: what stays verbatim next to a summary, and how far back a stale summary is bridged.
    CONTEXT_MIN_VERBATIM_MESSAGES: int = 6
    CONTEXT_MAX_UNSUMMARIZED_MESSAGES: int = 120
    CONTEXT_STALE_MESSAGES: int = 40                    # uncovered messages beyond this = the summary is stale
    # Question-aware retrieval of older ORIGINAL messages of the same conversation (lexical, no new index).
    HISTORY_SEARCH_ENABLED: bool = False                # OFF until llm-service renders retrieved_history in the prompt
    HISTORY_SEARCH_MAX_MESSAGES: int = 400
    HISTORY_SEARCH_TOP_K: int = 3
    HISTORY_SEARCH_MIN_COVERAGE: float = 0.5            # share of the question's distinctive terms a hit must contain
    HISTORY_SEARCH_RECALL_MIN_COVERAGE: float = 0.25    # ... when the question explicitly asks about earlier content
    HISTORY_RESULT_MAX_TOKENS: int = 350                # per excerpt
    HISTORY_TOTAL_MAX_TOKENS: int = 1000                # all excerpts together
    # Combined gRPC server port: serves both the internal LLM Service
    # (summarization/fact-extraction) and the Memory Context Service
    # (GetMemoryContext/GetShortTermMemory, consumed by llm-service's
    # ContextCollector — default matches llm-service's memory_service_port)
    MEMORY_CONTEXT_GRPC_PORT: int = 50051
    LLM_POOL_MAX_CONNECTIONS: int = 50
    LLM_POOL_MAX_KEEPALIVE_CONNECTIONS: int = 10
    LLM_MAX_CONCURRENT_REQUESTS: int = 10


# Singleton — import this everywhere

    @model_validator(mode="after")
    def _validate_compaction(self):
        if self.COMPACTION_TRIGGER_TOKENS < 500 or self.COMPACTION_MIN_NEW_MESSAGES < 1:
            raise ValueError("COMPACTION_TRIGGER_TOKENS must be >= 500 and COMPACTION_MIN_NEW_MESSAGES >= 1")
        if self.SUMMARY_MAX_TOKENS < 200 or self.SUMMARY_MAX_REFINEMENTS < 0:
            raise ValueError("SUMMARY_MAX_TOKENS must be >= 200 and SUMMARY_MAX_REFINEMENTS >= 0")
        if not 0.0 < self.SUMMARY_CONSOLIDATE_TARGET_RATIO < self.SUMMARY_CONSOLIDATE_RATIO <= 1.0:
            raise ValueError("SUMMARY_CONSOLIDATE_* ratios must satisfy 0 < target < ratio <= 1")
        if self.CONTEXT_MIN_VERBATIM_MESSAGES < 1 or self.CONTEXT_MAX_UNSUMMARIZED_MESSAGES < self.SHORT_TERM_MESSAGE_LIMIT:
            raise ValueError("CONTEXT_MIN_VERBATIM_MESSAGES must be >= 1 and CONTEXT_MAX_UNSUMMARIZED_MESSAGES >= SHORT_TERM_MESSAGE_LIMIT")
        if not 0.0 < self.HISTORY_SEARCH_RECALL_MIN_COVERAGE <= self.HISTORY_SEARCH_MIN_COVERAGE <= 1.0:
            raise ValueError("HISTORY_SEARCH coverage thresholds must satisfy 0 < recall <= default <= 1")
        return self

    @model_validator(mode="after")
    def _fail_fast_on_unsafe_production_config(self):
        """In production the service refuses to start with a development default or a weak secret (a known default
        would let anyone forge tokens). Development and test environments are unaffected. Secret VALUES are never
        included in the error message."""
        if self.ENVIRONMENT.strip().lower() not in ("production", "prod"):
            return self
        dev_values = {"supersecretjwtkeyforauthservicelocaldvelopment12345", "graphgpt-jwt-secret", "graphgpt-memory-secret", "changeme", "secret"}
        problems: list[str] = []

        def secret(name: str, value: str) -> None:
            if not value or value in dev_values or len(value) < 32:
                problems.append(f"{name} must be set to a unique secret of at least 32 characters")
        secret("API_KEY", self.API_KEY)
        secret("JWT_SECRET_KEY", self.JWT_SECRET_KEY)
        secret("USER_JWT_SECRET_KEY", self.USER_JWT_SECRET_KEY)
        if self.DEBUG:
            problems.append("DEBUG must be false")
        if problems:
            raise ValueError("Unsafe production configuration: " + "; ".join(problems))
        return self


settings = SystemSettings()
