from pydantic_settings import BaseSettings, SettingsConfigDict


class SystemSettings(BaseSettings):
    """
    Validates and stores all service execution configurations.
    Loads environment variables from .env file automatically.
    Settings are grouped by concern and match the HLD v4.1 / LLD v3.1 architecture.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "graphgpt-memory-service"
    APP_ENV: str = "production"
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

    # ─── Startup Validation ───────────────────────────────────────────────────
    STRICT_STARTUP_VALIDATION: bool = False

    RETRIEVAL_WEIGHT_SIMILARITY: float = 0.5
    RETRIEVAL_WEIGHT_RECENCY: float = 0.2
    RETRIEVAL_WEIGHT_IMPORTANCE: float = 0.3
    RETRIEVAL_DECAY_RATE: float = 0.05         # Exponential decay rate for recency scoring
    RETRIEVAL_TOP_K_FACTS: int = 10            # Max facts to retrieve from Milvus per query
    FACT_MERGE_SIMILARITY_THRESHOLD: float = 0.85   # Above this → supersede, below → insert new


# Singleton — import this everywhere
settings = SystemSettings()
