from pydantic_settings import BaseSettings, SettingsConfigDict

class SystemSettings(BaseSettings):
    """
    Validates and stores service execution configurations.
    Loads environment variables from .env file automatically.
    """
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    # Core Application Configuration
    APP_NAME: str = "graphgpt-memory-service"
    APP_ENV: str = "production"
    DEBUG: bool = False
    HTTP_HOST: str = "0.0.0.0"
    HTTP_PORT: int = 8000

    # Kafka Broker Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_GROUP_ID: str = "memory-service-consumers"
    KAFKA_SESSION_TIMEOUT_MS: int = 30000
    KAFKA_MAX_POLL_INTERVAL_MS: int = 300000
    
    # Internal Communication Topics
    KAFKA_SUMMARY_TOPIC: str = "memory.summary.request"
    KAFKA_FACT_TOPIC: str = "memory.fact.request"
    KAFKA_EMBEDDING_TOPIC: str = "memory.embedding.request"
    KAFKA_DELETE_TOPIC: str = "memory.delete.request"
    KAFKA_DLQ_TOPIC: str = "memory.dlq"

    # Redis Cache Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    SNAPSHOT_TTL_SECONDS: int = 2592000
    SHORT_TERM_MESSAGE_LIMIT: int = 20
    IDEMPOTENCY_TTL_SECONDS: int = 604800

    # Milvus Vector Database Configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    VECTOR_DIMENSION: int = 1536

    # Embedding Model Tracking Configs
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    EMBEDDING_MODEL_VERSION: str = "v1.0.0"

    # gRPC & Microservice Communication Configurations
    LLM_SERVICE_HOST: str = "localhost"
    LLM_SERVICE_PORT: int = 50051
    GRAPH_SERVICE_URL: str = "http://localhost:8001"
    GRPC_TIMEOUT_SECONDS: float = 5.0
    LLM_CONCURRENT_LIMIT: int = 50

    # Memory Search Scoring Weights (Must sum to 1.0)
    RETRIEVAL_WEIGHT_SIMILARITY: float = 0.5
    RETRIEVAL_WEIGHT_RECENCY: float = 0.2
    RETRIEVAL_WEIGHT_IMPORTANCE: float = 0.3
    RETRIEVAL_DECAY_RATE: float = 0.05
    RETRIEVAL_TOP_K_FACTS: int = 10

    # Deduplication Similarity Threshold
    FACT_MERGE_SIMILARITY_THRESHOLD: float = 0.85

# Singleton instantiation
settings = SystemSettings()
