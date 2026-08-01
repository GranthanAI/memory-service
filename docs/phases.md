# GraphGPT Memory Service - 20-Phase Implementation Plan
**Detailed Step-by-Step Coding Roadmap**  

---

## Roadmap Overview

This implementation plan outlines the 20 coding phases for building the **GraphGPT Memory Service** from setup to production deployment.

```text
       ┌────────────────────────────────────────────────────────┐
       │     Phases 1-5: Foundations, Core Config & Schemas     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │  Phases 6-10: Persistence, Repositories & Lineage DBs  │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │   Phases 11-15: gRPC AI Clients, Pipelines & Ingestion  │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     Phases 16-20: Workers, REST APIs, Observability    │
       └────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Environment & Project Setup
* **Objective:** Establish the development environment, virtual runtime settings, and dependency locks.
* **Tasks:**
  * Configure `pyproject.toml` with all service dependencies:
    * **Framework:** `fastapi`, `uvicorn`
    * **Kafka:** `aiokafka`
    * **Cache:** `redis[asyncio]`
    * **Vector DB:** `pymilvus`
    * **Persistent DB:** `cassandra-driver` *(new — Cassandra is the primary source of truth)*
    * **AI/gRPC:** `grpcio`, `grpcio-tools`
    * **Config:** `pydantic-settings`
    * **Observability:** `prometheus-client`
    * **Testing:** `pytest`, `pytest-asyncio`
    * **Formatting:** `black`, `isort`, `flake8`
  * Create `requirements.txt` from the lock file.
  * Initialize `.env` and `.env.example` with all configuration variables:
    * Kafka: bootstrap servers, group ID, session timeout, all topic names (summary, fact, embedding, delete, DLQ)
    * Redis: URL, snapshot TTL, idempotency TTL, lock TTL, watchdog interval
    * Cassandra: hosts, port, keyspace, timeout
    * Milvus: host, port, vector dimension, embedding model name and version, bulk insert batch size
    * gRPC: LLM host, port, pool size, timeout, health check interval
    * Graph Service: URL, timeout MS
    * Outbox: poll interval MS, batch size, stale processing timeout minutes
    * Circuit Breaker: failure threshold, recovery timeout, half-open limit
    * Retrieval scoring weights, decay rate, top-k, fact merge threshold
  * Create `.gitignore` to exclude `.env`, `__pycache__`, `.venv`, and `uv.lock`.
* **Verification:** Run `uv sync` and `pip install -r requirements.txt`. Verify the virtual environment boots and `python -c "import fastapi, aiokafka, cassandra, pymilvus, redis"` succeeds.

---

### Phase 2: Configuration & Logging Core (`app/core/`)
* **Objective:** Code the centralized system configuration loading and logging infrastructure for all downstream modules.
* **Tasks:**
  * Implement [config.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/core/config.py) using `pydantic-settings` `BaseSettings` (V2). Include all settings groups:
    * **Kafka** (bootstrap servers, group ID, session timeout, all topic names)
    * **Redis** (URL, TTLs, lock TTL, watchdog interval)
    * **Cassandra** (hosts, port, keyspace, timeout) *(new)*
    * **Milvus** (host, port, dimension, model name/version, bulk insert batch size)
    * **gRPC** (LLM host/port, pool size, timeout, health check interval)
    * **Graph Service** (URL, timeout MS) *(new — for graceful fallback)*
    * **Outbox** (poll interval MS, batch size, stale processing minutes) *(new)*
    * **Circuit Breaker** (failure threshold, recovery timeout, half-open limit) *(new)*
    * **Retrieval** (scoring weights, decay rate, top-k, fact merge threshold)
  * Write [logging.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/core/logging.py) with:
    * Async-safe `ContextVar` bindings for `trace_id`, `conversation_id`, `event_id`, `summary_version`.
    * `JSONFormatter` for production (structured JSON logs).
    * `ConsoleFormatter` for development (human-readable prefixed logs).
    * `setup_logging(debug_mode)` that switches formatters based on `settings.DEBUG`.
  * Code [exceptions.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/core/exceptions.py) with:
    * `MemoryServiceException` — base class.
    * `CircuitBreakerOpenException(service_name)` — raised when the circuit breaker is OPEN.
    * `DeduplicationException(event_id)` — raised when an event_id is already in `processed_events`.
    * `JobExecutionException(job_type, job_id, message)` — raised when a worker job fails irrecoverably.
* **Verification:** Run `uv run pytest tests/unit/test_core.py` — assert config defaults load cleanly, JSON log output includes trace context variables, and exceptions carry the correct metadata fields.

---

### Phase 3: Database Connection Adapters (`app/db/`)
* **Objective:** Initialize connection pools for Cassandra (primary store), Redis (hot cache), and Milvus (vector index). All three must be healthy before the service enters readiness.
* **Tasks:**
  * Code [db/cassandra.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/db/cassandra.py) *(new — primary source of truth)*:
    * Initialize a `cassandra.cluster.Cluster` with configurable `CASSANDRA_HOSTS` and `CASSANDRA_PORT`.
    * Create a `Session` bound to `CASSANDRA_KEYSPACE`.
    * Expose `get_session() -> Session` for repository injection.
    * On first connect, execute CQL `CREATE KEYSPACE IF NOT EXISTS` and all 8 table schemas (see LLD §3):
      * `conversation_snapshots`
      * `conversation_summaries`
      * `processed_events` (7-day TTL)
      * `outbox_jobs` (partitioned by `status`)
      * `outbox_processing_index` (for stale PROCESSING reaping without ALLOW FILTERING)
      * `retry_jobs` (partitioned by `status`)
      * `user_facts` (partitioned by `user_id, category`)
      * `conversation_recent_messages` (durable short-term window backup)
  * Code [db/redis.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/db/redis.py):
    * Initialize a `redis.asyncio.ConnectionPool` from `REDIS_URL`.
    * Expose `get_redis_client() -> aioredis.Redis` using the pool.
    * Implement `init_redis_pool()` and `close_redis_pool()` lifecycle functions.
  * Code [db/milvus.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/db/milvus.py):
    * Wrap `pymilvus.connections.connect(host, port)` and `disconnect()`.
    * Expose `check_milvus_ready()` via `utility.list_collections()` ping.
  * Implement [db/session.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/db/session.py):
    * `initialize_db_sessions()` — starts all three pools in order: Cassandra → Redis → Milvus.
    * Verifies each with a health check (Cassandra `SELECT now() FROM system.local`, Redis `PING`, Milvus `list_collections`).
    * Raises `RuntimeError` with the identity of the failing service if any check fails.
    * `close_db_sessions()` — gracefully disconnects all three pools.
* **Verification:** Run `uv run pytest tests/unit/test_db.py` — mock all three drivers and assert that:
  * `initialize_db_sessions()` calls `connect`, `ping`, and `list_collections` in the correct order.
  * A Cassandra connection failure raises `RuntimeError` and triggers cleanup.
  * A Redis ping failure raises `RuntimeError` and triggers cleanup.
  * `close_db_sessions()` disconnects all three without errors.

---

### Phase 4: Shared Utilities (`app/utils/`)
* **Objective:** Write common helper functions for compression, serialization, and locks.
* **Tasks:**
  * Implement [compression.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/utils/compression.py) with zlib compression/decompression methods for summary texts.
  * Implement [locks.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/utils/locks.py) to provide distributed Redis lock capabilities via `set(nx=True, ex=ttl)`.
  * Write serialization and timing helpers in `serialization.py` and `timers.py`.
* **Verification:** Run unit tests in `tests/unit/test_utils.py` to verify compression ratios and locks.

---

### Phase 5: Domain Models & Pydantic Schemas (`app/models/` & `app/schemas/`)
* **Objective:** Define data validation schemas and request/response DTOs.
* **Tasks:**
  * Code schemas for events in `schemas/events.py` (`MessageCreatedPayload`, `ResponseCompletedPayload`).
  * Implement domain models in `models/snapshot.py`, `models/summary.py`, and `models/memory.py` to represent internal service objects.
  * Write request and response structures in `schemas/requests.py` and `schemas/responses.py` for REST context endpoints.
* **Verification:** Run validation unit tests to verify that invalid payloads raise expected validation errors.

---

### Phase 6: Event Idempotency Service (`app/services/idempotency_service.py`)
* **Objective:** Set up idempotency checking for incoming messages.
* **Tasks:**
  * Implement [idempotency_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/idempotency_service.py) to track processed event keys.
  * Implement `processed_event_repository.py` to manage `event_idempotency:{event_id}` Redis keys with a 7-day TTL.
* **Verification:** Run integration tests to confirm that duplicate event keys are marked as processed and ignored.

---

### Phase 7: Redis Repository Layer (`app/repositories/redis_repository.py`)
* **Objective:** Build the persistence layer for Redis data.
* **Tasks:**
  * Implement [redis_repository.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/repositories/redis_repository.py) to query and write Hash and List keys.
  * Write transaction pipelines (`pipeline(transaction=True)`) to bundle operations for snapshot state and recent message lists.
* **Verification:** Run integration tests asserting correct writes and pipeline rollbacks for Redis operations.

---

### Phase 8: Snapshot Builder Service (`app/services/snapshot_service.py`)
* **Objective:** Implement snapshot updating and sliding window management.
* **Tasks:**
  * Implement [snapshot_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/snapshot_service.py).
  * Code `append_message_atomic` to append messages to the conversation list and check if the summarization limit is reached.
  * Configure sliding window limits to keep the recent message cache capped at double the limit.
* **Verification:** Verify that append operations update conversation counts and states correctly in Redis.

---

### Phase 9: Graph Lineage Resilient Client (`app/clients/graph_client.py`)
* **Objective:** Build the resilient client for Graph Service lineage checks.
* **Tasks:**
  * Implement [graph_client.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/clients/graph_client.py) using `httpx.AsyncClient` with timeouts.
  * Code a custom `CircuitBreaker` pattern to monitor Graph Service connections.
  * Code a retry loop with exponential backoff and a fallback mechanism that returns only the current conversation ID if the client fails.
* **Verification:** Mock connection timeouts and verify that the circuit breaker trips and returns the fallback lineage list.

---

### Phase 10: Milvus Repository Layer (`app/repositories/milvus_repository.py`)
* **Objective:** Set up vector search collections in Milvus.
* **Tasks:**
  * Implement [milvus_repository.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/repositories/milvus_repository.py).
  * Build collection schema templates for `user_memory_vectors` and `semantic_memory_vectors` using HNSW vector indexing.
  * Code insertion and deletion queries using dynamic partition keys (`user_id`).
* **Verification:** Verify that vectors and their metadata fields index successfully and can be queried.

---

### Phase 11: LLM Client gRPC Implementation (`app/clients/llm_client.py`)
* **Objective:** Implement gRPC communication with the LLM Service.
* **Tasks:**
  * Code [llm_client.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/clients/llm_client.py) using `grpc.aio` insecurity channels.
  * Bind generated protobuf stubs for `SummarizeConversation`, `ExtractFacts`, and `GenerateEmbeddings`.
  * Add a concurrency rate limiter using `asyncio.Semaphore` capped at 50 requests.
* **Verification:** Run connection tests to confirm gRPC requests execute successfully and respects the concurrency limits.

---

### Phase 12: Memory State Machine (`app/services/memory_service.py`)
* **Objective:** Implement conversation workflow states.
* **Tasks:**
  * Code the `MemoryState` enum and the state transition logic in [services/memory_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/memory_service.py).
  * Add validation checks to ensure conversation updates follow allowed state transitions (`NEW` -> `ACTIVE` -> `SUMMARY_PENDING` -> `SUMMARIZED` -> `FACT_PENDING` -> `EMBEDDED` -> `READY`).
* **Verification:** Run state transition tests to verify that invalid state changes raise expected errors.

---

### Phase 13: Incremental Summarization Service (`app/services/summary_service.py`)
* **Objective:** Set up summarization processing for conversation message updates.
* **Tasks:**
  * Implement [summary_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/summary_service.py) to manage worker-side summarization.
  * Extract delta message slices using the last summary message ID watermark.
  * Ask the LLM service to summarize, update Redis summary keys with compressed strings, increment `summary_version`, and update the watermark.
* **Verification:** Mock LLM summarization responses and verify that summary states and message watermarks update correctly.

---

### Phase 14: User Fact Merging Logic (`app/services/long_memory_service.py`)
* **Objective:** Implement fact deduplication to prevent redundant entries.
* **Tasks:**
  * Implement deduplication logic in [long_memory_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/long_memory_service.py).
  * Query existing facts in Milvus for similarity overlaps (threshold check: `0.85`).
  * If a duplicate is found, update the existing entry (execute delete-then-insert update sequence). Otherwise, insert the statement as a new fact.
* **Verification:** Mock existing vectors and verify that matching statements trigger update queries instead of creating duplicates.

---

### Phase 15: Decoupled Scoring & Ranking Engine (`app/services/ranking_service.py`)
* **Objective:** Implement ranking calculations for memory retrieval.
* **Tasks:**
  * Implement [ranking_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/ranking_service.py).
  * Code scoring calculations using the configured retrieval weights:
    $$\text{Score} = w_{\text{vector}} S_{vector} + w_{\text{time}} S_{time} + w_{\text{importance}} S_{importance}$$
  * Implement time decay using the exponential decay formula ($e^{-\lambda t}$).
* **Verification:** Write unit tests to check that scoring matches expected mathematical weights.

---

### Phase 16: Structured Context Builder (`app/services/context_builder.py`)
* **Objective:** Coordinate retrieval queries across all memory layers.
* **Tasks:**
  * Implement [context_builder.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/context_builder.py).
  * Query short-term messages (Redis), ancestor summaries (Redis), long-term facts (Milvus), and semantic concepts (Milvus) concurrently.
  * Format and compile the final structured retrieval response object.
* **Verification:** Verify that context queries return clean JSON payloads containing metrics and content from all memory layers.

---

### Phase 17: Kafka Task Producer (`app/events/kafka_producer.py`)
* **Objective:** Set up task publishing for background worker pipelines.
* **Tasks:**
  * Implement [kafka_producer.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/events/kafka_producer.py) using `AIOKafkaProducer`.
  * Set up publishing to internal worker topics (`memory.summary.request`, `memory.fact.request`, `memory.embedding.request`, `memory.delete.request`).
  * Ensure messages route by partition key (`conversation_id`) to preserve ordering.
* **Verification:** Publish test payloads and verify that messages partition correctly.

---

### Phase 18: Kafka Consumer Group Ingestor (`app/events/kafka_consumer.py`)
* **Objective:** Configure event ingestion and task dispatching.
* **Tasks:**
  * Implement [kafka_consumer.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/events/kafka_consumer.py) to poll API events.
  * Implement [dispatcher.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/events/dispatcher.py) to check idempotency and route payloads to workers.
  * Code offset commit logic to implement at-least-once delivery guarantees.
* **Verification:** Verify that events poll successfully, check idempotency keys, and commit offsets in the ingestion loop.

---

### Phase 19: Background Worker Daemons (`app/workers/`)
* **Objective:** Build background workers to process task queues.
* **Tasks:**
  * Implement workers for summary tasks, fact extraction, and vector embedding generation.
  * Implement retry loops and Dead-Letter Queue (DLQ) publishing for failed jobs.
  * Implement the cleanup worker to run sweeps (sweeping expired keys, compacting collections).
* **Verification:** Run end-to-end integration tests to verify background workers process tasks, update states, and log traces correctly.

---

### Phase 20: REST Controllers, Prometheus Metrics & Readiness checks (`app/api/`)
* **Objective:** Set up API endpoints, observability metrics, and health probes.
* **Tasks:**
  * Code API routes in `api/internal/memory.py` to serve structured memory contexts.
  * Implement Prometheus middleware to track and expose metrics (lag metrics, processing latencies).
  * Implement the `/ready` endpoint to verify connections to Redis, Milvus, and the LLM gRPC channel.
* **Verification:** Run load tests to verify HTTP API query speeds, check `/ready` endpoint status, and verify metrics outputs.
