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
* **Objective:** Define data validation schemas, domain dataclasses, and request/response DTOs.
* **Tasks:**
  * Implement domain models in `app/models/`:
    * `snapshot.py`: Lightweight `ConversationSnapshot` dataclass containing only metadata state (no `recent_messages` JSON blob).
    * `summary.py`: `SummaryRecord` representing versioned summary text and generation metadata.
    * `memory.py`: `MemoryState` enum and basic long-term fact types.
  * Code events schema in `app/schemas/events.py` for ingestion (`MessageCreatedPayload`, `ResponseCompletedPayload`).
  * Implement requests/responses in `app/schemas/requests.py` and `app/schemas/responses.py` for the REST context endpoint.
    * **CRITICAL**: The request schema must expose `query: str` instead of embedding vectors, hiding the embedding logic behind the service boundary.
* **Verification:** Run unit validation tests to verify that invalid inputs fail validation and that the request schema does not expose any vector fields.

---

### Phase 6: Event Idempotency Service (`app/services/idempotency_service.py`)
* **Objective:** Ensure exactly-once event processing by tracking handled events in the Cassandra source of truth.
* **Tasks:**
  * Implement `processed_event_repository.py` to write and check rows in the Cassandra `processed_events` table.
  * Implement [idempotency_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/idempotency_service.py) that queries the repository before initiating any state changes.
  * Configure Cassandra rows to use a 7-day TTL (`default_time_to_live = 604800`) to cover all realistic Kafka replay windows.
* **Verification:** Run integration tests verifying that trying to process an event with an existing `event_id` throws a `DeduplicationException`.

---

### Phase 7: Cassandra Repository Layer (`app/repositories/cassandra_repository.py`)
* **Objective:** Build the primary data access layer for Cassandra persistent state.
* **Tasks:**
  * Implement [cassandra_repository.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/repositories/cassandra_repository.py) with methods for:
    * Fetching and saving `conversation_snapshots` metadata.
    * Reading and writing versioned `conversation_summaries`.
    * Retrieving category-scoped `user_facts`.
    * Performing time-ordered queries on `conversation_recent_messages`.
    * Reading, writing, and updating `outbox_jobs` and `retry_jobs`.
* **Verification:** Run integration tests to confirm data is correctly persisted, fetched, and sorted in local Cassandra.

---

### Phase 8: Redis Repository Layer (`app/repositories/redis_repository.py`)
* **Objective:** Build the hot cache repository layer for sub-millisecond retrieval.
* **Tasks:**
  * Implement [redis_repository.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/repositories/redis_repository.py) to read and write:
    * Snapshot hot cache (using Redis Hash keys).
    * Compressed summary hot cache (using zstd).
    * Recent message sliding lists (using Redis Lists with `LPUSH` + `LTRIM` to enforce the N-message limit).
    * Invalidation methods that clear all cache keys (`snapshot:`, `recent:`, `summary:`) for a given conversation.
* **Verification:** Verify that Redis caches compress/decompress summaries and limit list sizes correctly.

---

### Phase 9: Snapshot Builder Service (`app/services/snapshot_service.py`)
* **Objective:** Implement atomic batch mutations and cache invalidations.
* **Tasks:**
  * Implement [snapshot_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/snapshot_service.py).
  * Code `commit_snapshot_and_outbox` to execute a **Cassandra Logged Batch** containing:
    * Snapshot metadata update.
    * New message append into `conversation_recent_messages`.
    * Ingestion idempotency row addition in `processed_events`.
    * Task dispatch row addition in `outbox_jobs` (status `PENDING`).
  * Implement post-commit hooks to delete related Redis cache keys to trigger read-through hydration on the next query.
* **Verification:** Verify that the logged batch executes atomically and cache invalidation deletes Redis keys.

---

### Phase 10: Milvus Repository Layer (`app/repositories/milvus_repository.py`)
* **Objective:** Set up vector schema creation and dynamic partitioning.
* **Tasks:**
  * Implement [milvus_repository.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/repositories/milvus_repository.py).
  * Build the collection schema for `user_memory_vectors` using HNSW vector indexing.
  * Configure `user_id` as the Dynamic Partition Key (`is_partition_key=True`) to route vectors cleanly at scale (100M users).
  * Code bulk insertion and deletion methods.
* **Verification:** Test vector search matching inside a user partition and verify distance metric returns cosine values.

---

### Phase 11: LLM Client gRPC Implementation (`app/clients/llm_client.py`)
* **Objective:** Build a resilient connection pool to invoke inference microservices.
* **Tasks:**
  * Implement `AsyncGRPCConnectionPool` to manage a pool of `grpc.aio` channels with round-robin routing.
  * Code a background loop to perform health checks and replace channels that enter `TRANSIENT_FAILURE`.
  * Wrap all outbound calls with a custom state-based **Circuit Breaker** (CLOSED, OPEN, HALF_OPEN) using thresholds configured in Phase 2.
* **Verification:** Mock channel connection failures and verify the pool replaces dead channels and the circuit breaker trips.

---

### Phase 12: Memory State Machine (`app/services/memory_service.py`)
* **Objective:** Coordinate workflow state transitions across pipeline phases.
* **Tasks:**
  * Implement the state transition rules in [services/memory_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/memory_service.py).
  * Manage transitions: `ACTIVE` -> `SUMMARY_PENDING` -> `SUMMARIZING` -> `FACT_PENDING` -> `EXTRACTING_FACTS` -> `EMBEDDING_PENDING` -> `READY`.
  * Ensure failure paths transition states to `FAILED` and record context in `retry_jobs`.
* **Verification:** Write unit tests asserting state machines only allow valid transitions.

---

### Phase 13: Incremental Summarization Service (`app/services/summary_service.py`)
* **Objective:** Process conversation summaries incrementally to preserve prompt constraints.
* **Tasks:**
  * Implement [summary_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/summary_service.py).
  * Fetch the previous summary from Cassandra.
  * Fetch the latest 20 messages from `conversation_recent_messages`.
  * Call the LLM gRPC service, passing only `previous_summary + new_messages` (the **Incremental Summarization Algorithm**).
  * Write the new summary back to Cassandra and invalidate the Redis cache.
* **Verification:** Verify that prompts generated for the LLM stay bounded and don't dump the entire message history.

---

### Phase 14: User Fact Merging Logic (`app/services/long_memory_service.py`)
* **Objective:** Implement long-term fact extraction and the Fact Merge Policy.
* **Tasks:**
  * Implement [long_memory_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/long_memory_service.py).
  * Query existing facts for a user from Milvus using similarity thresholds (`FACT_MERGE_SIMILARITY_THRESHOLD`).
  * Apply the **Fact Merge Policy**:
    * If similarity is low: Insert as a new fact.
    * If similarity is high and new importance is higher: Supersede the old fact (delete-then-insert update sequence).
    * If similarity is high but new importance is lower: Ignore.
    * If statement is identical: Skip.
* **Verification:** Assert that similar facts correctly trigger updates or discards according to the merge policy.

---

### Phase 15: Decoupled Scoring & Ranking Engine (`app/services/ranking_service.py`)
* **Objective:** Calculate retrieval importance weights.
* **Tasks:**
  * Implement [ranking_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/ranking_service.py).
  * Combine vector similarity, fact importance, and recency decay.
  * Code exponential decay using time differences:
    $$\text{Score} = w_{\text{sim}} S_{\text{sim}} + w_{\text{rec}} e^{-\lambda t} + w_{\text{imp}} S_{\text{imp}}$$
* **Verification:** Run unit tests confirming score distributions correspond to decay expectations.

---

### Phase 16: Structured Context Builder (`app/services/context_builder.py` & `retrieval_service.py`)
* **Objective:** Retrieve memory context concurrently with graceful fallbacks.
* **Tasks:**
  * Implement [retrieval_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/retrieval_service.py) with a read-through fallback logic (Redis cache miss -> load from Cassandra `conversation_recent_messages` -> repopulate Redis).
  * Implement [context_builder.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/context_builder.py) to gather short-term window, summary, ancestor summaries (Graph client), and long-term facts.
  * Code a 200ms timeout on the Graph client call; fall back to the current summary only if it exceeds this threshold.
* **Verification:** Test the retrieval flow under mocked Graph Service timeouts and check that context is assembled without blocking.

---

### Phase 17: Outbox Daemon Worker (`app/workers/outbox_worker.py`)
* **Objective:** Build the reliable message delivery daemon using LWT.
* **Tasks:**
  * Implement [outbox_worker.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/workers/outbox_worker.py) running a loop.
  * Poll Cassandra `outbox_jobs` for `PENDING` records.
  * Atomically transition status `PENDING` -> `PROCESSING` using Cassandra LWT (`IF status = 'PENDING'`).
  * On success, publish to Kafka (ordering preserved by routing on `conversation_id`).
  * Delete the job row on successful publish.
  * Respect the configurable `OUTBOX_POLL_INTERVAL_MS` environment setting.
* **Verification:** Spawn multiple outbox worker loops concurrently and confirm LWT prevents duplicate publishing to Kafka.

---

### Phase 18: Kafka Consumer Ingestor & Retry Scheduler (`app/events/`)
* **Objective:** Manage incoming events and retry pipelines.
* **Tasks:**
  * Implement `kafka_consumer.py` to poll API messages.
  * Implement `dispatcher.py` to check event idempotency and commit offsets.
  * Implement a retry scheduler that polls `retry_jobs`, checks if `next_retry < NOW()`, increments attempts, and re-dispatches up to `max_retry` before moving the job to the DLQ.
* **Verification:** Test that failed consumer processing correctly populates the `retry_jobs` schedule and eventually lands in the DLQ.

---

### Phase 19: Background Worker Daemons (`app/workers/`)
* **Objective:** Implement async consumers for heavy summarization and extraction tasks.
* **Tasks:**
  * Implement workers for summary generation, fact extraction, and vector embedding.
  * Each worker claims task records from the Kafka topic, calls the LLM gRPC pool, persists changes to Cassandra, and dispatches the next step.
* **Verification:** Verify end-to-end event flows: ingestion -> snapshot batch commit -> outbox dispatch -> summary worker -> fact worker -> embedding worker -> ready.

---

### Phase 20: REST Controllers, Probes, and Cleanup Reaper (`app/api/` & `app/workers/`)
* **Objective:** Expose HTTP API endpoints, setup cleanup reapers, and publish readiness stats.
* **Tasks:**
  * Implement REST context endpoints under `api/internal/memory.py` using `FastAPI`.
  * Implement the Cleanup Worker (Reaper) in `app/workers/cleanup_worker.py` to sweep the `outbox_processing_index` table using time-bucket ranges, reclaiming stuck `PROCESSING` outbox rows without triggering `ALLOW FILTERING`.
  * Code readiness probes (`/ready`) that verify connection health across Cassandra, Redis, Milvus, and LLM gRPC channel pools.
  * Export Prometheus metrics.
* **Verification:** Run load tests validating latency profiles, check `/ready` output, and verify cleanup reaper resets stuck jobs cleanly.

