# GraphGPT Memory Service — Implementation Update 1
**Covering Phases 1 to 6 (Foundation, Core, DB Adapters, Utilities, Schemas & Idempotency)**  
**Version:** 1.0  
**Date:** 2026-08-02  

---

## 1. Introduction & Executive Summary

The GraphGPT Memory Service is a production-grade cognitive memory system designed to manage conversation context, versioned summaries, user facts, and semantic vectors at a scale of 100M+ users. 

This document details the work accomplished during the first six phases of implementation. All core structural components, adapters, serialization blocks, locking schemas, validation endpoints, and idempotency validators have been fully coded, test-verified, and pushed to the main repository branch.

### Summary of Accomplishments:
* **Architecture Shift**: Established Apache Cassandra as the authoritative source of truth. Redis has been configured strictly as a transient hot cache, ensuring no data loss occurs upon Redis flushes.
* **Compatibility Fixes**: Solved Python 3.12 event-loop driver collection issues natively on Windows systems by integrating the `pyasyncore` backport dependency.
* **Robust Synchronization**: Coded a tokenized Redlock mechanism using Lua scripts for atomic releases and an async `RedisLockWatchdog` background task to extend lock TTLs.
* **Schema Decoupling**: Implemented request contracts that conceal vector embeddings from HTTP clients, preventing model leaks across service boundaries.
* **Durable Idempotency**: Configured Cassandra-driven event deduplication using a 7-day TTL registry.
* **Test Verification**: Created a test pipeline covering 27 unit and integration test assertions (all currently green).

---

## 2. Phase 1: Environment & Project Setup

Phase 1 established the workspace dependencies, virtual runtime configurations, and environmental variable schemas.

### 2.1 Dependencies Configuration (`pyproject.toml`)
We configured the project using modern Python libraries, explicitly pinning components for framework speed, driver stability, and validation strictness:
```toml
[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.109.0"
uvicorn = "^0.27.0"
aiokafka = "^0.10.0"
redis = {extras = ["hiredis"], version = "^5.0.1"}
pymilvus = "^2.3.5"
cassandra-driver = "^3.29.0"          # Primary source of truth — Cassandra
grpcio = "^1.60.0"
grpcio-tools = "^1.60.0"
pydantic = "^2.6.0"
pydantic-settings = "^2.1.0"
httpx = "^0.26.0"
prometheus-client = "^0.19.0"
zstandard = "^0.22.0"                 # zstd compression for summary texts
pyasyncore = "^1.0.5"                 # Python 3.12 compatibility for cassandra-driver
```

### 2.2 Python 3.12 Compatibility Design
In Python 3.12, the standard library `asyncore` module was deprecated and removed. Datastax's `cassandra-driver` internally imports `asyncore` to establish default event loop connections. This triggers a `ModuleNotFoundError` during cluster initialization on Python 3.12.
To resolve this, we integrated `pyasyncore` in the project dependencies. This package injects `asyncore` back into `sys.modules`, allowing the Cassandra driver to load its connection classes cleanly on Python 3.12 without requiring complex container workarounds or downgrading to Python 3.11.

---

## 3. Phase 2: Configuration & Logging Core

Phase 2 focused on creating validated configuration schemas and structured, context-aware JSON logging.

### 3.1 Settings Management (`app/core/config.py`)
Configurations are managed via Pydantic Settings V2, grouped logically:
* **Kafka**: Broker connection parameters and internal worker topic specifications.
* **Cassandra**: Contact points, keyspace names, and transaction timeout rules.
* **Redis**: Cache expiration policies, Redlock TTLs, and watchdog check durations.
* **gRPC Pool**: Size limits, request timeouts, and active channel health intervals.
* **Circuit Breakers**: Failure thresholds and recovery durations for HTTP/gRPC clients.

### 3.2 Context-Aware JSON Logging (`app/core/logging.py`)
To enable end-to-end trace correlation in a distributed pipeline, we wrote a logging infrastructure powered by `contextvars.ContextVar`.
When a worker consumes an event or the API receives a request, the trace parameters (`trace_id`, `conversation_id`, `event_id`, and `summary_version`) are bound to the execution thread context. A custom `ContextFilter` automatically injects these into every `LogRecord`:

```python
# app/core/logging.py
from contextvars import ContextVar
import logging

var_trace_id: ContextVar[str] = ContextVar("trace_id", default=None)
var_conversation_id: ContextVar[str] = ContextVar("conversation_id", default=None)

class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = var_trace_id.get()
        record.conversation_id = var_conversation_id.get()
        return True
```
For production, the log is output as a structured JSON string containing these parameters, which is parsed by centralized logging platforms (e.g., Elasticsearch or Datadog). For local development, a colored `ConsoleFormatter` prints human-readable strings with trace headers prefixed.

---

## 4. Phase 3: Database Connection Adapters

Phase 3 established connection pools for Cassandra, Redis, and Milvus, as well as a centralized lifecycle orchestrator in `app/db/session.py`.

### 4.1 Cassandra Adapter (`app/db/cassandra.py`)
The adapter initializes contact points and applies the 8 CQL tables required by the design:
```sql
-- Schema definitions applied on cluster initialization
CREATE TABLE IF NOT EXISTS conversation_snapshots (
    conversation_id     TEXT,
    user_id             TEXT,
    message_count       INT,
    state               TEXT,
    summary_version     INT,
    fact_version        INT,
    snapshot_version    INT,
    last_summary_msg_id TEXT,
    updated_at          TIMESTAMP,
    PRIMARY KEY (conversation_id)
);

CREATE TABLE IF NOT EXISTS conversation_recent_messages (
    conversation_id     TEXT,
    message_id          TEXT,
    role                TEXT,
    content             TEXT,
    created_at          TIMESTAMP,
    PRIMARY KEY (conversation_id, created_at, message_id)
) WITH CLUSTERING ORDER BY (created_at DESC, message_id ASC);

CREATE TABLE IF NOT EXISTS processed_events (
    event_id            TEXT,
    conversation_id     TEXT,
    processed_at        TIMESTAMP,
    PRIMARY KEY (event_id)
) WITH default_time_to_live = 604800;

CREATE TABLE IF NOT EXISTS outbox_jobs (
    job_id              UUID,
    status              TEXT,
    topic               TEXT,
    conversation_id     TEXT,
    payload             TEXT,
    attempt_count       INT,
    last_error          TEXT,
    created_at          TIMESTAMP,
    claimed_at          TIMESTAMP,
    PRIMARY KEY (status, created_at, job_id)
) WITH CLUSTERING ORDER BY (created_at ASC, job_id ASC);

CREATE TABLE IF NOT EXISTS outbox_processing_index (
    claimed_date        TEXT,
    claimed_at          TIMESTAMP,
    job_id              UUID,
    PRIMARY KEY ((claimed_date), claimed_at, job_id)
) WITH CLUSTERING ORDER BY (claimed_at ASC, job_id ASC);
```

### 4.2 Centralized Startup Lifecycle (`app/db/session.py`)
To prevent the application from booting with broken dependencies, we implemented a strict fail-fast connection initialization sequence:
1. **Cassandra**: Verified with `SELECT now() FROM system.local`.
2. **Redis**: Verified with `client.ping()`.
3. **Milvus**: Verified with `utility.list_collections()`.

If any connection fails, the orchestrator prints a critical log, rolls back any established connections in reverse order, and raises a `RuntimeError` naming the failing service.

```python
# app/db/session.py snippet
async def initialize_db_sessions() -> None:
    # 1. Connect Cassandra
    connect_cassandra()
    cassandra_ready = check_cassandra_ready()

    # 2. Connect Redis
    init_redis_pool(settings.REDIS_URL)
    # Ping Redis...
    
    # 3. Connect Milvus
    connect_milvus(settings.MILVUS_HOST, settings.MILVUS_PORT)
    # List collections...

    if not cassandra_ready or not redis_ready or not milvus_ready:
        # Tear down partially initialized pools
        await close_db_sessions()
        raise RuntimeError("Database session initialization failed.")
```

---

## 5. Phase 4: Shared Utilities

Phase 4 implemented core algorithms for compression, distributed locking, serialization, and latency monitoring.

### 5.1 Zstd Compression Utility (`app/utils/compression.py`)
We used the `zstandard` library to compress summaries and metadata payloads before caching them in Redis:
```python
# app/utils/compression.py
import zstandard as zstd

def compress_string(text: str, level: int = 3) -> bytes:
    if not text:
        return b""
    cctx = zstd.ZstdCompressor(level=level)
    return cctx.compress(text.encode("utf-8"))

def decompress_to_string(compressed: bytes) -> str:
    if not compressed:
        return ""
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(compressed).decode("utf-8")
```

### 5.2 Tokenized Redlock & Watchdog Heartbeat (`app/utils/locks.py`)
A major issue with naive distributed locks is that slow tasks (like LLM generations) can exceed the lock TTL, allowing another worker to acquire the lock and perform duplicate work. If the first worker completes its task and calls `delete(lock_key)`, it might delete the new worker's lock.

To solve this, we implemented a token-based lock:
* **UUID Tokens**: Lock acquisition writes a unique UUID value to the Redis key.
* **Atomic Releases**: Lock release evaluates a Lua script to ensure the lock is deleted *only* if the key holds the caller's UUID.
* **Watchdog Heartbeat**: The `RedisLockWatchdog` class runs an async background heartbeat that periodically inspects the key and extends its expiration while the main task is still running.

```python
# app/utils/locks.py snippet
UNLOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

async def release_redis_lock(client: aioredis.Redis, lock_key: str, owner_token: str) -> bool:
    result = await client.eval(UNLOCK_LUA, 1, lock_key, owner_token)
    return bool(result)
```

---

## 6. Phase 5: Domain Models & Pydantic Schemas

Phase 5 defined domain representations and request/response DTO structures.

### 6.1 Domain Models
We wrote lightweight models to represent the core states:
* `ConversationSnapshot`: Represents metadata only, containing `message_count`, `state`, `summary_version`, and watermarks.
* `SummaryRecord`: Tracks the text content of versioned summaries.
* `MemoryState` enum: Defines state transitions for pipeline tracking.

### 6.2 Vector-Concealed HTTP Schemas (`app/schemas/requests.py`)
Exposing vector embeddings over public API endpoints couples clients to specific models and configurations.
To decouple the API interface, we designed the REST context request schemas to accept a raw text query, keeping embedding generation private to the service:

```python
# app/schemas/requests.py
from pydantic import BaseModel, Field

class GetContextRequest(BaseModel):
    conversation_id: str = Field(...)
    user_id: str = Field(...)
    query: str = Field(..., description="Raw text search query. Embedding is generated internally.")
    top_k_facts: int = Field(10, ge=1, le=50)
```

---

## 7. Phase 6: Event Idempotency Service

Phase 6 built the deduplication engine to prevent duplicate event ingestion.

### 7.1 Processed Event Repository (`app/repositories/processed_event_repository.py`)
Handles access to the `processed_events` table in Cassandra:
```python
# app/repositories/processed_event_repository.py snippet
class ProcessedEventRepository:
    def __init__(self, session: Session):
        self.session = session
        self._select_stmt = self.session.prepare(
            "SELECT event_id FROM processed_events WHERE event_id = ?"
        )
        self._insert_stmt = self.session.prepare(
            "INSERT INTO processed_events (event_id, conversation_id, processed_at) VALUES (?, ?, ?)"
        )

    def is_event_processed(self, event_id: str) -> bool:
        rows = self.session.execute(self._select_stmt, (event_id,))
        return bool(rows.one())
```

### 7.2 Idempotency Check Gating (`app/services/idempotency_service.py`)
The service checks the database for duplicate IDs before starting mutations, raising a `DeduplicationException` if found:
```python
# app/services/idempotency_service.py
class IdempotencyService:
    def __init__(self, processed_event_repo: ProcessedEventRepository):
        self.processed_event_repo = processed_event_repo

    def check_and_register(self, event_id: str, conversation_id: str) -> None:
        if self.processed_event_repo.is_event_processed(event_id):
            raise DeduplicationException(event_id)
        self.processed_event_repo.register_event(event_id, conversation_id)
```

---

## 8. Automated Testing & Verification

A robust test pipeline is implemented using `pytest` and `pytest-asyncio`. Tests cover unit verifications and integration checks against live database instances.

### 8.1 Current Test Execution Status
We execute tests inside the virtual environment:
```bash
.\.venv\Scripts\python.exe -m pytest -v
```

All 27 assertions passed successfully:
```text
tests/integration/test_idempotency.py::test_idempotency_service_flow PASSED
tests/unit/test_core.py::test_system_settings_loading PASSED
tests/unit/test_core.py::test_custom_exceptions PASSED
tests/unit/test_core.py::test_context_vars_and_logging PASSED
tests/unit/test_db.py::TestCassandraAdapter::test_connect_cassandra_parses_multi_node_hosts PASSED
tests/unit/test_db.py::TestCassandraAdapter::test_check_cassandra_ready_returns_true_on_success PASSED
tests/unit/test_db.py::TestCassandraAdapter::test_check_cassandra_ready_returns_false_on_exception PASSED
tests/unit/test_db.py::TestCassandraAdapter::test_check_cassandra_ready_returns_false_when_not_initialized PASSED
tests/unit/test_db.py::TestCassandraAdapter::test_get_session_raises_if_not_initialized PASSED
tests/unit/test_db.py::TestCassandraAdapter::test_disconnect_cassandra_shuts_down_both PASSED
tests/unit/test_db.py::TestSessionOrchestrator::test_initialization_success_all_three_healthy PASSED
tests/unit/test_db.py::TestSessionOrchestrator::test_cassandra_failure_raises_runtime_error PASSED
tests/unit/test_db.py::TestSessionOrchestrator::test_redis_failure_raises_runtime_error PASSED
tests/unit/test_db.py::TestSessionOrchestrator::test_milvus_failure_raises_runtime_error PASSED
tests/unit/test_db.py::TestSessionOrchestrator::test_close_sessions_completes_without_error PASSED
tests/unit/test_schemas.py::test_domain_models_initialization PASSED
tests/unit/test_schemas.py::test_kafka_event_payloads_validation PASSED
tests/unit/test_schemas.py::test_api_requests_hides_vectors PASSED
tests/unit/test_schemas.py::test_api_responses_serialization PASSED
tests/unit/test_utils.py::test_compression_and_decompression PASSED
tests/unit/test_utils.py::test_redis_lock_lifecycle_success PASSED
tests/unit/test_utils.py::test_redis_lock_acquire_failure PASSED
tests/unit/test_utils.py::test_redis_lock_release_mismatch PASSED
tests/unit/test_utils.py::test_redis_lock_watchdog_heartbeat PASSED
tests/unit/test_utils.py::test_redis_lock_watchdog_stops_if_ownership_lost PASSED
tests/unit/test_utils.py::test_json_serialization PASSED
tests/unit/test_utils.py::test_timer_context_manager PASSED

======================= 27 passed in 6.87s =======================
```

---

## 9. Architectural Design Patterns & Decisions

During these phases, we made several key architectural decisions:

### Pattern 1: Cassandra as Source of Truth
* **Context**: Naive designs use Redis as the primary database, leaving the system vulnerable to data loss upon Redis flushes.
* **Decision**: All core state is written to Cassandra. If Redis fails, read-through fallbacks automatically rehydrate the cache from Cassandra.

### Pattern 2: Redis as a Pure Hot Cache
* **Context**: Long summary texts and metadata snapshots are read frequently during chat operations.
* **Decision**: Redis acts as a high-speed cache. To minimize memory footprint, summary text is stored compressed with `zstd`.

### Pattern 3: Tokenized Locking
* **Context**: Stale workers can delete newer locks if they complete their operations past the lock TTL.
* **Decision**: We use UUIDs to track lock ownership and Lua scripts to ensure clients can only release locks they own.

### Pattern 4: Avoiding ALLOW FILTERING Scans
* **Context**: Scanning the `outbox_jobs` table for stuck `PROCESSING` jobs requires filtering on a non-partition column, which triggers a full table scan in Cassandra.
* **Decision**: We added an `outbox_processing_index` table partitioned by date. Stale jobs are identified by querying this index for the target date, avoiding full scans.

### Pattern 5: Vector Concealment
* **Context**: Exposing vector embeddings over the public API forces clients to know about the embedding model.
* **Decision**: The REST API takes raw query text. The Memory Service calls the LLM Service to generate the embedding internally, keeping model configurations private.

---

## 10. Conclusion & Next Steps

Phases 1 through 6 have established a stable foundation for the Memory Service. The core databases, logging systems, and utilities are fully implemented and verified.

We are ready to proceed to:
* **Phase 7**: Cassandra Repository Layer (`app/repositories/cassandra_repository.py`)
* **Phase 8**: Redis Repository Layer (`app/repositories/redis_repository.py`)
* **Phase 9**: Snapshot Builder Service DML implementation.
