# GraphGPT Memory Service — Low-Level Design (LLD)
**Version:** 3.0 (Production-Grade)  
**Status:** Approved for Implementation  
**HLD Reference:** [hld.md](./hld.md)  
**Scale:** 100M Users, 1M+ Writes/sec  
**Last Updated:** 2026-08-02  

---

## Revision History

| Version | Date | Description |
| :--- | :--- | :--- |
| 1.0 | 2026-07-31 | Initial draft |
| 2.0 | 2026-08-01 | Worker architecture, idempotency, state machine |
| 2.3 | 2026-08-01 | Cassandra schemas, outbox pattern, Redis watchdogs |
| **3.0** | **2026-08-02** | **Critical fixes: LWT claiming, lock ownership, snapshot schema, partition strategy, gRPC health checks, configurable polling, circuit breaker** |

---

## 1. Directory Structure

```text
memory-service/
│
├── app/
│   ├── api/
│   │   ├── dependencies.py            # FastAPI dependency injection
│   │   ├── routers.py                 # Route registration
│   │   └── internal/
│   │       ├── health.py              # GET /health, GET /ready
│   │       └── memory.py             # GET /internal/memory/context
│   │
│   ├── clients/
│   │   ├── graph_client.py            # Graph Service HTTP client
│   │   ├── llm_client.py             # gRPC pool client (LLM + Embedding)
│   │   ├── milvus_client.py          # Milvus bulk insert + search client
│   │   └── redis_client.py           # Redis async client factory
│   │
│   ├── core/
│   │   ├── config.py                 # Pydantic Settings V2
│   │   ├── constants.py              # Enum, topic names, key prefixes
│   │   ├── exceptions.py             # CircuitBreakerOpen, Deduplication, JobExecution
│   │   ├── logging.py                # JSON structured logger + context vars
│   │   └── security.py               # Service-to-service auth
│   │
│   ├── db/
│   │   ├── redis.py                  # Redis ConnectionPool lifecycle
│   │   ├── milvus.py                 # Milvus connect/disconnect/ping
│   │   ├── cassandra.py             # Cassandra Cluster + Session management
│   │   └── session.py               # Startup/shutdown orchestrator
│   │
│   ├── events/
│   │   ├── kafka_consumer.py         # AIOKafka consumer runner
│   │   ├── kafka_producer.py         # AIOKafka producer
│   │   ├── dispatcher.py            # Event routing + idempotency gate
│   │   ├── consumers.py             # Typed event handler registry
│   │   ├── producers.py             # Typed outbox publish helpers
│   │   └── topics.py                # Topic name constants
│   │
│   ├── models/
│   │   ├── event.py                 # Kafka event models
│   │   ├── memory.py                # MemoryState enum, MemoryRecord
│   │   ├── snapshot.py             # ConversationSnapshot dataclass (lightweight)
│   │   ├── summary.py              # SummaryRecord with versioning
│   │   ├── context.py              # ContextResponse assembly model
│   │   └── embedding.py            # EmbeddingRecord + versioning
│   │
│   ├── repositories/
│   │   ├── cassandra_repository.py  # All Cassandra read/write operations
│   │   ├── redis_repository.py      # All Redis cache read/write operations
│   │   ├── milvus_repository.py    # Milvus collection + search operations
│   │   └── processed_event_repository.py  # Idempotency table
│   │
│   ├── schemas/
│   │   ├── requests.py              # Pydantic request validators
│   │   ├── responses.py             # Pydantic response serializers
│   │   ├── grpc.py                 # gRPC request/response schemas
│   │   └── events.py               # Kafka event schemas
│   │
│   ├── services/
│   │   ├── memory_service.py        # Top-level coordinator
│   │   ├── snapshot_service.py      # Snapshot Builder + Outbox commit
│   │   ├── summary_service.py       # Summary generation + versioning
│   │   ├── long_memory_service.py   # User fact persistence + retrieval
│   │   ├── semantic_memory_service.py # Semantic embedding management
│   │   ├── retrieval_service.py     # Cache + Cassandra + Milvus lookup
│   │   ├── ranking_service.py       # Score assembly (recency × importance × similarity)
│   │   ├── context_builder.py      # Final context JSON assembler
│   │   ├── embedding_service.py    # Embedding request orchestrator
│   │   ├── cleanup_service.py      # TTL expiry + soft-delete sweeps
│   │   └── idempotency_service.py  # Event deduplication logic
│   │
│   ├── workers/
│   │   ├── outbox_worker.py        # Outbox Daemon: LWT claim → Kafka publish → DELETE
│   │   ├── summary_worker.py       # Summary Worker consumer
│   │   ├── fact_worker.py          # Fact Extraction Worker consumer
│   │   ├── embedding_worker.py     # Embedding Bulk Insert Worker consumer
│   │   ├── delete_worker.py        # Memory deletion worker
│   │   └── cleanup_worker.py       # Reaper: reclaim stale PROCESSING rows
│   │
│   ├── utils/
│   │   ├── compression.py          # zlib compress/decompress for summary text
│   │   ├── hashing.py              # SHA-256 content hash deduplication
│   │   ├── locks.py                # RedisLockWatchdog with UUID ownership tokens
│   │   ├── ranking.py              # Scoring math helpers
│   │   ├── serialization.py        # CustomJSONEncoder (datetime, UUID, Pydantic)
│   │   └── timers.py               # Context manager latency timer
│   │
│   ├── proto/
│   │   ├── llm.proto               # gRPC LLM service contract
│   │   └── graph.proto             # gRPC Graph service contract
│   │
│   ├── main.py                     # FastAPI app factory
│   └── lifespan.py                 # Startup/shutdown hooks
│
├── tests/
│   ├── unit/
│   │   ├── test_core.py
│   │   ├── test_db.py
│   │   ├── test_utils.py
│   │   └── test_repositories.py
│   └── integration/
│       └── test_cassandra.py
│
├── docs/
│   ├── hld.md
│   ├── lld.md
│   └── phases.md
│
├── Makefile
├── pyproject.toml
├── requirements.txt
├── docker-compose.yml
└── .env
```

---

## 2. Configuration

```python
# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class SystemSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Application
    APP_NAME: str = "graphgpt-memory-service"
    APP_ENV: str = "production"
    DEBUG: bool = False
    HTTP_HOST: str = "0.0.0.0"
    HTTP_PORT: int = 8000

    # Cassandra
    CASSANDRA_HOSTS: str = "localhost"        # comma-separated for multi-node
    CASSANDRA_PORT: int = 9042
    CASSANDRA_KEYSPACE: str = "graphgpt_memory"
    CASSANDRA_TIMEOUT_SECONDS: float = 5.0

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_GROUP_ID: str = "memory-service-consumers"
    KAFKA_SESSION_TIMEOUT_MS: int = 30000
    KAFKA_MAX_POLL_INTERVAL_MS: int = 300000
    KAFKA_SUMMARY_TOPIC: str = "memory.summary.request"
    KAFKA_FACT_TOPIC: str = "memory.fact.request"
    KAFKA_EMBEDDING_TOPIC: str = "memory.embedding.request"
    KAFKA_DELETE_TOPIC: str = "memory.delete.request"
    KAFKA_DLQ_TOPIC: str = "memory.dlq"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    SNAPSHOT_TTL_SECONDS: int = 2592000     # 30 days
    SHORT_TERM_MESSAGE_LIMIT: int = 20
    IDEMPOTENCY_TTL_SECONDS: int = 604800   # 7 days
    REDIS_LOCK_TTL_SECONDS: int = 5
    REDIS_LOCK_WATCHDOG_INTERVAL: float = 2.0

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    VECTOR_DIMENSION: int = 1536
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    EMBEDDING_MODEL_VERSION: str = "v1.0.0"
    MILVUS_BULK_INSERT_BATCH_SIZE: int = 100

    # gRPC
    LLM_SERVICE_HOST: str = "localhost"
    LLM_SERVICE_PORT: int = 50051
    GRPC_POOL_SIZE: int = 5
    GRPC_TIMEOUT_SECONDS: float = 5.0
    GRPC_HEALTH_CHECK_INTERVAL_SECONDS: float = 30.0

    # Graph Service
    GRAPH_SERVICE_URL: str = "http://localhost:8001"

    # Retrieval Scoring Weights
    RETRIEVAL_WEIGHT_SIMILARITY: float = 0.5
    RETRIEVAL_WEIGHT_RECENCY: float = 0.2
    RETRIEVAL_WEIGHT_IMPORTANCE: float = 0.3
    RETRIEVAL_DECAY_RATE: float = 0.05
    RETRIEVAL_TOP_K_FACTS: int = 10
    FACT_MERGE_SIMILARITY_THRESHOLD: float = 0.85

    # Short-Term Memory Window
    # Recent messages are stored in BOTH Redis (hot) and Cassandra (durable).
    # Kafka is transport only — never relied upon for recovery.
    SHORT_TERM_MESSAGE_LIMIT: int = 20

    # Outbox
    OUTBOX_POLL_INTERVAL_MS: int = 1000     # Configurable polling interval
    OUTBOX_BATCH_SIZE: int = 50
    OUTBOX_STALE_PROCESSING_MINUTES: int = 5

    # Circuit Breaker
    CB_FAILURE_THRESHOLD: int = 5
    CB_RECOVERY_TIMEOUT_SECONDS: float = 60.0
    CB_HALF_OPEN_LIMIT: int = 2

settings = SystemSettings()
```

---

## 3. Cassandra Schema (CQL)

> **Design Note:** Cassandra Logged Batches used in this service guarantee atomic mutation delivery (all mutations are applied or retried). They do **NOT** provide ACID transaction isolation. All write paths are designed to be idempotent to compensate for Cassandra's lack of rollback support.

```sql
CREATE KEYSPACE IF NOT EXISTS graphgpt_memory
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3}
AND durable_writes = true;

USE graphgpt_memory;

-- ─────────────────────────────────────────────────────────────────────────────
-- Table 1: Lightweight Conversation Snapshots
-- Stores metadata state only. Recent messages are NOT stored here.
-- Recent messages live in Redis; source is Kafka (for replay).
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversation_snapshots (
    conversation_id     TEXT,
    user_id             TEXT,
    message_count       INT,
    state               TEXT,       -- MemoryState enum value
    summary_version     INT,        -- Increments on each new summary
    fact_version        INT,        -- Increments on each fact batch commit
    snapshot_version    INT,        -- Monotonic mutation counter
    last_summary_msg_id TEXT,       -- Pointer to message that triggered last summary
    updated_at          TIMESTAMP,
    PRIMARY KEY (conversation_id)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Table 2: Conversation Summaries
-- Versioned for cache invalidation. Includes LLM model tracking.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversation_summaries (
    conversation_id     TEXT,
    summary_text        TEXT,
    summary_version     INT,
    model_name          TEXT,
    model_version       TEXT,
    generated_at        TIMESTAMP,
    PRIMARY KEY (conversation_id)
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Table 3: Idempotency Registry
-- 7-day TTL prevents duplicate processing of replayed Kafka events.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS processed_events (
    event_id            TEXT,
    conversation_id     TEXT,
    processed_at        TIMESTAMP,
    PRIMARY KEY (event_id)
) WITH default_time_to_live = 604800;

-- ─────────────────────────────────────────────────────────────────────────────
-- Table 4: Outbox Jobs
-- Includes status + attempt tracking for worker claiming and reaper.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS outbox_jobs (
    job_id              UUID,
    status              TEXT,       -- PENDING | PROCESSING | DONE | FAILED
    topic               TEXT,
    conversation_id     TEXT,
    payload             TEXT,       -- JSON-serialized task payload
    attempt_count       INT,
    last_error          TEXT,
    created_at          TIMESTAMP,
    claimed_at          TIMESTAMP,  -- Set when status transitions to PROCESSING
    PRIMARY KEY (status, created_at, job_id)
) WITH CLUSTERING ORDER BY (created_at ASC, job_id ASC);

-- ─────────────────────────────────────────────────────────────────────────────
-- Table 5: Retry Jobs
-- Tracks failed background jobs for retry scheduling and DLQ.
-- Partition by status+next_retry for efficient scheduler polling.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS retry_jobs (
    status              TEXT,       -- PENDING | PROCESSING | FAILED
    next_retry          TIMESTAMP,
    job_id              UUID,
    job_type            TEXT,       -- summary | fact | embedding | delete
    payload             TEXT,
    retry_count         INT,
    max_retry           INT,        -- Per-job-type configurable retry limit (e.g. summary=5, embedding=2)
    last_error          TEXT,
    created_at          TIMESTAMP,
    PRIMARY KEY ((status), next_retry, job_id)
) WITH CLUSTERING ORDER BY (next_retry ASC, job_id ASC);

-- ─────────────────────────────────────────────────────────────────────────────
-- Table 5b: Outbox Processing Index (for Stale PROCESSING reaping)
-- Avoids ALLOW FILTERING on the main outbox_jobs table.
-- The cleanup worker inserts a row here when it claims a job.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS outbox_processing_index (
    claimed_date        TEXT,       -- DATE string (e.g. '2026-08-02') for time-bucketing
    claimed_at          TIMESTAMP,
    job_id              UUID,
    PRIMARY KEY ((claimed_date), claimed_at, job_id)
) WITH CLUSTERING ORDER BY (claimed_at ASC, job_id ASC);

-- ─────────────────────────────────────────────────────────────────────────────
-- Table 6: User Facts (Long-Term Memory)
-- Partition by (user_id, category) for fast category-scoped lookups.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_facts (
    user_id             TEXT,
    category            TEXT,       -- preferences | decisions | habits | entities
    fact_id             UUID,
    conversation_id     TEXT,
    statement           TEXT,
    importance          FLOAT,
    fact_version        INT,
    embedding_version   TEXT,       -- model version used to generate embedding
    created_at          TIMESTAMP,
    updated_at          TIMESTAMP,
    PRIMARY KEY ((user_id, category), fact_id)
) WITH CLUSTERING ORDER BY (fact_id ASC);

-- ─────────────────────────────────────────────────────────────────────────────
-- Table 7: Recent Message Window (Short-Term Memory Durable Backup)
-- Addresses a critical gap: Redis can be evicted; Kafka retention is finite
-- (typically 7 days). Conversations can be months old.
-- Redis is the hot cache. This table is the durable fallback.
-- Only the last N messages are kept; older rows are deleted asynchronously.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversation_recent_messages (
    conversation_id     TEXT,
    message_id          TEXT,       -- UUID string from Conversation Service
    role                TEXT,       -- 'user' | 'assistant' | 'system'
    content             TEXT,
    created_at          TIMESTAMP,
    PRIMARY KEY (conversation_id, created_at, message_id)
) WITH CLUSTERING ORDER BY (created_at DESC, message_id ASC);
```

---

## 4. Snapshot Model (Lightweight — No Recent Messages JSON)

```python
# app/models/snapshot.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum

class MemoryState(str, Enum):
    ACTIVE = "ACTIVE"
    SUMMARY_PENDING = "SUMMARY_PENDING"
    SUMMARIZING = "SUMMARIZING"
    FACT_PENDING = "FACT_PENDING"
    EXTRACTING_FACTS = "EXTRACTING_FACTS"
    EMBEDDING_PENDING = "EMBEDDING_PENDING"
    READY = "READY"
    FAILED = "FAILED"

@dataclass
class ConversationSnapshot:
    """
    Lightweight state metadata. Does NOT carry recent_messages as a JSON blob.

    Recent message window storage strategy:
      - Redis (recent:{conversation_id}): hot cache, sub-millisecond reads.
      - Cassandra (conversation_recent_messages): durable backup.
        Used when Redis is cold/evicted AND Kafka retention has expired
        (Kafka retention = 7 days, but conversations can be months old).
      - Kafka: transport only. Never relied on as a recovery source.

    On Redis miss: load from Cassandra, repopulate Redis. No Kafka seek needed.
    """
    conversation_id: str
    user_id: str
    message_count: int
    state: MemoryState
    summary_version: int
    fact_version: int
    snapshot_version: int
    last_summary_msg_id: Optional[str]
    updated_at: datetime
```

---

## 5. Outbox Pattern with LWT Claiming

### 5.1 Atomic Batch Write

```python
# app/services/snapshot_service.py
from cassandra.query import BatchStatement, BatchType
from cassandra.cluster import Session
import uuid, json
from datetime import datetime, timezone

class SnapshotService:
    def __init__(self, session: Session):
        self.session = session
        self._prepare_statements()

    def _prepare_statements(self):
        self._snap_upsert = self.session.prepare("""
            INSERT INTO conversation_snapshots
            (conversation_id, user_id, message_count, state,
             summary_version, fact_version, snapshot_version,
             last_summary_msg_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)
        self._outbox_insert = self.session.prepare("""
            INSERT INTO outbox_jobs
            (job_id, status, topic, conversation_id, payload,
             attempt_count, created_at)
            VALUES (?, 'PENDING', ?, ?, ?, 0, ?)
        """)
        self._idemp_insert = self.session.prepare("""
            INSERT INTO processed_events
            (event_id, conversation_id, processed_at)
            VALUES (?, ?, ?)
        """)

    def commit_snapshot_and_outbox(
        self,
        snapshot: dict,
        event_id: str,
        outbox_topic: str,
        outbox_payload: dict
    ) -> None:
        """
        Writes snapshot state, outbox job, and idempotency marker in a single
        Cassandra Logged Batch. This guarantees atomic delivery across mutations
        (all applied or retried). NOT full ACID — no rollback, no isolation.
        All paths are designed to be idempotent to compensate.
        """
        now = datetime.now(timezone.utc)
        batch = BatchStatement(batch_type=BatchType.LOGGED)

        batch.add(self._snap_upsert, (
            snapshot["conversation_id"],
            snapshot["user_id"],
            snapshot["message_count"],
            snapshot["state"],
            snapshot["summary_version"],
            snapshot["fact_version"],
            snapshot["snapshot_version"],
            snapshot.get("last_summary_msg_id"),
            now,
        ))

        batch.add(self._outbox_insert, (
            uuid.uuid4(),
            outbox_topic,
            snapshot["conversation_id"],
            json.dumps(outbox_payload),
            now,
        ))

        batch.add(self._idemp_insert, (
            event_id,
            snapshot["conversation_id"],
            now,
        ))

        self.session.execute(batch)
```

### 5.2 Outbox Worker with LWT Claiming

```python
# app/workers/outbox_worker.py
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from app.core.config import settings

logger = logging.getLogger("memory_service.outbox_worker")

class OutboxDaemonWorker:
    """
    Polls outbox_jobs for PENDING rows, atomically claims them using
    Cassandra LWT (Lightweight Transactions) to prevent duplicate Kafka
    publishing by concurrent worker instances.

    Flow: PENDING → (LWT) → PROCESSING → Kafka Publish → DELETE
    """
    def __init__(self, cassandra_session, producer):
        self.session = cassandra_session
        self.producer = producer
        self.is_running = False
        self._poll_interval = settings.OUTBOX_POLL_INTERVAL_MS / 1000.0
        self._prepare_statements()

    def _prepare_statements(self):
        # Claim a PENDING row only if status is still PENDING (LWT)
        self._claim_stmt = self.session.prepare("""
            UPDATE outbox_jobs
            SET status = 'PROCESSING', claimed_at = ?
            WHERE status = 'PENDING' AND created_at = ? AND job_id = ?
            IF status = 'PENDING'
        """)
        self._delete_stmt = self.session.prepare("""
            DELETE FROM outbox_jobs
            WHERE status = 'PROCESSING' AND created_at = ? AND job_id = ?
        """)
        self._fail_stmt = self.session.prepare("""
            UPDATE outbox_jobs
            SET attempt_count = ?, last_error = ?
            WHERE status = 'PROCESSING' AND created_at = ? AND job_id = ?
        """)

    async def start(self) -> None:
        self.is_running = True
        logger.info(
            f"Outbox Daemon started. Poll interval: {self._poll_interval}s, "
            f"Batch size: {settings.OUTBOX_BATCH_SIZE}"
        )
        while self.is_running:
            try:
                await self._process_batch()
            except Exception as e:
                logger.error(f"Outbox Daemon loop error: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _process_batch(self) -> None:
        rows = self.session.execute(
            "SELECT job_id, topic, conversation_id, payload, attempt_count, created_at "
            "FROM outbox_jobs WHERE status = 'PENDING' LIMIT %s",
            (settings.OUTBOX_BATCH_SIZE,)
        )

        for row in rows:
            now = datetime.now(timezone.utc)

            # LWT Claim — only one worker wins
            result = self.session.execute(
                self._claim_stmt, (now, row.created_at, row.job_id)
            )
            applied = result.one().applied

            if not applied:
                # Another worker claimed this row first — skip
                continue

            try:
                import json
                payload = json.loads(row.payload)
                await self.producer.publish_task(
                    topic=row.topic,
                    conversation_id=row.conversation_id,
                    payload=payload
                )
                # Successfully published — delete the outbox row
                self.session.execute(
                    self._delete_stmt, (row.created_at, row.job_id)
                )
            except Exception as e:
                logger.error(f"Outbox job {row.job_id} publish failed: {e}")
                # Update attempt count and last error, keep row for reaper
                self.session.execute(
                    self._fail_stmt,
                    (row.attempt_count + 1, str(e), row.created_at, row.job_id)
                )

    async def stop(self) -> None:
        self.is_running = False
```

---

## 6. Cleanup Worker (Stale PROCESSING Reaper)

```python
# app/workers/cleanup_worker.py
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from app.core.config import settings

logger = logging.getLogger("memory_service.cleanup_worker")

class CleanupWorker:
    """
    Periodically reclaims outbox_jobs rows stuck in PROCESSING state
    (e.g. due to worker crashes) and resets them to PENDING for retry.
    """
    def __init__(self, session):
        self.session = session
        self.is_running = False

    async def start(self) -> None:
        self.is_running = True
        while self.is_running:
            try:
                stale_before = datetime.now(timezone.utc) - timedelta(
                    minutes=settings.OUTBOX_STALE_PROCESSING_MINUTES
                )
                # Query the processing index table instead of ALLOW FILTERING on outbox_jobs.
                # outbox_processing_index is keyed on (claimed_date, claimed_at, job_id),
                # which allows efficient time-range scans without a full table scan.
                today = stale_before.strftime('%Y-%m-%d')
                rows = self.session.execute(
                    "SELECT job_id FROM outbox_processing_index "
                    "WHERE claimed_date = %s AND claimed_at < %s",
                    (today, stale_before)
                )
                reclaim_stmt = self.session.prepare("""
                    UPDATE outbox_jobs
                    SET status = 'PENDING', claimed_at = null
                    WHERE status = 'PROCESSING' AND created_at = ? AND job_id = ?
                    IF status = 'PROCESSING'
                """)
                delete_index_stmt = self.session.prepare(
                    "DELETE FROM outbox_processing_index "
                    "WHERE claimed_date = ? AND claimed_at = ? AND job_id = ?"
                )
                count = 0
                for row in rows:
                    result = self.session.execute(reclaim_stmt, (row.created_at, row.job_id))
                    if result.one().applied:
                        self.session.execute(delete_index_stmt, (today, stale_before, row.job_id))
                        count += 1
                if count:
                    logger.info(f"Reaper reclaimed {count} stale PROCESSING outbox rows.")
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
            await asyncio.sleep(60)
```

---

## 7. Redis Lock with UUID Ownership Token & Watchdog

**Issue 8 fix:** The lock value is a random UUID. Only the lock holder who knows the UUID can release it. Watchdog extends TTL during long LLM calls.

```python
# app/utils/locks.py
import asyncio
import logging
import uuid
from typing import Optional
import redis.asyncio as aioredis
from app.core.config import settings

logger = logging.getLogger("memory_service.utils.locks")

UNLOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

async def acquire_redis_lock(
    client: aioredis.Redis,
    lock_key: str,
    ttl_seconds: int = None,
    owner_token: str = None
) -> Optional[str]:
    """
    Acquires a distributed Redis lock using SETNX with a UUID ownership token.
    Returns the token if acquired, None if the lock is already held.
    The caller MUST use the returned token to release the lock.
    """
    ttl = ttl_seconds or settings.REDIS_LOCK_TTL_SECONDS
    token = owner_token or str(uuid.uuid4())
    acquired = await client.set(lock_key, token, ex=ttl, nx=True)
    return token if acquired else None

async def release_redis_lock(
    client: aioredis.Redis,
    lock_key: str,
    owner_token: str
) -> bool:
    """
    Releases the Redis lock ONLY if the caller's owner_token matches.
    Uses Lua script for atomic compare-and-delete.
    """
    try:
        result = await client.eval(UNLOCK_LUA, 1, lock_key, owner_token)
        return bool(result)
    except Exception as e:
        logger.error(f"Failed to release lock '{lock_key}': {e}")
        return False


class RedisLockWatchdog:
    """
    Background coroutine that extends a Redis lock's TTL while the owning
    task is still executing. Prevents expiry during slow LLM calls.
    """
    def __init__(
        self,
        client: aioredis.Redis,
        lock_key: str,
        owner_token: str,
        interval: float = None,
        extend_by: int = None
    ):
        self.client = client
        self.lock_key = lock_key
        self.owner_token = owner_token
        self.interval = interval or settings.REDIS_LOCK_WATCHDOG_INTERVAL
        self.extend_by = extend_by or settings.REDIS_LOCK_TTL_SECONDS
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        """Extend lock TTL only if we still own it."""
        while True:
            try:
                await asyncio.sleep(self.interval)
                current_holder = await self.client.get(self.lock_key)
                if current_holder == self.owner_token:
                    await self.client.expire(self.lock_key, self.extend_by)
                else:
                    logger.warning(
                        f"Watchdog: lock '{self.lock_key}' ownership lost. Stopping watchdog."
                    )
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error on key '{self.lock_key}': {e}")
                break

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
```

---

## 8. gRPC Connection Pool with Health Checks

```python
# app/clients/llm_client.py
import asyncio
import logging
from typing import List, Optional
import grpc
from grpc import aio as grpc_aio
from app.core.config import settings

logger = logging.getLogger("memory_service.grpc_pool")

class AsyncGRPCConnectionPool:
    """
    Pool of async gRPC channels with round-robin routing and periodic health checks.
    Dead channels are replaced automatically.
    """
    def __init__(self, target: str, pool_size: int = None):
        self.target = target
        self.pool_size = pool_size or settings.GRPC_POOL_SIZE
        self._channels: List[Optional[grpc_aio.Channel]] = []
        self._index = 0
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        logger.info(f"Creating gRPC pool: target={self.target}, size={self.pool_size}")
        for _ in range(self.pool_size):
            channel = self._create_channel()
            self._channels.append(channel)
        asyncio.create_task(self._health_check_loop())

    def _create_channel(self) -> grpc_aio.Channel:
        return grpc_aio.insecure_channel(
            self.target,
            options=[
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000),
                ('grpc.keepalive_permit_without_calls', 1),
                ('grpc.max_receive_message_length', 10 * 1024 * 1024),
            ]
        )

    async def get_channel(self) -> grpc_aio.Channel:
        async with self._lock:
            for _ in range(self.pool_size):
                ch = self._channels[self._index]
                self._index = (self._index + 1) % self.pool_size
                if ch is not None:
                    return ch
        raise RuntimeError("No healthy gRPC channels available in pool.")

    async def _health_check_loop(self) -> None:
        """Periodically verify channels and replace dead ones."""
        while True:
            await asyncio.sleep(settings.GRPC_HEALTH_CHECK_INTERVAL_SECONDS)
            for i, channel in enumerate(self._channels):
                if channel is None:
                    self._channels[i] = self._create_channel()
                    logger.info(f"Replaced dead gRPC channel at index {i}.")
                else:
                    try:
                        state = channel.get_state(try_to_connect=True)
                        if state == grpc.ChannelConnectivity.TRANSIENT_FAILURE:
                            logger.warning(
                                f"gRPC channel {i} in TRANSIENT_FAILURE — replacing."
                            )
                            await channel.close()
                            self._channels[i] = self._create_channel()
                    except Exception as e:
                        logger.error(f"gRPC health check error on channel {i}: {e}")

    async def close(self) -> None:
        for i, ch in enumerate(self._channels):
            if ch:
                try:
                    await ch.close()
                except Exception as e:
                    logger.warning(f"Error closing gRPC channel {i}: {e}")
        self._channels.clear()
```

---

## 9. Milvus Partition Strategy & Bulk Ingestion

### 9.1 Partition Key Strategy

Milvus supports `is_partition_key=True` on a scalar field starting from Milvus 2.2.9+. This causes Milvus to internally route vectors to virtual partitions by hash. For 100M users, this is the recommended approach — it avoids explicit partition creation and eliminates the per-user partition limit.

For deployments on older Milvus or at extreme per-user scale, fall back to scalar filtering (`expr="user_id == 'X'"`), which leverages the scalar index.

```python
# app/clients/milvus_client.py
from pymilvus import FieldSchema, DataType, CollectionSchema, Collection, connections, utility

def create_user_memory_collection() -> Collection:
    fields = [
        FieldSchema("fact_id",         DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema("user_id",         DataType.VARCHAR, max_length=64, is_partition_key=True),
        FieldSchema("conversation_id", DataType.VARCHAR, max_length=64),
        FieldSchema("category",        DataType.VARCHAR, max_length=32),
        FieldSchema("statement",       DataType.VARCHAR, max_length=1024),
        FieldSchema("importance",      DataType.FLOAT),
        FieldSchema("fact_version",    DataType.INT32),
        FieldSchema("embedding_ver",   DataType.VARCHAR, max_length=32),
        FieldSchema("created_at",      DataType.DOUBLE),
        FieldSchema("vector",          DataType.FLOAT_VECTOR, dim=1536),
    ]
    schema = CollectionSchema(fields, "User memory vectors with user_id partition key")
    collection = Collection("user_memory_vectors", schema)
    collection.create_index("vector", {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 256}
    })
    return collection
```

### 9.2 Bulk Insert

```python
# app/workers/embedding_worker.py
async def insert_bulk_vectors(self, records: list[dict]) -> None:
    """
    Inserts vectors in configurable batches to reduce I/O round trips.
    Batch size is controlled by MILVUS_BULK_INSERT_BATCH_SIZE setting.
    """
    collection = Collection("user_memory_vectors")
    batch_size = settings.MILVUS_BULK_INSERT_BATCH_SIZE

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        columns = {field: [] for field in batch[0]}
        for rec in batch:
            for field, value in rec.items():
                columns[field].append(value)

        collection.insert([columns[f] for f in columns])
        logger.info(f"Bulk inserted batch of {len(batch)} vectors into Milvus.")

    collection.flush()
```

---

## 10. Repository Pattern

All storage reads are isolated by engine in separate repository classes. Services never touch the DB driver directly.

### 10.1 Cassandra Repository (Write Model)

```python
# app/repositories/cassandra_repository.py
class CassandraRepository:
    def get_snapshot(self, conversation_id: str) -> Optional[dict]: ...
    def upsert_snapshot(self, snapshot: dict) -> None: ...
    def get_summary(self, conversation_id: str) -> Optional[dict]: ...
    def upsert_summary(self, summary: dict) -> None: ...
    def get_facts(self, user_id: str, category: str) -> list[dict]: ...
    def upsert_fact(self, fact: dict) -> None: ...
```

### 10.2 Redis Repository (Read Cache Model)

```python
# app/repositories/redis_repository.py
class RedisRepository:
    async def get_snapshot_cache(self, conversation_id: str) -> Optional[dict]: ...
    async def set_snapshot_cache(self, conversation_id: str, data: dict, ttl: int) -> None: ...
    async def get_summary_cache(self, conversation_id: str) -> Optional[str]: ...
    async def set_summary_cache(self, conversation_id: str, compressed: bytes, ttl: int) -> None: ...
    async def get_recent_messages(self, conversation_id: str) -> list[dict]: ...
    async def push_recent_message(self, conversation_id: str, msg: dict) -> None: ...
    async def trim_recent_messages(self, conversation_id: str, limit: int) -> None: ...
    async def invalidate_context_cache(self, conversation_id: str) -> None: ...
```

---

## 11. Cache Hydration (Read-Through Fallback)

```python
# app/services/retrieval_service.py
async def get_or_hydrate_snapshot(
    self,
    conversation_id: str
) -> Optional[dict]:
    """
    1. Check Redis hot cache.
    2. On miss: read from Cassandra, repopulate Redis.
    """
    cached = await self.redis_repo.get_snapshot_cache(conversation_id)
    if cached:
        REDIS_HIT.inc()
        return cached

    REDIS_MISS.inc()
    snapshot = self.cassandra_repo.get_snapshot(conversation_id)
    if snapshot:
        await self.redis_repo.set_snapshot_cache(
            conversation_id, snapshot, ttl=settings.SNAPSHOT_TTL_SECONDS
        )
    return snapshot
```

---

## 12. Cache Invalidation on State Mutation

```python
# app/services/snapshot_service.py
async def post_commit_invalidation(self, conversation_id: str) -> None:
    """
    After committing snapshot + outbox to Cassandra, delete stale Redis cache
    so the next read gets fresh data from Cassandra.
    """
    await self.redis_repo.invalidate_context_cache(conversation_id)
    logger.info(f"Cache invalidated for conversation: {conversation_id}")
```

---

## 13. Circuit Breaker Pattern

All outbound gRPC and HTTP calls (LLM Service, Graph Service) are wrapped with a circuit breaker to prevent cascading failures.

```python
# app/core/exceptions.py
class CircuitBreakerOpenException(MemoryServiceException):
    def __init__(self, service_name: str, message: str = ""):
        super().__init__(f"Circuit breaker OPEN for service '{service_name}'. {message}")
        self.service_name = service_name

# app/clients/llm_client.py
class LLMClient:
    def __init__(self, pool: AsyncGRPCConnectionPool):
        self.pool = pool
        self._failures = 0
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._last_failure_time: float = 0.0

    async def call_with_circuit_breaker(self, stub_fn, *args, **kwargs):
        if self._state == "OPEN":
            if time.monotonic() - self._last_failure_time > settings.CB_RECOVERY_TIMEOUT_SECONDS:
                self._state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenException("llm-service")

        try:
            channel = await self.pool.get_channel()
            result = await stub_fn(channel, *args, **kwargs)
            if self._state == "HALF_OPEN":
                self._state = "CLOSED"
                self._failures = 0
            return result
        except Exception as e:
            self._failures += 1
            self._last_failure_time = time.monotonic()
            if self._failures >= settings.CB_FAILURE_THRESHOLD:
                self._state = "OPEN"
                logger.critical(f"Circuit breaker OPENED for LLM service after {self._failures} failures.")
            raise
```

---

## 14. Metrics

```python
# app/core/metrics.py
from prometheus_client import Counter, Gauge, Histogram

REDIS_HIT   = Counter("memory_redis_hit_total",   "Redis cache hits")
REDIS_MISS  = Counter("memory_redis_miss_total",  "Redis cache misses")
LOCK_WAIT   = Histogram("memory_redis_lock_wait_seconds", "Lock acquisition wait time")
MILVUS_QPS  = Counter("memory_milvus_queries_total", "Milvus search queries")
CTX_BUILD   = Histogram("memory_context_build_seconds", "Context assembly latency")

SUMMARY_Q   = Gauge("memory_summary_queue_size",   "Pending summary jobs")
FACT_Q      = Gauge("memory_fact_queue_size",      "Pending fact jobs")
EMBEDDING_Q = Gauge("memory_embedding_queue_size", "Pending embedding jobs")
DLQ_SIZE    = Gauge("memory_dlq_size",             "DLQ job count")
OUTBOX_PEND = Gauge("memory_outbox_pending_total", "Outbox PENDING rows")
RETRY_PEND  = Gauge("memory_retry_pending_total",  "Retry table PENDING rows")
GRPC_ERRORS = Counter("memory_grpc_channel_errors_total", "gRPC channel errors")
```

---

## 15. Recovery Runbook

### 15.1 Redis Cache Lost
1. Cache reads return misses → fall back to Cassandra automatically.
2. `recent:{conversation_id}` missing → query last 20 rows from `conversation_recent_messages` in Cassandra. Repopulate Redis. **No Kafka replay needed.**
3. Snapshots: read from `conversation_snapshots`. Summaries: read from `conversation_summaries`.
4. Run bulk cache builder script to sweep all active snapshots and repopulate Redis in bulk.

### 15.2 Worker Crashed Mid-Job
1. Cleanup Worker (`CleanupWorker`) reclaims stale `PROCESSING` outbox rows after `OUTBOX_STALE_PROCESSING_MINUTES` minutes.
2. All worker consumers check idempotency via Cassandra before processing. Replays are safe.

### 15.3 Full Service Downtime (6+ Hours)
```text
1. Restart the service.
2. Kafka consumer resumes from last committed offset.
3. Idempotency table (processed_events, 7-day TTL) skips already-committed events.
4. Outbox Reaper reclaims any stale PROCESSING rows.
5. Cache Hydration fills Redis from Cassandra on demand.
6. Milvus rebuild: scan user_facts → regenerate embeddings via LLM Service → bulk insert.
```

### 15.4 Milvus Data Loss
1. Read all `user_facts` from Cassandra.
2. Batch embed using LLM gRPC pool.
3. Bulk insert into Milvus (`MILVUS_BULK_INSERT_BATCH_SIZE` records at a time).
4. All `fact_id` and metadata are preserved in Cassandra — vectors are reproducible.

---

## 16. State Machine Reference

```text
ACTIVE
  │── (threshold: message_count % N == 0) ──► SUMMARY_PENDING
  │                                                 │
  │                          (outbox claimed) ──► SUMMARIZING
  │                                                 │
  │                          (LLM complete)  ──► FACT_PENDING
  │                                                 │
  │                          (fact worker)   ──► EXTRACTING_FACTS
  │                                                 │
  │                          (facts saved)   ──► EMBEDDING_PENDING
  │                                                 │
  │                          (bulk insert)   ──► READY ──► ACTIVE
  │
  └── (5 consecutive failures) ──► FAILED ──► retry_jobs (DLQ)
```

---

## 17. Fact Merge Policy

When a new fact is extracted that may contradict or supersede an existing one, the Fact Worker applies the following merge policy:

| Scenario | Policy | Implementation |
| :--- | :--- | :--- |
| New fact similarity < `FACT_MERGE_SIMILARITY_THRESHOLD` | **Insert as new** | Create a new `fact_id` row |
| Similarity ≥ threshold, new importance ≥ existing | **Supersede** | Soft-delete old, insert new (increment `fact_version`) |
| Similarity ≥ threshold, new importance < existing | **Ignore** | Discard incoming fact |
| Exact statement match (hash equality) | **Skip** | Idempotent, no write |

**Supersede Flow:**
```text
[New Fact Extracted]
       │
       ▼
[Vector Search in Milvus: user_id partition, top_1]
       │
       ├── similarity < threshold → INSERT new fact
       │
       └── similarity ≥ threshold
              │
              ├── new importance ≥ old → DELETE old (Milvus) + Cassandra soft-delete
              │                         INSERT new (Milvus + Cassandra, fact_version++)
              │
              └── new importance < old → DISCARD (no write)
```

All fact versions are tracked via `fact_version` in the `user_facts` Cassandra table. Older versions are not physically deleted immediately — a background cleanup sweep purges them on a schedule.

---

## 18. Summary Merge Algorithm

When the conversation reaches the summary threshold (every N messages):

**What the LLM receives:**
```json
{
  "previous_summary": "<Summary V{n} text from Cassandra>",
  "new_messages": [ ... last 20 messages ... ],
  "instructions": "Integrate the new messages into an updated summary. Preserve key facts from the previous summary. Be concise."
}
```

**What is stored:**
```text
Cassandra: conversation_summaries.summary_text = <new text>
Cassandra: conversation_summaries.summary_version = n+1
Cassandra: conversation_snapshots.summary_version = n+1
Redis: DELETE summary:{conversation_id}  (force cache hydration on next read)
```

Summary V(n+1) is built from the **previous summary plus new messages only** — not the entire conversation history. This keeps the LLM prompt bounded and predictable regardless of conversation length.

---

## 19. Graph Service Timeout & Fallback

The `RetrievalService` calls the Graph Service to fetch ancestor summaries. To prevent a Graph Service outage from blocking context assembly:

```python
GRAPH_SERVICE_TIMEOUT_MS: int = 200  # Configurable in settings

async def get_parent_summaries(self, conversation_id: str) -> list[dict]:
    try:
        async with asyncio.timeout(self.timeout_seconds):
            return await self.graph_client.get_ancestors(conversation_id)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning(
            f"Graph Service unavailable for {conversation_id}: {e}. "
            f"Falling back to current summary only."
        )
        return []  # Graceful degradation: only current summary is used
```

The context response still returns the current summary and recent messages. Parent summaries are omitted with a `parent_summaries_available: false` flag in the response metadata.

## 20. Production Hardened Internal LLM Engine

We have hardened the internal LLM Engine to support high-throughput, battle-tested production operations with the following design details:

### 20.1 Connection Pooling
- Built on top of `AsyncGroq` using a shared, pre-configured `httpx.AsyncClient` session.
- Configured connection limits via `httpx.Limits`:
  - `LLM_POOL_MAX_CONNECTIONS`: Defaults to `50` concurrent connections.
  - `LLM_POOL_MAX_KEEPALIVE_CONNECTIONS`: Defaults to `10` idle connections.
- Retains connection reusability across all generation and health-check requests, drastically reducing TCP handshake overhead.

### 20.2 Rate Limiting / Concurrency Limiter
- Implemented client-side concurrency throttling inside `LLMManager` using `asyncio.Semaphore`.
- Enforces `LLM_MAX_CONCURRENT_REQUESTS` (default `10`).
- Prevents upstream rate limits (429 Too Many Requests) and resource exhaustion by throttling high-concurrency spikes locally without thread blockage.

### 20.3 Request Tracing
- Integrates with the async-safe structured log context via `ContextVar`.
- HTTP endpoints trace requests using a middleware that checks/generates `X-Trace-ID` and injects it into logging outputs.
- Inbound gRPC requests capture trace keys from gRPC invocation metadata.
- Background asynchronous workers (`SummaryWorker`, `FactWorker`) inject active `conversation_id`, `version`, and `trace_id` headers into thread logging scopes.

### 20.4 Observability & Metrics
Exposes three central Prometheus metrics:
* `memory_llm_requests_total`: Counter for total generation calls, labeled by `provider`, `model`, `action`, and completion `status`.
* `memory_llm_latency_seconds`: Histogram measuring execution duration.
* `memory_llm_tokens_total`: Counter tracking estimated prompt and completion tokens.

---
**End of Low-Level Design — v4.0 (Production-Hardened)**
