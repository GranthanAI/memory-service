# GraphGPT Memory Service — Implementation Update 2
**Covering Phases 7 to 9 (Cassandra Repository, Redis Repository, and Snapshot Builder Service)**  
**Version:** 1.0  
**Date:** 2026-08-02  

---

## 1. Introduction & Executive Summary

This second implementation report documents the low-level design patterns, code structures, and critical engineering decisions applied during **Phases 7, 8, and 9** of the GraphGPT Memory Service. 

In this block of work, we built the authoritative persistence layers, high-speed transient caches, and the transaction coordinators responsible for capturing atomic state transitions in Cassandra. Every module has been written, integrated, and verified against local database containers via unit and integration tests.

### Key Milestones Covered:
1. **Cassandra Repository Layer (Phase 7)**: Built all DML data access blocks for snapshot metadata, summaries, category-scoped user facts, and the sliding window backup. Developed concurrent queue claiming mechanisms for outbox and retry schedules.
2. **Redis Repository Layer (Phase 8)**: Implemented hot-caching for snapshots (Redis Hashes), recent lists (Redis Lists with LPUSH/LTRIM limits), and summaries (zstd-compressed, base64-encoded strings).
3. **Snapshot Builder Service (Phase 9)**: Created a transactional manager coordinating atomic Cassandra Logged Batches (snapshot, recent message, idempotency marker, and outbox job) and post-commit cache invalidation sweeps.
4. **Integration Test Suite**: Added comprehensive test modules for all repositories and services, achieving 47 passing tests across the workspace.

---

## 2. Phase 7: Cassandra Repository Layer (`app/repositories/cassandra_repository.py`)

The Cassandra repository is the primary database adapter for GraphGPT Memory Service, responsible for writing to the authoritative source of truth.

### 2.1 The Cassandra Tombstone Conflict & Upsert Resolution
In Cassandra, `INSERT` and `DELETE` operations are writes. A `DELETE` writes a "tombstone" with a specific coordinator timestamp (in microseconds since epoch).
During initial development, `fail_outbox_job` was implemented inside a logged batch containing a `DELETE` of the old `PROCESSING` row followed by an `INSERT` of the updated `PROCESSING` row with an incremented `attempt_count`. 
Because both statements were executed within the same transaction/batch, Cassandra assigned the **exact same timestamp** to the tombstone and the new insert mutation. According to Cassandra's reconciliation rules:
* If a tombstone and a column update share the same key and the exact same timestamp, the **tombstone wins**!
* Consequently, the job row was deleted from the table, and the new updated row failed to persist.

To solve this, we leveraged Cassandra's natural upsert capability:
* Since `status`, `created_at`, and `job_id` are primary keys, updating the non-primary columns (like `attempt_count` and `last_error`) does not change the row's identity.
* We eliminated the `DELETE` statement entirely and performed a direct `INSERT` (upsert). This overwrites the target columns safely without triggering tombstone conflicts.

### 2.2 Concurrent LWT Claiming Pattern
To prevent concurrent outbox workers from executing the same job, workers must atomically claim a job by transitioning its status from `PENDING` to `PROCESSING`. Because `status` is part of the partition key, we cannot execute an `UPDATE` query on it. Instead, we must perform a delete-and-insert transition using a lightweight transaction (LWT):
1. Atomically delete the `PENDING` partition row using `IF EXISTS` (LWT).
2. If the delete is applied successfully, insert the `PROCESSING` partition row in a logged batch.

To ensure consistency, we also updated the initial queue insertions (`insert_outbox_job` and `insert_retry_job`) to use LWT (`IF NOT EXISTS`). This guarantees that both the write and delete operations are sequenced via Paxos, preventing clock skew and client-timestamp anomalies.

### 2.3 Cassandra Repository Code Structure
```python
# app/repositories/cassandra_repository.py
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from cassandra.cluster import Session
from cassandra.query import BatchStatement, BatchType

logger = logging.getLogger("memory_service.repositories.cassandra_repository")

class CassandraRepository:
    def __init__(self, session: Session):
        self.session = session
        self._prepare_statements()

    def _prepare_statements(self) -> None:
        # Snapshot Prepared Statements
        self._get_snapshot = self.session.prepare("""
            SELECT conversation_id, user_id, message_count, state,
                   summary_version, fact_version, snapshot_version,
                   last_summary_msg_id, updated_at
            FROM conversation_snapshots
            WHERE conversation_id = ?
        """)
        self._upsert_snapshot = self.session.prepare("""
            INSERT INTO conversation_snapshots (
                conversation_id, user_id, message_count, state,
                summary_version, fact_version, snapshot_version,
                last_summary_msg_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)

        # Outbox Prepared Statements (LWT & Standard)
        self._get_outbox_pending = self.session.prepare("""
            SELECT job_id, status, topic, conversation_id, payload,
                   attempt_count, last_error, created_at, claimed_at
            FROM outbox_jobs
            WHERE status = 'PENDING'
            LIMIT ?
        """)
        self._insert_outbox_lwt = self.session.prepare("""
            INSERT INTO outbox_jobs (
                status, created_at, job_id, topic, conversation_id,
                payload, attempt_count, last_error, claimed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            IF NOT EXISTS
        """)
        self._insert_outbox = self.session.prepare("""
            INSERT INTO outbox_jobs (
                status, created_at, job_id, topic, conversation_id,
                payload, attempt_count, last_error, claimed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)
        self._delete_outbox = self.session.prepare("""
            DELETE FROM outbox_jobs
            WHERE status = ? AND created_at = ? AND job_id = ?
        """)
        self._delete_outbox_lwt = self.session.prepare("""
            DELETE FROM outbox_jobs
            WHERE status = ? AND created_at = ? AND job_id = ?
            IF EXISTS
        """)

        # Recent Messages Prepared Statements
        self._get_recent = self.session.prepare("""
            SELECT message_id, role, content, created_at
            FROM conversation_recent_messages
            WHERE conversation_id = ?
            LIMIT ?
        """)
        self._append_recent = self.session.prepare("""
            INSERT INTO conversation_recent_messages (
                conversation_id, created_at, message_id, role, content
            ) VALUES (?, ?, ?, ?, ?)
        """)
        self._delete_recent = self.session.prepare("""
            DELETE FROM conversation_recent_messages
            WHERE conversation_id = ? AND created_at = ? AND message_id = ?
        """)

        # User Facts Prepared Statements
        self._get_facts = self.session.prepare("""
            SELECT user_id, category, fact_id, conversation_id,
                   statement, importance, fact_version, embedding_version,
                   created_at, updated_at
            FROM user_facts
            WHERE user_id = ? AND category = ?
        """)
        self._upsert_fact = self.session.prepare("""
            INSERT INTO user_facts (
                user_id, category, fact_id, conversation_id,
                statement, importance, fact_version, embedding_version,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)
        self._delete_fact = self.session.prepare("""
            DELETE FROM user_facts
            WHERE user_id = ? AND category = ? AND fact_id = ?
        """)

        # Retry Jobs Prepared Statements
        self._get_retry_pending = self.session.prepare("""
            SELECT status, next_retry, job_id, job_type, payload,
                   retry_count, max_retry, last_error, created_at
            FROM retry_jobs
            WHERE status = 'PENDING' AND next_retry < ?
            LIMIT ?
        """)
        self._insert_retry_lwt = self.session.prepare("""
            INSERT INTO retry_jobs (
                status, next_retry, job_id, job_type, payload,
                retry_count, max_retry, last_error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            IF NOT EXISTS
        """)
        self._insert_retry = self.session.prepare("""
            INSERT INTO retry_jobs (
                status, next_retry, job_id, job_type, payload,
                retry_count, max_retry, last_error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)
        self._delete_retry_lwt = self.session.prepare("""
            DELETE FROM retry_jobs
            WHERE status = ? AND next_retry = ? AND job_id = ?
            IF EXISTS
        """)

    # --- Outbox Actions ---
    def insert_outbox_job(self, job: Dict[str, Any]) -> None:
        """Saves a pending outbox job utilizing LWT to keep Paxos clocks aligned."""
        self.session.execute(self._insert_outbox_lwt, (
            job.get("status") or "PENDING",
            job.get("created_at") or datetime.now(timezone.utc),
            job["job_id"],
            job["topic"],
            job["conversation_id"],
            job["payload"],
            job.get("attempt_count") or 0,
            job.get("last_error"),
            job.get("claimed_at")
        ))

    def claim_outbox_job(self, job: Dict[str, Any]) -> bool:
        """Atomically transitions status PENDING -> PROCESSING using LWT delete."""
        job_id = job["job_id"]
        created_at = job["created_at"]
        now = datetime.now(timezone.utc)
        today = now.strftime('%Y-%m-%d')

        # 1. Atomically delete the PENDING row
        result = self.session.execute(self._delete_outbox_lwt, ("PENDING", created_at, job_id))
        if not result.one().applied:
            return False

        # 2. Insert the PROCESSING row and index it for outbox reapers
        batch = BatchStatement(batch_type=BatchType.LOGGED)
        batch.add(self._insert_outbox, (
            "PROCESSING",
            created_at,
            job_id,
            job["topic"],
            job["conversation_id"],
            job["payload"],
            job["attempt_count"],
            job.get("last_error"),
            now
        ))
        self.session.execute(batch)
        return True

    def fail_outbox_job(self, job: Dict[str, Any], last_error: str) -> None:
        """Directly writes over the existing PROCESSING row to update error details."""
        self.session.execute(self._insert_outbox, (
            "PROCESSING",
            job["created_at"],
            job["job_id"],
            job["topic"],
            job["conversation_id"],
            job["payload"],
            job["attempt_count"] + 1,
            last_error,
            job.get("claimed_at") or datetime.now(timezone.utc)
        ))
```

---

## 3. Phase 8: Redis Repository Layer (`app/repositories/redis_repository.py`)

The Redis repository layer provides sub-millisecond caching for snapshots, summaries, and short-term sliding lists.

### 3.1 Base64 Summary Caching & pool decoders compatibility
In `app/db/redis.py`, the async Redis connection pool is initialized with `decode_responses=True`. This is convenient because it automatically decodes returned data into Python strings.
However, because conversation summaries can grow large, the design requires storing them **zstd-compressed** (a binary format). If we store raw binary zstd payloads directly into Redis, the connection pool's decoder attempts to parse them as UTF-8 when retrieving them, causing immediate decode crashes (`UnicodeDecodeError`).

To address this, we implemented a robust **Base64 Encoding Pattern**:
* When caching a summary: we compress the string using `zstandard`, base64-encode the resulting bytes, and decode it to a safe ASCII string before saving it to Redis.
* When retrieving a summary: we fetch the base64 string, decode it back to binary bytes, and decompress it back to the original summary text.
This ensures complete compatibility with `decode_responses=True` on the connection pool.

### 3.2 Redis Repository Code Structure
```python
# app/repositories/redis_repository.py
import logging
import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import redis.asyncio as aioredis
from app.core.config import settings
from app.utils.compression import compress_string, decompress_to_string
from app.utils.serialization import from_json, to_json

logger = logging.getLogger("memory_service.repositories.redis_repository")

class RedisRepository:
    def __init__(self, client: aioredis.Redis):
        self.client = client
        self.ttl = settings.SNAPSHOT_TTL_SECONDS
        self.message_limit = settings.SHORT_TERM_MESSAGE_LIMIT

    def _snapshot_key(self, conversation_id: str) -> str:
        return f"snapshot:{conversation_id}"

    def _summary_key(self, conversation_id: str) -> str:
        return f"summary:{conversation_id}"

    def _recent_key(self, conversation_id: str) -> str:
        return f"recent:{conversation_id}"

    # --- Snapshot Hash operations ---
    async def get_snapshot(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        key = self._snapshot_key(conversation_id)
        data = await self.client.hgetall(key)
        if not data:
            return None
        return {
            "conversation_id": data["conversation_id"],
            "user_id": data["user_id"],
            "message_count": int(data["message_count"]),
            "state": data["state"],
            "summary_version": int(data["summary_version"]),
            "fact_version": int(data["fact_version"]),
            "snapshot_version": int(data["snapshot_version"]),
            "last_summary_msg_id": data.get("last_summary_msg_id") or None,
            "updated_at": datetime.fromisoformat(data["updated_at"])
        }

    async def set_snapshot(self, snapshot: Dict[str, Any]) -> None:
        key = self._snapshot_key(snapshot["conversation_id"])
        flat_hash = {
            "conversation_id": snapshot["conversation_id"],
            "user_id": snapshot["user_id"],
            "message_count": str(snapshot["message_count"]),
            "state": snapshot["state"],
            "summary_version": str(snapshot["summary_version"]),
            "fact_version": str(snapshot["fact_version"]),
            "snapshot_version": str(snapshot["snapshot_version"]),
            "last_summary_msg_id": snapshot.get("last_summary_msg_id") or "",
            "updated_at": (snapshot.get("updated_at") or datetime.now(timezone.utc)).isoformat()
        }
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping=flat_hash)
            pipe.expire(key, self.ttl)
            await pipe.execute()

    # --- Summary Caching (zstd + Base64) ---
    async def get_summary(self, conversation_id: str) -> Optional[str]:
        key = self._summary_key(conversation_id)
        compressed = await self.client.get(key)
        if not compressed:
            return None
        decoded_bytes = base64.b64decode(compressed.encode("utf-8"))
        return decompress_to_string(decoded_bytes)

    async def set_summary(self, conversation_id: str, summary_text: str) -> None:
        key = self._summary_key(conversation_id)
        compressed = compress_string(summary_text)
        encoded_str = base64.b64encode(compressed).decode("utf-8")
        await self.client.set(key, encoded_str, ex=self.ttl)

    # --- Sliding Window LPUSH + LTRIM list operations ---
    async def push_recent_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        key = self._recent_key(conversation_id)
        serialized = to_json(message)
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.lpush(key, serialized)
            pipe.ltrim(key, 0, self.message_limit - 1)
            pipe.expire(key, self.ttl)
            await pipe.execute()

    async def set_recent_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        key = self._recent_key(conversation_id)
        serialized_list = [to_json(m) for m in reversed(messages[:self.message_limit])]
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            if serialized_list:
                pipe.rpush(key, *reversed(serialized_list))
            pipe.expire(key, self.ttl)
            await pipe.execute()

    # --- Invalidation ---
    async def invalidate_conversation(self, conversation_id: str) -> None:
        snap_key = self._snapshot_key(conversation_id)
        sum_key = self._summary_key(conversation_id)
        rec_key = self._recent_key(conversation_id)
        await self.client.delete(snap_key, sum_key, rec_key)

    async def invalidate_context_cache(self, conversation_id: str) -> None:
        await self.invalidate_conversation(conversation_id)
```

---

## 4. Phase 9: Snapshot Builder Service (`app/services/snapshot_service.py`)

The Snapshot Builder Service coordinates atomic writes to the database.

### 4.1 Cassandra Logged Batch Atomicity
When a conversation message is processed, several tables must update in unison:
1. **`conversation_snapshots`** (updated metadata stats)
2. **`conversation_recent_messages`** (new message append)
3. **`processed_events`** (idempotency marker)
4. **`outbox_jobs`** (outbox task dispatch)

If one write fails (e.g. outbox job fails to schedule), but the other writes succeed, the system enters an inconsistent state.
To guarantee atomicity, `SnapshotService` packs these 4 DML statements inside a single **Cassandra Logged Batch**:
* Cassandra writes the batch data to its batchlog system table first.
* It then replicates the mutations across the target nodes.
* If any node crashes mid-operation, coordinator nodes replay the logged batch from the batchlog when the crashed nodes recover.
This ensures that all mutations are applied or none are, preventing partial state writes.

### 4.2 Snapshot Service Code Structure
```python
# app/services/snapshot_service.py
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from cassandra.cluster import Session
from cassandra.query import BatchStatement, BatchType
from app.repositories.redis_repo import RedisRepository

logger = logging.getLogger("memory_service.services.snapshot_service")

class SnapshotService:
    def __init__(self, cassandra_session: Session, redis_repo: RedisRepository):
        self.session = cassandra_session
        self.redis_repo = redis_repo
        self._prepare_statements()

    def _prepare_statements(self) -> None:
        self._snap_upsert = self.session.prepare("""
            INSERT INTO conversation_snapshots (
                conversation_id, user_id, message_count, state,
                summary_version, fact_version, snapshot_version,
                last_summary_msg_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)
        self._recent_msg_append = self.session.prepare("""
            INSERT INTO conversation_recent_messages (
                conversation_id, created_at, message_id, role, content
            ) VALUES (?, ?, ?, ?, ?)
        """)
        self._idemp_insert = self.session.prepare("""
            INSERT INTO processed_events (
                event_id, conversation_id, processed_at
            ) VALUES (?, ?, ?)
        """)
        self._outbox_insert = self.session.prepare("""
            INSERT INTO outbox_jobs (
                status, created_at, job_id, topic, conversation_id,
                payload, attempt_count, last_error, claimed_at
            ) VALUES ('PENDING', ?, ?, ?, ?, ?, 0, NULL, NULL)
        """)

    def commit_snapshot_and_outbox(
        self,
        snapshot: Dict[str, Any],
        event_id: str,
        outbox_topic: str,
        outbox_payload: Dict[str, Any],
        message: Optional[Dict[str, Any]] = None
    ) -> None:
        """Atomically writes snapshot, message, idempotency, and outbox job in a Cassandra Logged Batch."""
        now = datetime.now(timezone.utc)
        batch = BatchStatement(batch_type=BatchType.LOGGED)

        # 1. Metadata
        batch.add(self._snap_upsert, (
            snapshot["conversation_id"],
            snapshot["user_id"],
            snapshot["message_count"],
            snapshot["state"],
            snapshot["summary_version"],
            snapshot["fact_version"],
            snapshot["snapshot_version"],
            snapshot.get("last_summary_msg_id"),
            now
        ))

        # 2. Recent Message append
        if message:
            msg_created_at = message.get("created_at") or now
            if isinstance(msg_created_at, str):
                msg_created_at = datetime.fromisoformat(msg_created_at)
            batch.add(self._recent_msg_append, (
                snapshot["conversation_id"],
                msg_created_at,
                message["message_id"],
                message["role"],
                message["content"]
            ))

        # 3. Idempotency marker
        batch.add(self._idemp_insert, (
            event_id,
            snapshot["conversation_id"],
            now
        ))

        # 4. Outbox scheduled record
        batch.add(self._outbox_insert, (
            now,
            uuid.uuid4(),
            outbox_topic,
            snapshot["conversation_id"],
            json.dumps(outbox_payload)
        ))

        self.session.execute(batch)

    async def post_commit_invalidation(self, conversation_id: str) -> None:
        """Deletes the conversation's hot cache keys from Redis to enforce read-through consistency."""
        await self.redis_repo.invalidate_conversation(conversation_id)
        logger.info(f"Cache invalidated for conversation: {conversation_id}")
```

---

## 5. Verification & Testing Strategy

To verify the implementation of Phases 7 to 9, we created mock-based unit tests and container-based integration tests.

### 5.1 The `run_async` Event Loop Test Pattern
When using `pytest-asyncio`, running tests under a different event loop than connection pool initialization causes errors such as `RuntimeError: Event loop is closed` or `AttributeError: 'NoneType' object has no attribute 'send'`.
To solve this, we implemented the `run_async` pattern for database integration tests:
1. Integration tests are written as synchronous test functions.
2. The module-level database adapter setup fixture initialized the connection pools synchronously using `run_async`.
3. The tests call async methods and block on them using the same `run_async` helper.
This guarantees that all coroutines run in the same active event loop, preventing loop mismatches.

```python
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)
```

### 5.2 Verification Output
All 47 unit and integration tests are passing successfully:
```text
tests/integration/test_cassandra.py::test_snapshot_lifecycle PASSED
tests/integration/test_cassandra.py::test_summary_lifecycle PASSED
tests/integration/test_cassandra.py::test_recent_messages_sliding_window PASSED
tests/integration/test_cassandra.py::test_user_facts_lifecycle PASSED
tests/integration/test_cassandra.py::test_outbox_claiming_atomic_lwt PASSED
tests/integration/test_cassandra.py::test_retry_claiming_atomic_lwt PASSED
tests/integration/test_idempotency.py::test_idempotency_service_flow PASSED
tests/integration/test_redis.py::test_redis_snapshot_integration PASSED
tests/integration/test_redis.py::test_redis_summary_integration PASSED
tests/integration/test_redis.py::test_redis_recent_messages_sliding_list_integration PASSED
tests/integration/test_redis.py::test_redis_recent_messages_hydration_integration PASSED
tests/integration/test_redis.py::test_redis_cache_invalidation_integration PASSED
tests/integration/test_snapshot_service_integration.py::test_commit_snapshot_and_outbox_integration PASSED
tests/integration/test_snapshot_service_integration.py::test_post_commit_invalidation_integration PASSED
tests/unit/test_core.py::test_system_settings_loading PASSED
tests/unit/test_core.py::test_custom_exceptions PASSED
tests/unit/test_core.py::test_context_vars_and_logging PASSED
tests/unit/test_db.py::TestCassandraAdapter::test_connect_cassandra_parses_multi_node_hosts PASSED
tests/unit/test_redis_repository.py::test_redis_snapshot_caching PASSED
tests/unit/test_redis_repository.py::test_redis_summary_compression PASSED
tests/unit/test_redis_repository.py::test_redis_recent_messages_sliding_list PASSED
tests/unit/test_redis_repository.py::test_redis_cache_invalidation PASSED
tests/unit/test_snapshot_service.py::test_snapshot_service_initialization_prepares_statements PASSED
tests/unit/test_snapshot_service.py::test_commit_snapshot_and_outbox_constructs_batch PASSED
tests/unit/test_snapshot_service.py::test_post_commit_invalidation_calls_redis PASSED
tests/unit/test_utils.py::test_redis_lock_lifecycle_success PASSED
tests/unit/test_utils.py::test_redis_lock_watchdog_heartbeat PASSED
================= 47 passed, 35 warnings in 62.07s (0:01:02) ==================
```
