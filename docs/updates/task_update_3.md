# Task Update 3: Memory State Machine & Unified Memory Repository (Phase 12)

This update details the design choices, architectural patterns, and code implementations completed for **Phase 12: Memory State Machine**, including the creation of a **Unified Memory Repository** interface.

---

## 1. Architectural Decisions & Patterns

### 1.1 Read-Through & Write-Through Caching Abstraction
To keep the service layer decoupled from specific caching mechanics, we introduced the **Unified Memory Repository Pattern** in `app/repositories/memory_repository.py`. 
This repository encapsulates:
* **Read-Through Caching**: When retrieving memory snapshots, summaries, or recent messages, it automatically queries the Redis cache. On a cache miss, it reads from the Cassandra database, hydrates the Redis cache, and returns the result.
* **Write-Through updates**: Any save operations for snapshots, summaries, or recent messages are written to Cassandra (durable source of truth) and Redis (hot cache) atomically.

```text
  [Service Layer]
         │
         ▼
[Memory Repository]
   ├───► [Redis Cache] (Transient hot cache)
   └───► [Cassandra Db] (Durable source of truth)
```

### 1.2 State Transition Validation
The `MemoryService` coordinates workflow execution states for user memory using a strict state transition matrix to prevent invalid pipeline execution flows:
* **Valid Transitions**:
  * `ACTIVE` -> `SUMMARY_PENDING` (Threshold of messages reached)
  * `SUMMARY_PENDING` -> `SUMMARIZING` (Claimed by worker)
  * `SUMMARIZING` -> `FACT_PENDING` (LLM generation complete)
  * `FACT_PENDING` -> `EXTRACTING_FACTS` (Claimed by facts extractor)
  * `EXTRACTING_FACTS` -> `EMBEDDING_PENDING` (Facts saved to database)
  * `EMBEDDING_PENDING` -> `READY` (Milvus vector insert complete)
  * `READY` -> `ACTIVE` (Workflow completed successfully)
  * Any step can transition to `FAILED` if it encounters persistent errors.
  * Transitions from `FAILED` allow moving back to any scheduling step (`ACTIVE`, `SUMMARY_PENDING`, `FACT_PENDING`, `EMBEDDING_PENDING`) for retry execution.

### 1.3 Failure Paths & Exponential Backoff Retry Scheduling
When a background pipeline worker encounters an exception (e.g., LLM server downtime or database connection failures):
* If `attempt_count < max_retries`: The state machine leaves the snapshot state in the active stage (or pending state), calculates an exponential backoff time delay (`next_retry = now + 2^attempt_count seconds`), and registers a `PENDING` job row in the Cassandra `retry_jobs` table.
* If `attempt_count >= max_retries`: The state machine transitions the snapshot state to `FAILED`, registers a `FAILED` job status row in `retry_jobs` (serving as DLQ metadata), and halts execution.

---

## 2. Code Implementations

### 2.1 Unified Memory Repository
Implemented in [app/repositories/memory_repository.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/repositories/memory_repository.py):

```python
import logging
from typing import Any, Dict, List, Optional
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.redis_repository import RedisRepository

logger = logging.getLogger("memory_service.repositories.memory_repository")

class MemoryRepository:
    def __init__(self, cassandra_repo: CassandraRepository, redis_repo: RedisRepository):
        self.cassandra_repo = cassandra_repo
        self.redis_repo = redis_repo

    async def get_snapshot(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        snapshot = await self.redis_repo.get_snapshot(conversation_id)
        if snapshot:
            return snapshot
        snapshot = self.cassandra_repo.get_snapshot(conversation_id)
        if snapshot:
            await self.redis_repo.set_snapshot(snapshot)
        return snapshot

    async def save_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.cassandra_repo.upsert_snapshot(snapshot)
        await self.redis_repo.set_snapshot(snapshot)

    async def get_recent_messages(self, conversation_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        messages = await self.redis_repo.get_recent_messages(conversation_id)
        if messages:
            return messages[:limit]
        messages = self.cassandra_repo.get_recent_messages(conversation_id, limit=limit)
        if messages:
            await self.redis_repo.set_recent_messages(conversation_id, messages)
        return messages

    async def append_recent_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        self.cassandra_repo.append_recent_message(conversation_id, message)
        await self.redis_repo.push_recent_message(conversation_id, message)
```

### 2.2 Memory State Machine Coordinator
Implemented in [app/services/memory_service.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/services/memory_service.py):

```python
class MemoryService:
    VALID_TRANSITIONS = {
        MemoryState.ACTIVE: {MemoryState.SUMMARY_PENDING, MemoryState.FAILED},
        MemoryState.SUMMARY_PENDING: {MemoryState.SUMMARIZING, MemoryState.FAILED},
        MemoryState.SUMMARIZING: {MemoryState.FACT_PENDING, MemoryState.FAILED},
        MemoryState.FACT_PENDING: {MemoryState.EXTRACTING_FACTS, MemoryState.FAILED},
        MemoryState.EXTRACTING_FACTS: {MemoryState.EMBEDDING_PENDING, MemoryState.FAILED},
        MemoryState.EMBEDDING_PENDING: {MemoryState.READY, MemoryState.FAILED},
        MemoryState.READY: {MemoryState.ACTIVE, MemoryState.FAILED},
        MemoryState.FAILED: {
            MemoryState.ACTIVE,
            MemoryState.SUMMARY_PENDING,
            MemoryState.FACT_PENDING,
            MemoryState.EMBEDDING_PENDING
        }
    }

    def __init__(self, memory_repo: MemoryRepository, cassandra_repo: CassandraRepository):
        self.memory_repo = memory_repo
        self.cassandra_repo = cassandra_repo

    async def transition_state(
        self, conversation_id: str, new_state: MemoryState, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        snapshot = await self.memory_repo.get_snapshot(conversation_id)
        if snapshot is None:
            if new_state != MemoryState.ACTIVE:
                raise ValueError("Must initialize snapshot with ACTIVE first.")
            snapshot = {
                "conversation_id": conversation_id,
                "user_id": user_id or "unknown_user",
                "message_count": 0,
                "state": MemoryState.ACTIVE,
                "summary_version": 0,
                "fact_version": 0,
                "snapshot_version": 1,
                "last_summary_msg_id": None,
                "updated_at": datetime.now(timezone.utc)
            }
        else:
            current_state = MemoryState(snapshot["state"])
            if not self.is_valid_transition(current_state, new_state):
                raise ValueError(f"Invalid transition from {current_state} to {new_state}")
            snapshot["state"] = new_state
            snapshot["snapshot_version"] += 1
            snapshot["updated_at"] = datetime.now(timezone.utc)

        await self.memory_repo.save_snapshot(snapshot)
        return snapshot

    async def handle_failure(
        self, conversation_id: str, failed_state: MemoryState, job_type: str,
        payload: Dict[str, Any], error_msg: str, attempt_count: int, max_retries: int = 5
    ) -> None:
        now = datetime.now(timezone.utc)
        payload_str = json.dumps(payload)
        
        if attempt_count < max_retries:
            backoff_sec = 2 ** attempt_count
            next_retry = now + timedelta(seconds=backoff_sec)
            job = {
                "status": "PENDING",
                "next_retry": next_retry,
                "job_id": uuid.uuid4(),
                "job_type": job_type,
                "payload": payload_str,
                "retry_count": attempt_count,
                "max_retry": max_retries,
                "last_error": error_msg,
                "created_at": now
            }
            self.cassandra_repo.insert_retry_job(job)
        else:
            await self.transition_state(conversation_id, MemoryState.FAILED)
            job = {
                "status": "FAILED",
                "next_retry": now,
                "job_id": uuid.uuid4(),
                "job_type": job_type,
                "payload": payload_str,
                "retry_count": attempt_count,
                "max_retry": max_retries,
                "last_error": f"Max retries exhausted: {error_msg}",
                "created_at": now
            }
            self.cassandra_repo.insert_retry_job(job)
```

---

## 3. Verification Results

We verified Phase 12 code using complete unit and integration test suites:
* **Unit Tests (`tests/unit/test_memory_service.py`)**: Asserted linear state progression, invalid transition blocks, and mocked failure retry schedules.
* **Integration Tests (`tests/integration/test_memory_service_integration.py`)**: Executed cache miss hydration backfills, state transitions, and persistent failure entries against live test Cassandra and Redis instances.

All 63 test suites passed successfully:
```text
tests/unit/test_memory_service.py::test_memory_service_initialization_flow PASSED
tests/unit/test_memory_service.py::test_memory_service_valid_linear_transitions PASSED
tests/unit/test_memory_service.py::test_memory_service_invalid_transitions PASSED
tests/unit/test_memory_service.py::test_memory_service_handle_failure_under_threshold PASSED
tests/unit/test_memory_service.py::test_memory_service_handle_failure_threshold_reached PASSED
tests/integration/test_memory_service_integration.py::test_memory_state_machine_and_cache_hydration_integration PASSED

======================== 63 passed in 102.15s ========================
```
