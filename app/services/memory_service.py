"""
app/services/memory_service.py

Memory Service coordinates workflow state transitions across pipeline phases
and implements error-recovery retry/DLQ scheduling.
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Set

from app.models.memory import MemoryState
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.memory_repository import MemoryRepository

logger = logging.getLogger("memory_service.services.memory_service")


class MemoryService:
    """
    Coordinates workflow execution states for user memory snapshots.
    Validates state transitions and manages execution pipeline failure retries.
    """

    # Enforce valid forward/backward pipeline transition paths
    VALID_TRANSITIONS: Dict[MemoryState, Set[MemoryState]] = {
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

    def is_valid_transition(self, current_state: MemoryState, new_state: MemoryState) -> bool:
        """
        Asserts if the target transition is allowed by the state machine matrix.
        """
        allowed = self.VALID_TRANSITIONS.get(current_state, set())
        return new_state in allowed

    async def get_or_hydrate_snapshot(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Delegates snapshot lookup and hydration to unified MemoryRepository.
        """
        return await self.memory_repo.get_snapshot(conversation_id)

    async def transition_state(
        self,
        conversation_id: str,
        new_state: MemoryState,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates and transitions snapshot state, persisting the changes to Cassandra and Redis cache.
        If transitioning from a non-existent snapshot, initializes a new snapshot in ACTIVE state.
        """
        snapshot = await self.get_or_hydrate_snapshot(conversation_id)

        if snapshot is None:
            # Initialize a new snapshot structure
            if new_state != MemoryState.ACTIVE:
                raise ValueError(
                    f"Cannot transition to state {new_state} for non-existent snapshot {conversation_id}. "
                    "Must initialize with ACTIVE first."
                )
            
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
            logger.info(f"Initializing new conversation snapshot {conversation_id} in ACTIVE state.")
        else:
            current_state = MemoryState(snapshot["state"])
            if not self.is_valid_transition(current_state, new_state):
                raise ValueError(
                    f"Invalid state transition for conversation {conversation_id}: "
                    f"cannot move from {current_state} to {new_state}."
                )

            snapshot["state"] = new_state
            snapshot["snapshot_version"] += 1
            snapshot["updated_at"] = datetime.now(timezone.utc)
            logger.info(f"Transitioning snapshot {conversation_id} from {current_state} to {new_state}.")

        # Persist mutations across both layers via unified MemoryRepository
        await self.memory_repo.save_snapshot(snapshot)
        
        return snapshot

    async def handle_failure(
        self,
        conversation_id: str,
        failed_state: MemoryState,
        job_type: str,
        payload: Dict[str, Any],
        error_msg: str,
        attempt_count: int,
        max_retries: int = 5
    ) -> None:
        """
        Handles retry job registration and state transitions when pipeline stages fail.
        Schedules retries with backoff if attempts < max_retries, otherwise marks snapshot as FAILED.
        """
        now = datetime.now(timezone.utc)
        payload_str = json.dumps(payload)

        if attempt_count < max_retries:
            # Exponential backoff retry scheduling: next_retry = now + 2^(attempt_count) seconds
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
            
            logger.warning(
                f"Scheduling retry job {job['job_id']} (attempt {attempt_count}/{max_retries}) "
                f"for conversation {conversation_id} in {backoff_sec} seconds. Error: {error_msg}"
            )
            self.cassandra_repo.insert_retry_job(job)

        else:
            # Max retries exhausted - transition state to FAILED
            logger.critical(
                f"Max retries ({max_retries}) exhausted for conversation {conversation_id} during "
                f"state {failed_state}. Transitioning snapshot to FAILED."
            )
            
            # Transition state
            await self.transition_state(conversation_id, MemoryState.FAILED)

            # Insert final FAILED/DLQ record in retry_jobs for auditor visibility
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
