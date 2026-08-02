"""
app/repositories/cassandra_repository.py

Repository for handling all Cassandra DML read and write operations.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cassandra.cluster import Session
from cassandra.query import BatchStatement, BatchType

logger = logging.getLogger("memory_service.repositories.cassandra_repository")


class CassandraRepository:
    """
    Handles persistence and queries across all Cassandra tables.
    Includes Snapshots, Summaries, Outbox Jobs, Retry Jobs, User Facts,
    and durable recent message windows.
    """

    def __init__(self, session: Session):
        self.session = session
        self._prepare_statements()

    def _prepare_statements(self):
        # ─── Snapshot DML ────────────────────────────────────────────────────
        self._get_snap = self.session.prepare("""
            SELECT conversation_id, user_id, message_count, state,
                   summary_version, fact_version, snapshot_version,
                   last_summary_msg_id, updated_at
            FROM conversation_snapshots
            WHERE conversation_id = ?
        """)
        self._upsert_snap = self.session.prepare("""
            INSERT INTO conversation_snapshots (
                conversation_id, user_id, message_count, state,
                summary_version, fact_version, snapshot_version,
                last_summary_msg_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)

        # ─── Summary DML ─────────────────────────────────────────────────────
        self._get_summary = self.session.prepare("""
            SELECT conversation_id, summary_text, summary_version,
                   model_name, model_version, generated_at
            FROM conversation_summaries
            WHERE conversation_id = ?
        """)
        self._upsert_summary = self.session.prepare("""
            INSERT INTO conversation_summaries (
                conversation_id, summary_text, summary_version,
                model_name, model_version, generated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """)

        # ─── Recent Messages DML ─────────────────────────────────────────────
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

        # ─── User Facts DML ──────────────────────────────────────────────────
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

        # ─── Outbox DML ──────────────────────────────────────────────────────
        self._get_outbox_pending = self.session.prepare("""
            SELECT job_id, status, topic, conversation_id, payload,
                   attempt_count, last_error, created_at, claimed_at
            FROM outbox_jobs
            WHERE status = 'PENDING'
            LIMIT ?
        """)
        self._get_outbox_job = self.session.prepare("""
            SELECT job_id, status, topic, conversation_id, payload,
                   attempt_count, last_error, created_at, claimed_at
            FROM outbox_jobs
            WHERE status = ? AND created_at = ? AND job_id = ?
        """)
        self._insert_outbox = self.session.prepare("""
            INSERT INTO outbox_jobs (
                status, created_at, job_id, topic, conversation_id,
                payload, attempt_count, last_error, claimed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)
        self._insert_outbox_lwt = self.session.prepare("""
            INSERT INTO outbox_jobs (
                status, created_at, job_id, topic, conversation_id,
                payload, attempt_count, last_error, claimed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            IF NOT EXISTS
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

        # ─── Retry DML ───────────────────────────────────────────────────────
        self._get_retry_pending = self.session.prepare("""
            SELECT status, next_retry, job_id, job_type, payload,
                   retry_count, max_retry, last_error, created_at
            FROM retry_jobs
            WHERE status = 'PENDING' AND next_retry < ?
            LIMIT ?
        """)
        self._insert_retry = self.session.prepare("""
            INSERT INTO retry_jobs (
                status, next_retry, job_id, job_type, payload,
                retry_count, max_retry, last_error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)
        self._insert_retry_lwt = self.session.prepare("""
            INSERT INTO retry_jobs (
                status, next_retry, job_id, job_type, payload,
                retry_count, max_retry, last_error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            IF NOT EXISTS
        """)
        self._delete_retry_lwt = self.session.prepare("""
            DELETE FROM retry_jobs
            WHERE status = ? AND next_retry = ? AND job_id = ?
            IF EXISTS
        """)

        # ─── Outbox Processing Index DML ─────────────────────────────────────
        self._insert_processing_index = self.session.prepare("""
            INSERT INTO outbox_processing_index (
                claimed_date, claimed_at, job_id
            ) VALUES (?, ?, ?)
        """)
        self._delete_processing_index = self.session.prepare("""
            DELETE FROM outbox_processing_index
            WHERE claimed_date = ? AND claimed_at = ? AND job_id = ?
        """)

    # ─── Snapshot Operations ─────────────────────────────────────────────────

    def get_snapshot(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Reads a conversation snapshot by ID."""
        rows = self.session.execute(self._get_snap, (conversation_id,))
        row = rows.one()
        return row._asdict() if row else None

    def upsert_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Saves a conversation snapshot metadata state."""
        self.session.execute(self._upsert_snap, (
            snapshot["conversation_id"],
            snapshot["user_id"],
            snapshot["message_count"],
            snapshot["state"],
            snapshot["summary_version"],
            snapshot["fact_version"],
            snapshot["snapshot_version"],
            snapshot.get("last_summary_msg_id"),
            snapshot.get("updated_at") or datetime.now(timezone.utc)
        ))

    # ─── Summary Operations ──────────────────────────────────────────────────

    def get_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Reads a conversation summary by ID."""
        rows = self.session.execute(self._get_summary, (conversation_id,))
        row = rows.one()
        return row._asdict() if row else None

    def upsert_summary(self, summary: Dict[str, Any]) -> None:
        """Saves a versioned summary record."""
        self.session.execute(self._upsert_summary, (
            summary["conversation_id"],
            summary["summary_text"],
            summary["summary_version"],
            summary["model_name"],
            summary["model_version"],
            summary.get("generated_at") or datetime.now(timezone.utc)
        ))

    # ─── Recent Messages Operations ──────────────────────────────────────────

    def get_recent_messages(self, conversation_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Reads the sliding message window from Cassandra, sorted DESC (most recent first)."""
        rows = self.session.execute(self._get_recent, (conversation_id, limit))
        return [row._asdict() for row in rows]

    def append_recent_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """Appends a new message to the durable recent message window in Cassandra."""
        self.session.execute(self._append_recent, (
            conversation_id,
            message.get("created_at") or datetime.now(timezone.utc),
            message["message_id"],
            message["role"],
            message["content"]
        ))

    def delete_recent_message_row(self, conversation_id: str, created_at: datetime, message_id: str) -> None:
        """Removes a specific message row (used in sliding window pruning sweeps)."""
        self.session.execute(self._delete_recent, (conversation_id, created_at, message_id))

    # ─── User Facts Operations ───────────────────────────────────────────────

    def get_facts(self, user_id: str, category: str) -> List[Dict[str, Any]]:
        """Reads all facts under a user's category partition."""
        rows = self.session.execute(self._get_facts, (user_id, category))
        return [row._asdict() for row in rows]

    def upsert_fact(self, fact: Dict[str, Any]) -> None:
        """Saves a structured long-term fact."""
        self.session.execute(self._upsert_fact, (
            fact["user_id"],
            fact["category"],
            fact["fact_id"],
            fact.get("conversation_id"),
            fact["statement"],
            fact["importance"],
            fact["fact_version"],
            fact["embedding_version"],
            fact.get("created_at") or datetime.now(timezone.utc),
            fact.get("updated_at") or datetime.now(timezone.utc)
        ))

    def delete_fact(self, user_id: str, category: str, fact_id: uuid.UUID) -> None:
        """Removes a specific user fact."""
        self.session.execute(self._delete_fact, (user_id, category, fact_id))

    # ─── Outbox Operations ───────────────────────────────────────────────────

    def get_pending_outbox_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Queries up to LIMIT outbox jobs in PENDING state."""
        rows = self.session.execute(self._get_outbox_pending, (limit,))
        return [row._asdict() for row in rows]

    def get_outbox_job(self, status: str, created_at: datetime, job_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Fetches a specific outbox job."""
        rows = self.session.execute(self._get_outbox_job, (status, created_at, job_id))
        row = rows.one()
        return row._asdict() if row else None

    def insert_outbox_job(self, job: Dict[str, Any]) -> None:
        """Creates an outbox job row using LWT to ensure Paxos timestamp consistency."""
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

    def delete_outbox_job(self, status: str, created_at: datetime, job_id: uuid.UUID) -> None:
        """Removes a processed outbox job."""
        self.session.execute(self._delete_outbox, (status, created_at, job_id))

    def claim_outbox_job(self, job: Dict[str, Any]) -> bool:
        """
        Transitions job status PENDING -> PROCESSING atomically using Cassandra LWT.
        Since status is the partition key, we must delete the PENDING row and
        insert the PROCESSING row in a single transaction if the delete is applied.
        """
        job_id = job["job_id"]
        created_at = job["created_at"]
        now = datetime.now(timezone.utc)
        today = now.strftime('%Y-%m-%d')

        # 1. Attempt to atomically delete the PENDING row using LWT
        result = self.session.execute(self._delete_outbox_lwt, ("PENDING", created_at, job_id))
        applied = result.one().applied

        if not applied:
            # Another worker claimed it first
            return False

        # 2. If deleted successfully, write the PROCESSING row and index it for clean reaping
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
        batch.add(self._insert_processing_index, (
            today,
            now,
            job_id
        ))
        self.session.execute(batch)
        return True

    def fail_outbox_job(self, job: Dict[str, Any], last_error: str) -> None:
        """
        Records failure details for a job. Since status remains PROCESSING,
        we execute a direct INSERT (upsert) to overwrite and update the attempt_count and error fields.
        This avoids the delete-then-insert tombstone timestamp ordering issue.
        """
        created_at = job["created_at"]
        job_id = job["job_id"]
        claimed_at = job.get("claimed_at") or datetime.now(timezone.utc)

        self.session.execute(self._insert_outbox, (
            "PROCESSING",
            created_at,
            job_id,
            job["topic"],
            job["conversation_id"],
            job["payload"],
            job["attempt_count"] + 1,
            last_error,
            claimed_at
        ))

    # ─── Retry Operations ────────────────────────────────────────────────────

    def get_pending_retry_jobs(self, next_retry_before: datetime, limit: int = 50) -> List[Dict[str, Any]]:
        """Queries retry jobs that are scheduled for execution."""
        rows = self.session.execute(self._get_retry_pending, (next_retry_before, limit))
        return [row._asdict() for row in rows]

    def insert_retry_job(self, job: Dict[str, Any]) -> None:
        """Saves a retry job schedule using LWT to ensure Paxos timestamp consistency."""
        self.session.execute(self._insert_retry_lwt, (
            job.get("status") or "PENDING",
            job["next_retry"],
            job["job_id"],
            job["job_type"],
            job["payload"],
            job.get("retry_count") or 0,
            job["max_retry"],
            job.get("last_error"),
            job.get("created_at") or datetime.now(timezone.utc)
        ))

    def claim_retry_job(self, job: Dict[str, Any]) -> bool:
        """
        Atomic transition of retry job PENDING -> PROCESSING.
        Deletes the PENDING row using LWT and inserts the PROCESSING row if applied.
        """
        job_id = job["job_id"]
        next_retry = job["next_retry"]
        now = datetime.now(timezone.utc)

        result = self.session.execute(self._delete_retry_lwt, ("PENDING", next_retry, job_id))
        applied = result.one().applied

        if not applied:
            return False

        # Write the processing row
        self.session.execute(self._insert_retry, (
            "PROCESSING",
            next_retry,
            job_id,
            job["job_type"],
            job["payload"],
            job["retry_count"],
            job["max_retry"],
            job.get("last_error"),
            job.get("created_at") or now
        ))
        return True

    def delete_retry_job(self, status: str, next_retry: datetime, job_id: uuid.UUID) -> None:
        """Removes a retry job completely (called when complete or moved to DLQ)."""
        self.session.execute(self._delete_retry_lwt, (status, next_retry, job_id))

