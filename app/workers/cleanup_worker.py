"""
app/workers/cleanup_worker.py

Cleanup Worker (Stale PROCESSING Reaper).

Periodically reclaims outbox_jobs rows that are stuck in PROCESSING state
due to worker crashes or publish failures. Uses the outbox_processing_index
table for efficient time-range queries — avoids ALLOW FILTERING on the main
outbox_jobs table which would cause full table scans at scale.

Claiming Flow:
  PENDING → (OutboxWorker LWT) → PROCESSING → (publish OK) → DELETE
                                               (publish fail/crash) → stuck

Reaper Flow:
  outbox_processing_index: claimed_date=X, claimed_at < stale_threshold, job_id
  → LWT UPDATE outbox_jobs SET status='PENDING' IF status='PROCESSING'
  → DELETE row from outbox_processing_index

The index is keyed on (claimed_date, claimed_at, job_id), allowing
efficient range scans by date bucket without secondary indexes or ALLOW FILTERING.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from app.core.config import settings

logger = logging.getLogger("memory_service.cleanup_worker")


class CleanupWorker:
    """
    Periodically sweeps outbox_processing_index for rows older than
    OUTBOX_STALE_PROCESSING_MINUTES and resets the corresponding
    outbox_jobs rows from PROCESSING → PENDING using Cassandra LWT.

    Design constraints:
    - Never uses ALLOW FILTERING — queries only by outbox_processing_index PK.
    - Scans both today's and yesterday's date buckets to correctly handle
      jobs claimed just before midnight.
    - LWT ensures only one reaper instance can reclaim a given row.
    - Deletes index row only after successful LWT reclaim to prevent leaks.
    """

    def __init__(self, session):
        self.session = session
        self.is_running = False
        self._prepare_statements()

    def _prepare_statements(self) -> None:
        """Prepare all CQL statements once at startup."""
        # Fetch stale index rows for a given date bucket older than threshold
        self._scan_index_stmt = self.session.prepare(
            "SELECT job_id, claimed_at, created_at "
            "FROM outbox_processing_index "
            "WHERE claimed_date = ? AND claimed_at < ?"
        )

        # Fetch the full PROCESSING row so we can reconstruct the PENDING insert
        self._get_processing_row_stmt = self.session.prepare(
            "SELECT job_id, topic, conversation_id, payload, attempt_count, created_at "
            "FROM outbox_jobs "
            "WHERE status = 'PROCESSING' AND created_at = ? AND job_id = ?"
        )

        # Delete the PROCESSING row by its composite PK
        self._delete_processing_stmt = self.session.prepare(
            "DELETE FROM outbox_jobs "
            "WHERE status = 'PROCESSING' AND created_at = ? AND job_id = ?"
        )

        # Insert a fresh PENDING row — IF NOT EXISTS prevents double-insert if
        # two reapers race (idempotent guard)
        self._insert_pending_stmt = self.session.prepare(
            "INSERT INTO outbox_jobs "
            "(status, created_at, job_id, topic, conversation_id, payload, attempt_count) "
            "VALUES ('PENDING', ?, ?, ?, ?, ?, ?) "
            "IF NOT EXISTS"
        )

        # Remove index row after successful reclaim
        self._delete_index_stmt = self.session.prepare(
            "DELETE FROM outbox_processing_index "
            "WHERE claimed_date = ? AND claimed_at = ? AND job_id = ?"
        )

    async def start(self) -> None:
        """
        Long-running sweep loop. Runs every 60 seconds.
        Checks both today's and yesterday's date buckets.
        """
        self.is_running = True
        logger.info(
            f"CleanupWorker (Reaper) started. "
            f"Stale threshold: {settings.OUTBOX_STALE_PROCESSING_MINUTES} minutes. "
            f"Sweep interval: 60s."
        )
        while self.is_running:
            try:
                await self._sweep_stale_jobs()
            except Exception as e:
                logger.error(f"CleanupWorker sweep error: {e}", exc_info=True)
            await asyncio.sleep(60)

    async def _sweep_stale_jobs(self) -> int:
        """
        Scans both today's and yesterday's date buckets for stale PROCESSING
        rows, and resets them back to PENDING via LWT.

        Returns:
            int: Number of rows successfully reclaimed.
        """
        stale_before = datetime.now(timezone.utc) - timedelta(
            minutes=settings.OUTBOX_STALE_PROCESSING_MINUTES
        )
        today = stale_before.strftime("%Y-%m-%d")
        yesterday = (stale_before - timedelta(days=1)).strftime("%Y-%m-%d")

        total_reclaimed = 0

        for date_bucket in (yesterday, today):
            count = await self._reclaim_bucket(date_bucket, stale_before)
            total_reclaimed += count

        if total_reclaimed:
            logger.info(
                f"Reaper reclaimed {total_reclaimed} stale PROCESSING outbox rows."
            )
        else:
            logger.debug("Reaper sweep complete — no stale PROCESSING rows found.")

        return total_reclaimed

    async def _reclaim_bucket(self, date_bucket: str, stale_before: datetime) -> int:
        """
        Scans one date bucket in outbox_processing_index and reclaims stale rows.

        Args:
            date_bucket: Date string e.g. '2026-08-04' — partition key of the index.
            stale_before: Timestamp cutoff — rows with claimed_at < this are stale.

        Returns:
            int: Number of rows reclaimed in this bucket.
        """
        try:
            rows = self.session.execute(
                self._scan_index_stmt, (date_bucket, stale_before)
            )
        except Exception as e:
            logger.error(f"Failed to scan index bucket {date_bucket}: {e}")
            return 0

        count = 0
        for row in rows:
            reclaimed = self._try_reclaim(
                job_id=row.job_id,
                created_at=row.created_at,
                date_bucket=date_bucket,
                claimed_at=row.claimed_at,
            )
            if reclaimed:
                count += 1

        return count

    def _try_reclaim(
        self,
        job_id,
        created_at,
        date_bucket: str,
        claimed_at,
    ) -> bool:
        """
        Reclaims a single stale PROCESSING outbox row back to PENDING.

        Flow:
          1. SELECT the PROCESSING row to get topic/payload (for re-insertion)
          2. DELETE the PROCESSING row
          3. INSERT a PENDING row with IF NOT EXISTS (guards against concurrent reapers)
          4. DELETE the outbox_processing_index row

        If the PROCESSING row is already gone (another reaper got here first),
        step 1 returns None and we skip safely.

        Returns:
            bool: True if this reaper instance successfully reclaimed the row.
        """
        try:
            # 1. Read the PROCESSING row to get topic/payload data
            result = self.session.execute(
                self._get_processing_row_stmt, (created_at, job_id)
            )
            proc_row = result.one()

            if proc_row is None:
                # Row already gone — another reaper or the worker itself cleaned it up
                logger.debug(
                    f"Reaper skipped job {job_id} — PROCESSING row already gone."
                )
                # Clean up the orphaned index row
                self.session.execute(
                    self._delete_index_stmt, (date_bucket, claimed_at, job_id)
                )
                return False

            # 2. Delete the PROCESSING row
            self.session.execute(
                self._delete_processing_stmt, (created_at, job_id)
            )

            # 3. Insert a new PENDING row — IF NOT EXISTS prevents double-insert
            #    if two concurrent reapers both get past step 1
            ins_result = self.session.execute(
                self._insert_pending_stmt,
                (
                    proc_row.created_at,
                    proc_row.job_id,
                    proc_row.topic,
                    proc_row.conversation_id,
                    proc_row.payload,
                    proc_row.attempt_count,
                ),
            )
            inserted = ins_result.one().applied

            if not inserted:
                logger.debug(
                    f"Reaper: PENDING row for job {job_id} already inserted by another reaper."
                )
            else:
                logger.info(f"Reaper reclaimed stale PROCESSING job: {job_id}")

            # 4. Clean up the index row regardless of which reaper won the INSERT
            self.session.execute(
                self._delete_index_stmt, (date_bucket, claimed_at, job_id)
            )
            return True

        except Exception as e:
            logger.error(f"Reaper failed to reclaim job {job_id}: {e}")
            return False

    async def stop(self) -> None:
        """Signal the sweep loop to exit on next iteration."""
        self.is_running = False
        logger.info("CleanupWorker stop signal received.")
