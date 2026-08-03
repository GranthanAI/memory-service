"""
app/workers/outbox_worker.py

Outbox worker daemon for reliable task publishing to Kafka.
Uses Cassandra LWT to claim PENDING jobs, publishes to Kafka, and deletes completed rows.
"""

import asyncio
import json
import logging
from typing import Any

from app.core.config import settings
from app.repositories.cassandra_repository import CassandraRepository

logger = logging.getLogger("memory_service.outbox_worker")


class OutboxDaemonWorker:
    """
    Polls outbox_jobs for PENDING rows, atomically claims them using
    Cassandra LWT (Lightweight Transactions) via CassandraRepository
    to prevent duplicate Kafka publishing by concurrent worker instances.

    Flow: PENDING -> (LWT) -> PROCESSING -> Kafka Publish -> DELETE
    """

    def __init__(self, cassandra_session, producer):
        self.session = cassandra_session
        self.producer = producer
        self.is_running = False
        self._poll_interval = settings.OUTBOX_POLL_INTERVAL_MS / 1000.0
        self.cassandra_repo = CassandraRepository(self.session)

    def _prepare_statements(self):
        """No-op since prepared statements are encapsulated inside CassandraRepository."""
        pass

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
        rows = self.cassandra_repo.get_pending_outbox_jobs(limit=settings.OUTBOX_BATCH_SIZE)

        for row in rows:
            # LWT Claim — only one worker wins
            applied = self.cassandra_repo.claim_outbox_job(row)

            if not applied:
                # Another worker claimed this row first — skip
                continue

            try:
                payload = json.loads(row["payload"])
                await self.producer.publish_task(
                    topic=row["topic"],
                    conversation_id=row["conversation_id"],
                    payload=payload
                )
                # Successfully published — delete the outbox row
                self.cassandra_repo.delete_outbox_job(
                    status="PROCESSING",
                    created_at=row["created_at"],
                    job_id=row["job_id"]
                )
            except Exception as e:
                logger.error(f"Outbox job {row['job_id']} publish failed: {e}")
                # Update attempt count and last error, keep row for reaper
                self.cassandra_repo.fail_outbox_job(row, str(e))

    async def stop(self) -> None:
        self.is_running = False
        logger.info("Outbox Daemon worker stopped.")
