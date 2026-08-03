"""
app/events/retry_scheduler.py

Retry scheduler background worker.
Periodically scans Cassandra for scheduled retry jobs, claims them,
and re-publishes to Kafka or forwards to the Dead Letter Queue (DLQ).
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta

from app.core.config import settings
from app.repositories.cassandra_repository import CassandraRepository

logger = logging.getLogger("memory_service.events.retry_scheduler")


class RetryScheduler:
    """
    Polls the retry_jobs database partition, claims scheduled tasks, and dispatches them.
    If retry thresholds are exceeded, dispatches tasks to DLQ.
    """

    def __init__(self, cassandra_session, producer, poll_interval_seconds: float = 2.0):
        self.session = cassandra_session
        self.producer = producer
        self.cassandra_repo = CassandraRepository(self.session)
        self.is_running = False
        self._poll_interval = poll_interval_seconds
        self._task = None

    async def start(self) -> None:
        """Starts the retry scheduler polling daemon."""
        self.is_running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Retry Scheduler started. Poll interval: {self._poll_interval}s")

    async def _loop(self) -> None:
        """Continuous polling loop."""
        while self.is_running:
            try:
                await self._process_retries()
            except Exception as e:
                logger.error(f"Retry Scheduler iteration failed: {e}")
            await asyncio.sleep(self._poll_interval)

    async def _process_retries(self) -> None:
        """Polls, claims, and processes pending retry entries."""
        now = datetime.now(timezone.utc)
        jobs = self.cassandra_repo.get_pending_retry_jobs(next_retry_before=now, limit=50)

        # Mapping pipeline types to target Kafka topics
        topic_mapping = {
            "summary": settings.KAFKA_SUMMARY_TOPIC,
            "fact": settings.KAFKA_FACT_TOPIC,
            "embedding": settings.KAFKA_EMBEDDING_TOPIC,
            "delete": settings.KAFKA_DELETE_TOPIC
        }

        for job in jobs:
            # Atomic claim status transition (delete PENDING, insert PROCESSING if applied)
            applied = self.cassandra_repo.claim_retry_job(job)
            if not applied:
                continue

            try:
                payload = json.loads(job["payload"])
            except Exception as e:
                logger.error(f"Failed to parse retry job {job['job_id']} payload: {e}")
                # Corrupted payload - discard processing row and log permanent failure
                self.cassandra_repo.delete_retry_job("PROCESSING", job["next_retry"], job["job_id"])
                continue

            conversation_id = payload.get("conversation_id", "unknown")
            job_type = job["job_type"]
            retry_count = job["retry_count"]
            max_retry = job["max_retry"]

            if retry_count < max_retry:
                # 1. Re-dispatch task to pipeline topic
                target_topic = topic_mapping.get(job_type)
                if not target_topic:
                    logger.error(f"Invalid job type '{job_type}' for retry {job['job_id']}. Skipping.")
                    self.cassandra_repo.delete_retry_job("PROCESSING", job["next_retry"], job["job_id"])
                    continue

                try:
                    logger.info(
                        f"Re-dispatching job {job['job_id']} to topic '{target_topic}' "
                        f"(attempt {retry_count + 1}/{max_retry})"
                    )
                    # Embed incremented count within outbound payload
                    payload["attempt_count"] = retry_count + 1

                    await self.producer.publish_task(
                        topic=target_topic,
                        conversation_id=conversation_id,
                        payload=payload
                    )
                    # Dispatch succeeded - remove job metadata
                    self.cassandra_repo.delete_retry_job("PROCESSING", job["next_retry"], job["job_id"])
                except Exception as publish_error:
                    logger.error(f"Failed to publish re-dispatch for job {job['job_id']}: {publish_error}")
                    # Release claiming by inserting a new PENDING row back into Cassandra with backoff
                    self.cassandra_repo.delete_retry_job("PROCESSING", job["next_retry"], job["job_id"])
                    
                    backoff_sec = 2 ** (retry_count + 1)
                    next_retry_time = datetime.now(timezone.utc) + timedelta(seconds=backoff_sec)
                    
                    updated_job = {
                        "status": "PENDING",
                        "next_retry": next_retry_time,
                        "job_id": job["job_id"],
                        "job_type": job_type,
                        "payload": job["payload"],
                        "retry_count": retry_count + 1,
                        "max_retry": max_retry,
                        "last_error": str(publish_error),
                        "created_at": job["created_at"]
                    }
                    self.cassandra_repo.insert_retry_job(updated_job)

            else:
                # 2. Maximum retries exhausted - route payload to DLQ
                logger.critical(
                    f"Retry job {job['job_id']} reached max retry limit ({max_retry}). "
                    f"Failing to Dead Letter Queue (DLQ)."
                )
                try:
                    dlq_payload = {
                        "job_id": str(job["job_id"]),
                        "job_type": job_type,
                        "original_payload": payload,
                        "last_error": job.get("last_error")
                    }
                    await self.producer.publish_task(
                        topic=settings.KAFKA_DLQ_TOPIC,
                        conversation_id=conversation_id,
                        payload=dlq_payload
                    )
                    # Delete PROCESSING row
                    self.cassandra_repo.delete_retry_job("PROCESSING", job["next_retry"], job["job_id"])
                    
                    # Store permanent failure audit row in Cassandra
                    final_failed_job = {
                        "status": "FAILED",
                        "next_retry": datetime.now(timezone.utc),
                        "job_id": job["job_id"],
                        "job_type": job_type,
                        "payload": job["payload"],
                        "retry_count": retry_count,
                        "max_retry": max_retry,
                        "last_error": f"Max retries exhausted. Last error: {job.get('last_error')}",
                        "created_at": job["created_at"]
                    }
                    self.cassandra_repo.insert_retry_job(final_failed_job)
                except Exception as dlq_error:
                    logger.error(f"Failed to route job {job['job_id']} to DLQ: {dlq_error}")

    async def stop(self) -> None:
        """Stops the retry scheduler background loop."""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Retry Scheduler stopped.")
