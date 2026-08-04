"""
app/workers/summary_worker.py

Summary generation background worker.
Polls summary requests from Kafka, generates new summaries, and transitions states.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.models.memory import MemoryState
from app.repositories.cassandra_repository import CassandraRepository
from app.services.memory_service import MemoryService
from app.services.summary_service import SummaryService

logger = logging.getLogger("memory_service.workers.summary_worker")


class SummaryWorker:
    """
    Background worker that consumes memory.summary.request events, updates the
    conversation summary via LLM gRPC pool, and schedules fact extraction.
    """

    def __init__(
        self,
        cassandra_session,
        memory_service: MemoryService,
        summary_service: SummaryService,
        bootstrap_servers: str = settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id: str = "summary-worker-group"
    ):
        self.session = cassandra_session
        self.memory_service = memory_service
        self.summary_service = summary_service
        self.cassandra_repo = CassandraRepository(self.session)
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer = None
        self.is_running = False
        self._task = None

    async def start(self) -> None:
        """Starts the SummaryWorker consumer loop."""
        logger.info(f"Starting SummaryWorker on topic {settings.KAFKA_SUMMARY_TOPIC}")
        try:
            self._consumer = AIOKafkaConsumer(
                settings.KAFKA_SUMMARY_TOPIC,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                enable_auto_commit=False,
                session_timeout_ms=settings.KAFKA_SESSION_TIMEOUT_MS,
                max_poll_interval_ms=settings.KAFKA_MAX_POLL_INTERVAL_MS
            )
            await self._consumer.start()
            self.is_running = True
            self._task = asyncio.create_task(self._loop())
            logger.info("SummaryWorker started successfully.")
        except Exception as e:
            logger.critical(f"Failed to start SummaryWorker: {e}")
            raise e

    async def _loop(self) -> None:
        """Primary message loop."""
        while self.is_running:
            try:
                msg_batch = await self._consumer.getmany(timeout_ms=1000)
                if not msg_batch:
                    continue

                for tp, messages in msg_batch.items():
                    for msg in messages:
                        payload = None
                        conversation_id = "unknown"
                        try:
                            payload = json.loads(msg.value.decode("utf-8"))
                            conversation_id = payload["conversation_id"]
                            user_id = payload.get("user_id", "unknown_user")
                            version = payload.get("version", 0)

                            logger.info(f"Processing summary request for conversation: {conversation_id}")

                            # 1. Transition snapshot to SUMMARIZING
                            snapshot = await self.memory_service.transition_state(
                                conversation_id, MemoryState.SUMMARIZING
                            )

                            # 2. Call SummaryService to execute Incremental Summarization
                            updated_snap = await self.summary_service.process_incremental_summary(
                                conversation_id
                            )

                            # 3. Transition snapshot to FACT_PENDING
                            await self.memory_service.transition_state(
                                conversation_id, MemoryState.FACT_PENDING
                            )

                            # 4. Schedule outbox job for Fact Worker
                            outbox_job = {
                                "status": "PENDING",
                                "created_at": datetime.now(timezone.utc),
                                "job_id": uuid.uuid4(),
                                "topic": settings.KAFKA_FACT_TOPIC,
                                "conversation_id": conversation_id,
                                "payload": json.dumps({
                                    "conversation_id": conversation_id,
                                    "user_id": user_id,
                                    "version": updated_snap.get("summary_version") or version
                                }),
                                "attempt_count": 0
                            }
                            self.cassandra_repo.insert_outbox_job(outbox_job)

                        except Exception as e:
                            logger.error(f"Error in SummaryWorker processing job: {e}")
                            if payload and conversation_id != "unknown":
                                # Handle retry/DLQ scheduling
                                await self.memory_service.handle_failure(
                                    conversation_id=conversation_id,
                                    failed_state=MemoryState.SUMMARIZING,
                                    job_type="summary",
                                    payload=payload,
                                    error_msg=str(e),
                                    attempt_count=payload.get("attempt_count", 0)
                                )

                    # Commit partition batch offsets
                    await self._consumer.commit()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SummaryWorker polling loop error: {e}")
                await asyncio.sleep(2.0)

    async def stop(self) -> None:
        """Gracefully stops the worker connection and loops."""
        logger.info("Initiating graceful shutdown for SummaryWorker...")
        self.is_running = False
        if self._task:
            try:
                # Wait up to 10 seconds for the active message batch to finish processing
                await asyncio.wait_for(asyncio.shield(self._task), timeout=10.0)
                logger.info("SummaryWorker loop exited cleanly.")
            except asyncio.TimeoutError:
                logger.warning("Graceful shutdown timed out. Forcefully cancelling SummaryWorker loop.")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
            logger.info("SummaryWorker consumer stopped.")
