"""
app/workers/fact_worker.py

Fact extraction background worker.
Polls fact requests from Kafka, extracts new facts via LLM, and transitions states.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.models.memory import MemoryState
from app.services.llm_service import LLMService
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.memory_repository import MemoryRepository
from app.services.memory_service import MemoryService

logger = logging.getLogger("memory_service.workers.fact_worker")


class FactWorker:
    """
    Background worker that consumes memory.fact.request events, extracts
    facts from updated summaries using the LLM Service, and schedules vector embedding.
    """

    def __init__(
        self,
        cassandra_session,
        memory_service: MemoryService,
        memory_repo: MemoryRepository,
        llm_service: LLMService,
        bootstrap_servers: str = settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id: str = "fact-worker-group"
    ):
        self.session = cassandra_session
        self.memory_service = memory_service
        self.memory_repo = memory_repo
        self.llm_service = llm_service
        self.cassandra_repo = CassandraRepository(self.session)
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer = None
        self.is_running = False
        self._task = None

    async def start(self) -> None:
        """Starts the FactWorker consumer loop."""
        logger.info(f"Starting FactWorker on topic {settings.KAFKA_FACT_TOPIC}")
        try:
            self._consumer = AIOKafkaConsumer(
                settings.KAFKA_FACT_TOPIC,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                enable_auto_commit=False,
                session_timeout_ms=settings.KAFKA_SESSION_TIMEOUT_MS,
                max_poll_interval_ms=settings.KAFKA_MAX_POLL_INTERVAL_MS
            )
            await self._consumer.start()
            self.is_running = True
            self._task = asyncio.create_task(self._loop())
            logger.info("FactWorker started successfully.")
        except Exception as e:
            logger.critical(f"Failed to start FactWorker: {e}")
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

                            logger.info(f"Processing fact extraction request for conversation: {conversation_id}")

                            # 1. Transition snapshot to EXTRACTING_FACTS
                            await self.memory_service.transition_state(
                                conversation_id, MemoryState.EXTRACTING_FACTS
                            )

                            # 2. Retrieve the conversation summary & recent messages for context
                            summary_text = await self.memory_repo.get_summary(conversation_id) or ""
                            recent_messages = await self.memory_repo.get_recent_messages(conversation_id, limit=20)
                            
                            chronological_messages = list(reversed(recent_messages))
                            messages_payload = [
                                {
                                    "message_id": m["message_id"],
                                    "role": m["role"],
                                    "content": m["content"],
                                    "created_at": m.get("created_at").isoformat() if isinstance(m.get("created_at"), datetime) else str(m.get("created_at"))
                                }
                                for m in chronological_messages
                            ]

                            # 3. Call internal LLM Service to extract facts
                            from app.schemas.llm import FactExtractRequest
                            extract_request = FactExtractRequest(summary=summary_text)

                            logger.info(f"Invoking internal LLM service to extract facts from summary of conversation '{conversation_id}'.")
                            facts_response = await self.llm_service.extract_facts(extract_request)
                            facts = [
                                f"{f.category}:{f.importance}:{f.statement}"
                                for f in facts_response.facts
                            ]
                            logger.info(f"Extracted {len(facts)} facts from conversation '{conversation_id}'.")

                            # 4. Transition snapshot to EMBEDDING_PENDING
                            updated_snap = await self.memory_service.transition_state(
                                conversation_id, MemoryState.EMBEDDING_PENDING
                            )

                            # 5. Schedule outbox job for Embedding Worker
                            outbox_job = {
                                "status": "PENDING",
                                "created_at": datetime.now(timezone.utc),
                                "job_id": uuid.uuid4(),
                                "topic": settings.KAFKA_EMBEDDING_TOPIC,
                                "conversation_id": conversation_id,
                                "payload": json.dumps({
                                    "conversation_id": conversation_id,
                                    "user_id": user_id,
                                    "facts": facts,
                                    "version": updated_snap.get("fact_version") or version
                                }),
                                "attempt_count": 0
                            }
                            self.cassandra_repo.insert_outbox_job(outbox_job)

                        except Exception as e:
                            logger.error(f"Error in FactWorker processing job: {e}")
                            if payload and conversation_id != "unknown":
                                # Handle retry/DLQ scheduling
                                await self.memory_service.handle_failure(
                                    conversation_id=conversation_id,
                                    failed_state=MemoryState.EXTRACTING_FACTS,
                                    job_type="fact",
                                    payload=payload,
                                    error_msg=str(e),
                                    attempt_count=payload.get("attempt_count", 0)
                                )

                    # Commit partition batch offsets
                    await self._consumer.commit()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"FactWorker polling loop error: {e}")
                await asyncio.sleep(2.0)

    async def stop(self) -> None:
        """Gracefully stops the worker connection and loops."""
        logger.info("Initiating graceful shutdown for FactWorker...")
        self.is_running = False
        if self._task:
            try:
                # Wait up to 10 seconds for the active message batch to finish processing
                await asyncio.wait_for(asyncio.shield(self._task), timeout=10.0)
                logger.info("FactWorker loop exited cleanly.")
            except asyncio.TimeoutError:
                logger.warning("Graceful shutdown timed out. Forcefully cancelling FactWorker loop.")
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
            logger.info("FactWorker consumer stopped.")
