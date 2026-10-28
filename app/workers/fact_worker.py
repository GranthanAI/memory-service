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

from app.services.memory_controls import MemoryControls
from app.core.config import settings
from app.models.memory import MemoryState
from app.services.llm_service import LLMService
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.memory_repository import MemoryRepository
from app.services.memory_service import MemoryService
from app.core.logging import set_log_context, clear_log_context

logger = logging.getLogger("memory_service.workers.fact_worker")

SIDE_CHANNEL_REASONS = ("explicit", "idle")
EXTRACTION_MESSAGE_CHARS = 600
EXTRACTION_MAX_MESSAGES = 60


def build_extraction_notes(summary_text: str, messages: list) -> str:
    """Extraction input: the exact messages first (the evidence), the conversation summary after them as context only."""
    lines = [f"- {m.get('role')}: {str(m.get('content') or '').strip()[:EXTRACTION_MESSAGE_CHARS]}"
             for m in messages[-EXTRACTION_MAX_MESSAGES:] if str(m.get("content") or "").strip()]
    parts = ["Messages to extract from (the user's own statements are the evidence):", *lines]
    if (summary_text or "").strip():
        parts += ["", "Conversation summary (context only; do not extract from it alone):", summary_text.strip()]
    return "\n".join(parts)


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
        self.controls = MemoryControls(self.cassandra_repo)
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer = None
        self.is_running = False
        self._task = None

    async def _extract_facts(self, user_id: str, conversation_id: str, extract_request) -> list:
        """LLM fact extraction, unless the user turned memory off: then nothing is learned
        (no LLM call) and the snapshot state machine still advances with an empty list."""
        if not self.controls.is_enabled(user_id):
            logger.info(f"Memory is off for user {user_id}; skipping fact extraction.")
            return []
        logger.info(f"Invoking internal LLM service to extract facts from summary of conversation '{conversation_id}'.")
        facts_response = await self.llm_service.extract_facts(extract_request)
        return [f"{f.category}:{f.importance}:{f.statement}" for f in facts_response.facts]

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
                            trace_id = payload.get("trace_id") or str(uuid.uuid4())

                            set_log_context(
                                trace_id=trace_id,
                                conversation_id=conversation_id,
                                summary_version=version
                            )

                            logger.info(f"Processing fact extraction request for conversation: {conversation_id}")

                            # Explicit-memory / idle jobs are side-channel: they run next to the snapshot state machine,
                            # not through it (the snapshot is usually ACTIVE, which has no edge to EXTRACTING_FACTS).
                            reason = payload.get("reason")
                            side_channel = reason in SIDE_CHANNEL_REASONS
                            message_ids = [str(i) for i in (payload.get("message_ids") or [])]

                            # 1. Transition snapshot to EXTRACTING_FACTS
                            if not side_channel:
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
                            # The summary alone loses explicit statements ("remember that...", rules the user
                            # set), so the user's own recent words are included as evidence.
                            user_lines = [
                                f"- {str(m['content']).strip()[:300]}"
                                for m in chronological_messages
                                if m.get("role") == "user" and str(m.get("content") or "").strip()
                            ][-10:]
                            notes = summary_text
                            if user_lines:
                                notes = "\n".join([summary_text, "", "The user's own recent messages:", *user_lines])
                            if message_ids:
                                # Extract from EXACTLY the messages this job is about; the summary is only context.
                                wanted = set(message_ids)
                                consumed = [m for m in self.cassandra_repo.get_recent_messages(conversation_id, limit=400)
                                            if str(m.get("message_id")) in wanted]
                                consumed.sort(key=lambda m: (m.get("created_at"), str(m.get("message_id"))))
                                if consumed:
                                    notes = build_extraction_notes(summary_text, consumed)
                            extract_request = FactExtractRequest(summary=notes)

                            facts = await self._extract_facts(user_id, conversation_id, extract_request)
                            logger.info(f"Extracted {len(facts)} facts from conversation '{conversation_id}'.")

                            # 4. Transition snapshot to EMBEDDING_PENDING
                            if side_channel:
                                updated_snap = {"fact_version": version}
                            else:
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
                                    "version": updated_snap.get("fact_version") or version,
                                    **({"reason": reason} if side_channel else {}),
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
                        finally:
                            clear_log_context()

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
