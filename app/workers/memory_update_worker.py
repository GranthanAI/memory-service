"""
app/workers/memory_update_worker.py

Ingestion worker consuming memory.update.requested (published by llm-service's
KafkaPublisher.publish_memory_update after every completed response). This is
the missing link between real conversations and memory-service's storage —
without it, EventDispatcher/SummaryWorker/FactWorker/EmbeddingWorker never
receive any data to operate on, since nothing else in the system publishes to
memory.summary.request / memory.fact.request / memory.embedding.request.

The llm-service event carries one turn (user_message + assistant_response,
no message_id, no event_id) rather than the canonical MemoryEventEnvelope
shape EventDispatcher.dispatch() expects. This worker adapts by synthesizing
one dispatch() call per side of the turn (user, then assistant), deriving
stable message_id/event_id values via uuid5 so redelivery is naturally
idempotent (EventDispatcher's own idempotency check on event_id still applies).
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.events.dispatcher import EventDispatcher

logger = logging.getLogger("memory_service.workers.memory_update_worker")

_NAMESPACE = uuid.UUID("d6a1f9d2-8b3e-4b8a-9d5a-6f3b2e1c4a70")


def _derive_id(*parts: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, ":".join(parts)))


class MemoryUpdateWorker:
    """
    Background worker that consumes memory.update.requested events and feeds
    them into EventDispatcher as a user-turn + assistant-turn pair.
    """

    def __init__(
        self,
        dispatcher: EventDispatcher,
        bootstrap_servers: str = settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id: str = "memory-update-worker-group",
    ):
        self.dispatcher = dispatcher
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer = None
        self.is_running = False
        self._task = None

    async def start(self) -> None:
        logger.info(f"Starting MemoryUpdateWorker on topic {settings.KAFKA_MEMORY_UPDATE_TOPIC}")
        self._consumer = AIOKafkaConsumer(
            settings.KAFKA_MEMORY_UPDATE_TOPIC,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=False,
            session_timeout_ms=settings.KAFKA_SESSION_TIMEOUT_MS,
            max_poll_interval_ms=settings.KAFKA_MAX_POLL_INTERVAL_MS,
        )
        await self._consumer.start()
        self.is_running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("MemoryUpdateWorker started successfully.")

    async def _loop(self) -> None:
        while self.is_running:
            try:
                msg_batch = await self._consumer.getmany(timeout_ms=1000)
                if not msg_batch:
                    continue

                for tp, messages in msg_batch.items():
                    for msg in messages:
                        try:
                            payload = json.loads(msg.value.decode("utf-8"))
                            await self._process_turn(payload)
                        except Exception as e:
                            logger.error(f"MemoryUpdateWorker failed to process message: {e}")
                    await self._consumer.commit()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MemoryUpdateWorker polling loop error: {e}")
                await asyncio.sleep(2.0)

    async def _process_turn(self, payload: dict) -> None:
        conversation_id = payload["conversation_id"]
        user_id = payload["user_id"]
        timestamp = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()

        base_key = f"{conversation_id}:{timestamp}"

        user_envelope = {
            "event_id": _derive_id(base_key, "user"),
            "event_type": "chat.message.created",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "payload": {
                "message_id": _derive_id(base_key, "user", "msg"),
                "role": "user",
                "content": payload.get("user_message", ""),
                "created_at": timestamp,
            },
        }
        assistant_envelope = {
            "event_id": _derive_id(base_key, "assistant"),
            "event_type": "chat.response.completed",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "payload": {
                "message_id": _derive_id(base_key, "assistant", "msg"),
                "role": "assistant",
                "content": payload.get("assistant_response", ""),
                "created_at": timestamp,
            },
        }

        await self.dispatcher.dispatch(user_envelope)
        await self.dispatcher.dispatch(assistant_envelope)

        logger.info(f"Ingested conversation turn into memory pipeline for {conversation_id}")

    async def stop(self) -> None:
        logger.info("Initiating graceful shutdown for MemoryUpdateWorker...")
        self.is_running = False
        if self._task:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=10.0)
            except asyncio.TimeoutError:
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
            logger.info("MemoryUpdateWorker consumer stopped.")
