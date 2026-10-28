"""
app/workers/memory_update_worker.py

Ingestion worker consuming memory.update.requested (published by llm-service's
KafkaPublisher.publish_memory_update after every completed response). This is
the missing link between real conversations and memory-service's storage —
without it, EventDispatcher/SummaryWorker/FactWorker/EmbeddingWorker never
receive any data to operate on, since nothing else in the system publishes to
memory.summary.request / memory.fact.request / memory.embedding.request.

The llm-service event carries one turn (user_message + assistant_response)
rather than the canonical MemoryEventEnvelope shape EventDispatcher.dispatch()
expects. This worker adapts by synthesizing one dispatch() call per side of the
turn (user, then assistant), so redelivery is naturally idempotent
(EventDispatcher's own idempotency check on event_id still applies).

Message identity and order
--------------------------
Newer events carry optional `user_message_id`, `assistant_message_id`, `user_created_at` and `assistant_created_at`
(the real conversation-service ids and times). When present they are used as is. Legacy events carry none of them:
the ids stay uuid5-derived (so a redelivered legacy event is still idempotent) and the assistant message is stamped
LEGACY_ASSISTANT_OFFSET after the user message. Messages are ordered by (created_at, message_id), so equal timestamps
would let the random uuid5 decide the order of the two sides of a turn. In every case the assistant message is forced
strictly after the user message of its turn.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone

from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.events.dispatcher import EventDispatcher
from app.services.summary_service import _as_utc
from app.utils.message_ids import uuid7_datetime

logger = logging.getLogger("memory_service.workers.memory_update_worker")

_NAMESPACE = uuid.UUID("d6a1f9d2-8b3e-4b8a-9d5a-6f3b2e1c4a70")


# Cassandra timestamps have millisecond precision: a smaller offset could collapse back to the same value when stored.
LEGACY_ASSISTANT_OFFSET = timedelta(milliseconds=1)


def _derive_id(*parts: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, ":".join(parts)))


def turn_identity(payload: dict) -> dict:
    """(ids, timestamps) of the two messages of a turn. Accepts both the new event shape and the legacy one."""
    conversation_id = payload["conversation_id"]
    timestamp = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
    base_key = f"{conversation_id}:{timestamp}"

    user_mid = payload.get("user_message_id") or None
    assistant_mid = payload.get("assistant_message_id") or None
    event_ts = _as_utc(timestamp) or datetime.now(timezone.utc)

    user_ts = _as_utc(payload.get("user_created_at")) or uuid7_datetime(user_mid) or event_ts
    assistant_ts = _as_utc(payload.get("assistant_created_at")) or uuid7_datetime(assistant_mid)
    if assistant_ts is None:
        assistant_ts = max(event_ts, user_ts)
    # The assistant answers the user: never at or before the user message, whatever the source of the times.
    if assistant_ts <= user_ts:
        assistant_ts = user_ts + LEGACY_ASSISTANT_OFFSET

    return {
        "user_event_id": _derive_id(base_key, "user"),
        "assistant_event_id": _derive_id(base_key, "assistant"),
        "user_message_id": str(user_mid) if user_mid else _derive_id(base_key, "user", "msg"),
        "assistant_message_id": str(assistant_mid) if assistant_mid else _derive_id(base_key, "assistant", "msg"),
        "user_created_at": user_ts.isoformat(),
        "assistant_created_at": assistant_ts.isoformat(),
        "timestamp": timestamp,
    }


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
        ident = turn_identity(payload)
        timestamp = ident["timestamp"]

        user_envelope = {
            "event_id": ident["user_event_id"],
            "event_type": "chat.message.created",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "payload": {
                "message_id": ident["user_message_id"],
                "role": "user",
                "content": payload.get("user_message", ""),
                "created_at": ident["user_created_at"],
            },
        }
        assistant_envelope = {
            "event_id": ident["assistant_event_id"],
            "event_type": "chat.response.completed",
            "conversation_id": conversation_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "payload": {
                "message_id": ident["assistant_message_id"],
                "role": "assistant",
                "content": payload.get("assistant_response", ""),
                "created_at": ident["assistant_created_at"],
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
