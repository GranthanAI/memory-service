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
from typing import Any, Optional

from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.events.dispatcher import EventDispatcher
from app.services.summary_service import _as_utc
from app.core.metrics import MEMORY_UPDATE_RESULTS
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


# Ingestion failures are retried through the same retry_jobs table the other workers use (RetryScheduler re-publishes the
# original payload to the memory-update topic after an exponential backoff); after this many attempts the job is parked
# as FAILED (the dead-letter record) instead of being retried forever.
MAX_INGEST_ATTEMPTS = 5
JOB_TYPE = "memory_update"


class MemoryUpdateWorker:
    """
    Background worker that consumes memory.update.requested events and feeds
    them into EventDispatcher as a user-turn + assistant-turn pair.

    Failure handling: a turn is two dispatches, each atomic and idempotent by event id. If either fails, the turn is
    NOT dropped: a retry job carrying the original payload is written (reprocessing re-dispatches both sides; the side
    that already succeeded is skipped as a duplicate, so a half-stored turn is completed). The offset is committed only
    after the retry job is durable; if even that fails the offset is not committed and the consumer rewinds.
    """

    def __init__(
        self,
        dispatcher: EventDispatcher,
        bootstrap_servers: str = settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id: str = "memory-update-worker-group",
        cassandra_repo=None,
    ):
        self.dispatcher = dispatcher
        self.cassandra_repo = cassandra_repo        # retry/DLQ records (retry_jobs); None disables retries (old behaviour)
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
                        if not await self._handle_message(tp, msg):
                            break                      # could not even record the failure: rewound, will be redelivered
                    else:
                        await self._consumer.commit()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MemoryUpdateWorker polling loop error: {e}")
                await asyncio.sleep(2.0)

    async def _handle_message(self, tp, msg) -> bool:
        """Processes one Kafka message. True = safe to commit past it (processed, or a retry/dead-letter record is
        durable). False = nothing was recorded; the consumer was rewound to this message and must not commit."""
        raw = msg.value
        payload: Optional[dict] = None
        try:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload is not a JSON object")
            await self._process_turn(payload)
            MEMORY_UPDATE_RESULTS.labels(outcome="ingested").inc()
            return True
        except Exception as exc:
            logger.error(f"MemoryUpdateWorker failed to process message: {type(exc).__name__}: {exc}")
            try:
                self._schedule_retry_or_park(payload, raw, exc)
                return True
            except Exception as record_exc:
                logger.error(f"MemoryUpdateWorker could not record the failure ({type(record_exc).__name__}); rewinding")
                MEMORY_UPDATE_RESULTS.labels(outcome="rewound").inc()
                try:
                    self._consumer.seek(tp, msg.offset)
                except Exception:
                    pass
                await asyncio.sleep(2.0)
                return False

    def _schedule_retry_or_park(self, payload: Optional[dict], raw: bytes, exc: Exception) -> None:
        """Writes the retry job (or, past the attempt limit / for an undecodable payload, a FAILED dead-letter record)."""
        if self.cassandra_repo is None:
            raise RuntimeError("no retry store configured")
        now = datetime.now(timezone.utc)
        error = f"{type(exc).__name__}: {exc}"[:500]
        decodable = isinstance(payload, dict) and bool(payload.get("conversation_id"))
        attempt = int(payload.get("attempt_count", 0)) if decodable else MAX_INGEST_ATTEMPTS
        # Never retry what cannot succeed: an undecodable/foreign payload goes straight to the dead-letter record.
        if decodable and attempt + 1 < MAX_INGEST_ATTEMPTS and not isinstance(exc, (KeyError, ValueError)):
            retry_payload = {**payload, "attempt_count": attempt + 1}
            backoff = 2 ** (attempt + 1)
            status, next_retry, outcome = "PENDING", now + timedelta(seconds=backoff), "retry_scheduled"
            stored = json.dumps(retry_payload)
        else:
            status, next_retry, outcome = "FAILED", now, "dead_lettered"
            stored = json.dumps(payload) if decodable else raw.decode("utf-8", errors="replace")[:20000]
        self.cassandra_repo.insert_retry_job({
            "status": status,
            "next_retry": next_retry,
            "job_id": uuid.uuid4(),
            "job_type": JOB_TYPE,
            "payload": stored,
            "retry_count": attempt + 1 if status == "PENDING" else attempt,
            "max_retry": MAX_INGEST_ATTEMPTS,
            "last_error": error if status == "PENDING" else f"Dead-lettered: {error}",
            "created_at": now,
        })
        MEMORY_UPDATE_RESULTS.labels(outcome=outcome).inc()

    async def _replace_discarded_answer(self, payload: dict, conversation_id: str) -> None:
        """A regenerated answer REPLACES the one it supersedes in the durable window (and the cached one), instead of
        sitting next to it. Matches by message id, so only events that carry real ids can replace (legacy uuid5 ids never
        match). The old answer may already be inside a summary: summaries are rebuilt from raw messages going forward,
        they are not edited retroactively."""
        old_id = payload.get("replaces_message_id")
        if not old_id or self.cassandra_repo is None:
            return
        for row in self.cassandra_repo.get_recent_messages(conversation_id, limit=400):
            if str(row.get("message_id")) == str(old_id):
                self.cassandra_repo.delete_recent_message_row(conversation_id, row["created_at"], row["message_id"])
                memory_repo = getattr(self.dispatcher, "memory_repo", None)
                if memory_repo is not None:
                    await memory_repo.invalidate_conversation(conversation_id)
                MEMORY_UPDATE_RESULTS.labels(outcome="answer_replaced").inc()
                return

    def _enqueue_explicit_memory(self, payload: dict, conversation_id: str, user_id: str, user_message_id: str) -> None:
        """Fast path for short conversations: facts used to be extracted only after a summary, so a chat of a few messages
        never produced any. A message that asks to be remembered ("remember that...", "from now on...") schedules an
        extraction for THAT message right away (side-channel job: it does not touch the snapshot state machine)."""
        from app.services import memory_policy
        if (not settings.EXPLICIT_MEMORY_FAST_PATH or self.cassandra_repo is None
                or not memory_policy.is_explicit_memory_request(payload.get("user_message", ""))):
            return
        self.cassandra_repo.insert_outbox_job({
            "status": "PENDING", "created_at": datetime.now(timezone.utc), "job_id": uuid.uuid4(),
            "topic": settings.KAFKA_FACT_TOPIC, "conversation_id": conversation_id,
            "payload": json.dumps({"conversation_id": conversation_id, "user_id": user_id, "version": 0,
                                   "reason": "explicit", "message_ids": [user_message_id]}),
            "attempt_count": 0})
        MEMORY_UPDATE_RESULTS.labels(outcome="explicit_memory_queued").inc()

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

        await self._replace_discarded_answer(payload, conversation_id)
        self._enqueue_explicit_memory(payload, conversation_id, user_id, ident["user_message_id"])

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
