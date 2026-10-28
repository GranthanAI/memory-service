"""
app/workers/conversation_delete_worker.py

Consumes `conversation.deleted` (published by conversation-service when a conversation is deleted) and removes what
memory-service holds for it: the durable message window, the summary and all archived summary versions, the snapshot and
every cache key. Facts learned from the conversation are the USER's memory, not the conversation's, so they are kept unless
DELETE_FACTS_WITH_CONVERSATION is true (the user can delete them in the memory controls).

Consequence worth knowing: a branch made from the deleted conversation loses the inherited SUMMARY of it (it still receives the
parent's raw history from conversation-service). Idempotent: running twice is harmless.
"""
import asyncio
import json
import logging
import uuid
from typing import Any, Dict

from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.core.metrics import CONVERSATION_DELETE_RESULTS
from app.events.topics import CONVERSATION_DELETED
from app.services import memory_policy

logger = logging.getLogger("memory_service.workers.conversation_delete_worker")


class ConversationDeleteWorker:
    def __init__(self, memory_repo, cassandra_repo, milvus_repo, redis_client=None,
                 bootstrap_servers: str = settings.KAFKA_BOOTSTRAP_SERVERS,
                 group_id: str = "conversation-delete-worker-group"):
        self.memory_repo = memory_repo
        self.cassandra_repo = cassandra_repo
        self.milvus_repo = milvus_repo
        self.redis = redis_client
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer = None
        self.is_running = False
        self._task = None

    async def purge(self, event_payload: Dict[str, Any]) -> str:
        conversation_id = str(event_payload.get("conversation_id") or "")
        user_id = event_payload.get("user_id")
        if not conversation_id:
            return "ignored"
        snapshot = await self.memory_repo.get_snapshot(conversation_id)
        if snapshot and user_id and snapshot.get("user_id") not in (None, user_id):
            return "skipped"                                       # never purge another user's conversation
        self.cassandra_repo.delete_conversation_data(conversation_id)
        await self.memory_repo.invalidate_conversation(conversation_id)
        if self.redis is not None:
            try:
                await self.redis.delete(f"compact:msgs:{conversation_id}", f"compact:tokens:{conversation_id}",
                                        f"compact:inflight:{conversation_id}")
                if user_id:
                    await self.redis.zrem("compact:idle", f"{conversation_id}|{user_id}")
            except Exception as exc:
                logger.warning("could not clear compaction counters (%s)", type(exc).__name__)
        if settings.DELETE_FACTS_WITH_CONVERSATION and user_id:
            for category in memory_policy.CATEGORIES:
                for fact in self.cassandra_repo.get_facts(user_id, category, include_inactive=True):
                    if fact.get("conversation_id") == conversation_id:
                        self.cassandra_repo.delete_fact(user_id, category, fact["fact_id"])
                        self.milvus_repo.delete_fact(user_id, str(fact["fact_id"]))
        return "purged"

    async def handle(self, raw: bytes) -> str:
        try:
            envelope = json.loads(raw.decode("utf-8"))
            payload = envelope.get("payload", envelope) if isinstance(envelope, dict) else {}
            outcome = await self.purge(payload)
        except Exception as exc:
            logger.error("conversation purge failed (%s: %s)", type(exc).__name__, exc)
            outcome = "error"
        CONVERSATION_DELETE_RESULTS.labels(outcome=outcome).inc()
        return outcome

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            CONVERSATION_DELETED, bootstrap_servers=self.bootstrap_servers, group_id=self.group_id,
            enable_auto_commit=False, session_timeout_ms=settings.KAFKA_SESSION_TIMEOUT_MS,
            max_poll_interval_ms=settings.KAFKA_MAX_POLL_INTERVAL_MS)
        await self._consumer.start()
        self.is_running = True
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while self.is_running:
            try:
                batch = await self._consumer.getmany(timeout_ms=1000)
                if not batch:
                    continue
                for tp, messages in batch.items():
                    for msg in messages:
                        if await self.handle(msg.value) == "error":
                            # a purge that failed is retried: rewind to it and do not commit past it
                            self._consumer.seek(tp, msg.offset)
                            await asyncio.sleep(2.0)
                            break
                    else:
                        await self._consumer.commit()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"ConversationDeleteWorker loop error: {exc}")
                await asyncio.sleep(2.0)

    async def stop(self) -> None:
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
