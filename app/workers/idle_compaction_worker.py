"""
app/workers/idle_compaction_worker.py

Short conversations never reach the message-count or context-pressure thresholds, so nothing was ever summarized and no
long-term fact was extracted from them. This worker closes that gap: a conversation that has been quiet for
IDLE_COMPACTION_MINUTES and has messages not yet covered by its summary gets a summary request; the normal pipeline
(summary -> fact extraction -> embedding) then does the rest, including extracting facts from exactly those messages.

Activity is tracked in a Redis sorted set (`compact:idle`, score = last message time, member = "conversation|user"),
updated at ingestion. Claiming a due conversation is `ZREM` (exactly one worker removes it), so replicas never double-request.
A conversation whose pipeline is busy (snapshot not ACTIVE) is put back and retried on a later sweep.
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.metrics import IDLE_COMPACTION_RESULTS
from app.models.memory import MemoryState
from app.services.summary_service import FETCH_LIMIT, select_new_messages

logger = logging.getLogger("memory_service.workers.idle_compaction_worker")

IDLE_KEY = "compact:idle"
SWEEP_BATCH = 50


class IdleCompactionWorker:
    def __init__(self, redis_client, memory_repo, memory_service, cassandra_repo):
        self.redis = redis_client
        self.memory_repo = memory_repo
        self.memory_service = memory_service
        self.cassandra_repo = cassandra_repo
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

    def _has_unsummarized(self, conversation_id: str) -> bool:
        row = self.cassandra_repo.get_summary(conversation_id) or {}
        fetched = self.cassandra_repo.get_recent_messages(conversation_id, limit=FETCH_LIMIT)
        new, _gap = select_new_messages(fetched, row.get("through_message_id"), row.get("through_created_at"))
        return bool(new)

    async def _requeue(self, member: str, now: float) -> None:
        await self.redis.zadd(IDLE_KEY, {member: now})

    async def sweep(self, now: Optional[float] = None) -> Dict[str, int]:
        """Requests a summary for every conversation idle long enough. Returns counts per outcome."""
        now = time.time() if now is None else now
        cutoff = now - settings.IDLE_COMPACTION_MINUTES * 60
        counts: Dict[str, int] = {}
        members = await self.redis.zrangebyscore(IDLE_KEY, 0, cutoff, start=0, num=SWEEP_BATCH)
        for member in members or []:
            member = member.decode() if isinstance(member, bytes) else member
            if not await self.redis.zrem(IDLE_KEY, member):
                continue                                           # another replica claimed it
            outcome = await self._request_if_needed(member, now)
            counts[outcome] = counts.get(outcome, 0) + 1
            IDLE_COMPACTION_RESULTS.labels(outcome=outcome).inc()
        return counts

    async def _request_if_needed(self, member: str, now: float) -> str:
        conversation_id, _, user_id = member.partition("|")
        try:
            if not self._has_unsummarized(conversation_id):
                return "nothing_new"
            snapshot = await self.memory_repo.get_snapshot(conversation_id)
            if not snapshot:
                return "no_snapshot"
            if str(snapshot.get("state")) != MemoryState.ACTIVE.value:
                await self._requeue(member, now)                   # a pipeline run is in flight: look again later
                return "busy"
            await self.memory_service.transition_state(conversation_id, MemoryState.SUMMARY_PENDING)
            self.cassandra_repo.insert_outbox_job({
                "status": "PENDING", "created_at": datetime.now(timezone.utc), "job_id": uuid.uuid4(),
                "topic": settings.KAFKA_SUMMARY_TOPIC, "conversation_id": conversation_id,
                "payload": json.dumps({"conversation_id": conversation_id, "user_id": user_id or snapshot.get("user_id"),
                                       "version": int(snapshot.get("summary_version") or 0), "reason": "idle"}),
                "attempt_count": 0})
            return "requested"
        except Exception as exc:
            logger.warning("idle compaction for %s failed (%s); retrying later", conversation_id, type(exc).__name__)
            try:
                await self._requeue(member, now)
            except Exception:
                pass
            return "error"

    async def start(self) -> None:
        if not settings.IDLE_COMPACTION_ENABLED or self.redis is None:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("IdleCompactionWorker started (idle after %s min).", settings.IDLE_COMPACTION_MINUTES)

    async def _loop(self) -> None:
        while self.is_running:
            try:
                await self.sweep()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("idle compaction sweep failed: %s", type(exc).__name__)
            await asyncio.sleep(settings.IDLE_COMPACTION_POLL_SECONDS)

    async def stop(self) -> None:
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
