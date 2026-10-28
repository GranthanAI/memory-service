"""
app/workers/branch_summary_worker.py

Consumes `conversation.created` (published by conversation-service; for a branch it carries parent_conversation_id and
branch_message_id) and makes sure the parent has a summary that a child can inherit at the branch point:

  * a stored version (current or archived) already ends exactly at the branch message -> nothing to do;
  * the parent's current summary is behind the branch point (or there is none yet)    -> request a summary job for the
    parent BOUNDED at the branch message (`upto_message_id`), so it ends exactly there;
  * the parent was already summarized past the branch point and no archived version is valid there -> cannot be rebuilt
    from a newer summary; counted in `memory_branch_summary_total{outcome="unavailable"}`. The child still gets the
    parent's raw messages from conversation-service, and the context builder bridges from the best older version.

Best effort by design: a failure here never blocks anything (the request-time path degrades to an older version + raw
messages), so errors are logged and the offset is committed.
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.core.metrics import BRANCH_SUMMARY_RESULTS
from app.events.topics import CONVERSATION_CREATED
from app.services.branch_validity import summary_valid_at_branch
from app.services.context_builder import ContextBuilder

logger = logging.getLogger("memory_service.workers.branch_summary_worker")


class BranchSummaryWorker:
    def __init__(self, memory_repo, cassandra_repo, scheduler=None,
                 bootstrap_servers: str = settings.KAFKA_BOOTSTRAP_SERVERS, group_id: str = "branch-summary-worker-group"):
        self.memory_repo = memory_repo
        self.cassandra_repo = cassandra_repo
        self.scheduler = scheduler
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer = None
        self.is_running = False
        self._task = None

    async def ensure_branch_summary(self, event_payload: Dict[str, Any]) -> str:
        """Returns the outcome label: not_a_branch | present | requested | unavailable | skipped."""
        parent = event_payload.get("parent_conversation_id")
        upto = event_payload.get("branch_message_id")
        user_id = event_payload.get("user_id")
        if not parent or not upto:
            return "not_a_branch"
        snapshot = await self.memory_repo.get_snapshot(str(parent))
        if snapshot and user_id and snapshot.get("user_id") not in (None, user_id):
            return "skipped"                                        # never act on someone else's conversation
        candidates = ContextBuilder._summary_candidates(self.cassandra_repo, str(parent))
        valid = [c for c in candidates if (c.get("summary_text") or "").strip()
                 and summary_valid_at_branch(c.get("through_message_id"), upto, c.get("through_created_at"))]
        if any(str(c.get("through_message_id") or "").lower() == str(upto).lower() for c in valid):
            return "present"
        current = candidates[0] if candidates else None
        past_branch = bool(current and (current.get("summary_text") or "").strip() and not summary_valid_at_branch(
            current.get("through_message_id"), upto, current.get("through_created_at")))
        if past_branch and not valid:
            return "unavailable"
        if self.scheduler is not None and not await self.scheduler.claim(str(parent)):
            return "skipped"                                        # a summary job for the parent is already in flight
        self.cassandra_repo.insert_outbox_job({
            "status": "PENDING", "created_at": datetime.now(timezone.utc), "job_id": uuid.uuid4(),
            "topic": settings.KAFKA_SUMMARY_TOPIC, "conversation_id": str(parent),
            "payload": json.dumps({"conversation_id": str(parent), "user_id": user_id or (snapshot or {}).get("user_id"),
                                   "version": int((current or {}).get("summary_version") or 0),
                                   "reason": "branch", "upto_message_id": str(upto)}),
            "attempt_count": 0})
        return "requested"

    async def handle(self, raw: bytes) -> str:
        try:
            envelope = json.loads(raw.decode("utf-8"))
            payload = envelope.get("payload", envelope) if isinstance(envelope, dict) else {}
            outcome = await self.ensure_branch_summary(payload)
        except Exception as exc:                                    # best effort: see the module docstring
            logger.warning("branch summary check failed (%s: %s)", type(exc).__name__, exc)
            outcome = "error"
        BRANCH_SUMMARY_RESULTS.labels(outcome=outcome).inc()
        return outcome

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            CONVERSATION_CREATED,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=False,
            session_timeout_ms=settings.KAFKA_SESSION_TIMEOUT_MS,
            max_poll_interval_ms=settings.KAFKA_MAX_POLL_INTERVAL_MS,
        )
        await self._consumer.start()
        self.is_running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("BranchSummaryWorker started.")

    async def _loop(self) -> None:
        while self.is_running:
            try:
                batch = await self._consumer.getmany(timeout_ms=1000)
                if not batch:
                    continue
                for _tp, messages in batch.items():
                    for msg in messages:
                        await self.handle(msg.value)
                await self._consumer.commit()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"BranchSummaryWorker loop error: {exc}")
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
