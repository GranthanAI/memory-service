"""
app/services/compaction_scheduler.py

WHEN the conversation summary (compaction) is requested. The summary itself is still produced by SummaryService through the
existing outbox -> Kafka -> SummaryWorker path; this only decides that a request is due, and never creates a second one
while one is in flight.

Triggers (any one):
  count     SUMMARY_THRESHOLD messages (the original rule: message_count % threshold == 0), kept as the safety net.
  pressure  at least COMPACTION_MIN_NEW_MESSAGES messages since the last summary AND they carry at least
            COMPACTION_TRIGGER_TOKENS tokens. Twenty long messages (code, logs) fill a context window long before
            "20 messages" says so; a few short ones never trigger it. Pressure is accounted per message in Redis (O(1) at
            ingest, no history scan); if Redis loses the counters the count rule still fires.
  stale     the context builder found many messages not yet covered by the summary (summary lagging or a request lost).

Duplicates are harmless by design (a summary run with nothing new after the watermark commits nothing), but the in-flight
guard keeps them from being created at all.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

from app.core.config import SystemSettings, settings as default_settings
from app.core.metrics import COMPACTION_REQUESTS
from app.utils.token_counter import TokenCounter

logger = logging.getLogger("memory_service.services.compaction_scheduler")


class CompactionScheduler:
    def __init__(self, redis_client, cassandra_repo=None, counter: Optional[TokenCounter] = None,
                 cfg: SystemSettings = default_settings) -> None:
        self.redis = redis_client
        self.cassandra_repo = cassandra_repo
        self.counter = counter or TokenCounter()
        self.cfg = cfg

    @staticmethod
    def _k(kind: str, conversation_id: str) -> str:
        return f"compact:{kind}:{conversation_id}"

    async def note_message(self, conversation_id: str, content: str) -> Tuple[int, int]:
        """Accounts one new message toward the next summary. Returns (messages, tokens) pending since the last one."""
        if self.redis is None:
            return 0, 0
        ttl = self.cfg.COMPACTION_PENDING_TTL_SECONDS
        try:
            messages = int(await self.redis.incrby(self._k("msgs", conversation_id), 1))
            tokens = int(await self.redis.incrby(self._k("tokens", conversation_id), self.counter.count(content) + 4))
            await self.redis.expire(self._k("msgs", conversation_id), ttl)
            await self.redis.expire(self._k("tokens", conversation_id), ttl)
            return messages, tokens
        except Exception as exc:                       # accounting is an optimisation: the count rule still applies
            logger.warning("compaction accounting unavailable (%s)", type(exc).__name__)
            return 0, 0

    async def note_activity(self, conversation_id: str, user_id: str) -> None:
        """Remembers when the conversation last had a message, for the idle trigger (see IdleCompactionWorker)."""
        if self.redis is None:
            return
        try:
            import time
            await self.redis.zadd("compact:idle", {f"{conversation_id}|{user_id}": time.time()})
        except Exception as exc:
            logger.debug("idle accounting unavailable (%s)", type(exc).__name__)

    def pressure_due(self, pending_messages: int, pending_tokens: int) -> bool:
        return (pending_messages >= self.cfg.COMPACTION_MIN_NEW_MESSAGES
                and pending_tokens >= self.cfg.COMPACTION_TRIGGER_TOKENS)

    async def claim(self, conversation_id: str) -> bool:
        """True for exactly one caller while a request is in flight (SET NX with a TTL, so a lost job heals itself)."""
        if self.redis is None:
            return True
        try:
            return bool(await self.redis.set(self._k("inflight", conversation_id), "1",
                                             ex=self._inflight_ttl(), nx=True))
        except Exception:
            return True

    def _inflight_ttl(self) -> int:
        ttl = getattr(self.cfg, "compaction_inflight_ttl", None)
        return int(ttl) if isinstance(ttl, (int, float)) else self.cfg.COMPACTION_INFLIGHT_TTL_SECONDS

    async def renew(self, conversation_id: str) -> None:
        """Extends the in-flight guard of a running job (called after every model batch)."""
        if self.redis is None:
            return
        try:
            await self.redis.expire(self._k("inflight", conversation_id), self._inflight_ttl())
        except Exception as exc:
            logger.warning("could not renew the compaction guard (%s)", type(exc).__name__)

    async def reset(self, conversation_id: str, pending_messages: int = 0, pending_tokens: int = 0) -> None:
        """Called when a summary job ended. The pressure restarts from the new watermark: the counters are set to what
        is actually pending after it (messages that arrived while the job ran are NOT forgotten), and the in-flight
        guard is released."""
        if self.redis is None:
            return
        try:
            msgs_key, tokens_key = self._k("msgs", conversation_id), self._k("tokens", conversation_id)
            if pending_messages > 0:
                ttl = self.cfg.COMPACTION_PENDING_TTL_SECONDS
                await self.redis.set(msgs_key, int(pending_messages), ex=ttl)
                await self.redis.set(tokens_key, int(pending_tokens), ex=ttl)
                await self.redis.delete(self._k("inflight", conversation_id))
            else:
                await self.redis.delete(msgs_key, tokens_key, self._k("inflight", conversation_id))
        except Exception as exc:
            logger.warning("could not reset compaction accounting (%s)", type(exc).__name__)

    async def request_from_read(self, conversation_id: str, user_id: str, summary_version: int) -> bool:
        """The context builder saw a stale summary: schedule one update (never a duplicate). Best effort."""
        if self.cassandra_repo is None or not await self.claim(conversation_id):
            COMPACTION_REQUESTS.labels(reason="deduplicated").inc()
            return False
        try:
            from app.core.config import settings
            self.cassandra_repo.insert_outbox_job({
                "status": "PENDING", "created_at": datetime.now(timezone.utc), "job_id": uuid.uuid4(),
                "topic": settings.KAFKA_SUMMARY_TOPIC, "conversation_id": conversation_id,
                "payload": json.dumps({"conversation_id": conversation_id, "user_id": user_id,
                                       "version": summary_version, "reason": "stale_read"}),
                "attempt_count": 0})
        except Exception as exc:
            logger.warning("could not schedule a stale-summary update (%s)", type(exc).__name__)
            return False
        COMPACTION_REQUESTS.labels(reason="stale_read").inc()
        return True
