"""user.deleted (auth-service, integration plan 4.4.6): delete what memory-service holds for the user.

  1. the user's scheduled compaction: their members of the `compact:idle` sorted set are removed (and, while the conversation
     deletes below run, nothing re-adds them: the user cannot send messages any more);
  2. the user's saved facts: every row of every category (active AND retired history) in Cassandra, and every vector in Milvus;
  3. the user's memory settings and the tombstones of deleted wordings;
  4. per-conversation data (summaries, archived versions, snapshot, the durable message window, caches) of the conversations this
     service can attribute to the user (a fact's source conversation, the idle-compaction index), through the EXISTING path
     ConversationDeleteWorker.purge. Whatever it cannot attribute is removed by the `conversation.deleted` events that
     conversation-service publishes when it purges the user's conversations after its retention period (that worker already
     handles them); conversation-service blocks the user's routes at once, so nothing new can be added meanwhile.

Idempotent: a second run (event replay) finds nothing left and changes nothing. Raises on a real failure so the event is retried.
"""

import logging
from typing import Awaitable, Callable, Dict, Optional, Set

from app.services import memory_policy

logger = logging.getLogger("memory_service.services.user_deletion")

IDLE_KEY = "compact:idle"


class UserDeletionService:
    def __init__(self, cassandra_repo, milvus_repo, redis_client=None,
                 purge_conversation: Optional[Callable[[Dict], Awaitable[str]]] = None):
        self.cassandra = cassandra_repo
        self.milvus = milvus_repo
        self.redis = redis_client
        self.purge_conversation = purge_conversation

    async def _idle_members(self, user_id: str) -> Set[str]:
        """Members `conversation|user` of the idle-compaction index that belong to the user."""
        if self.redis is None:
            return set()
        suffix, members = f"|{user_id}", set()
        cursor = 0
        while True:
            cursor, batch = await self.redis.zscan(IDLE_KEY, cursor=cursor, match=f"*{suffix}", count=500)
            for member, _score in batch:
                members.add(member.decode() if isinstance(member, bytes) else member)
            if not cursor:
                return members

    async def delete_user(self, user_id: str) -> Dict[str, int]:
        stats = {"facts": 0, "conversations": 0, "idle_removed": 0}
        conversations: Set[str] = set()

        idle = await self._idle_members(user_id)
        for member in idle:
            await self.redis.zrem(IDLE_KEY, member)
            conversations.add(member.partition("|")[0])
        stats["idle_removed"] = len(idle)

        facts = []
        for category in memory_policy.CATEGORIES:
            facts += self.cassandra.get_facts(user_id, category, include_inactive=True)
        for fact in facts:
            if fact.get("conversation_id"):
                conversations.add(str(fact["conversation_id"]))
        try:
            self.milvus.delete_user_facts(user_id)
        except Exception:
            logger.error("user_deleted: Milvus vectors of user %s could not be deleted; the event will be retried", user_id)
            raise
        for fact in facts:
            self.cassandra.delete_fact(user_id, fact["category"], fact["fact_id"])
        stats["facts"] = len(facts)
        self.cassandra.delete_user_memory_controls(user_id)

        if self.purge_conversation is not None:
            for conversation_id in sorted(conversations):
                await self.purge_conversation({"conversation_id": conversation_id, "user_id": user_id})
                stats["conversations"] += 1
        logger.info("user_deleted_memory_removed user=%s %s", user_id, stats)
        return stats
