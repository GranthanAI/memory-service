"""
app/repositories/memory_repository.py

Unified Memory Repository orchestrating read-through caching and write-through
persistence across Cassandra (Source of Truth) and Redis (Hot Cache).
"""

import logging
from typing import Any, Dict, List, Optional

from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.redis_repository import RedisRepository

logger = logging.getLogger("memory_service.repositories.memory_repository")


class MemoryRepository:
    """
    Combines Cassandra and Redis repositories to provide transparent, high-performance
    read-through caching and write-through updates for snapshots, summaries, and messages.
    """

    def __init__(self, cassandra_repo: CassandraRepository, redis_repo: RedisRepository):
        self.cassandra_repo = cassandra_repo
        self.redis_repo = redis_repo

    # ─── Snapshot Operations ─────────────────────────────────────────────────

    async def get_snapshot(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Reads snapshot from Redis hot cache, falling back to Cassandra and backfilling cache on miss.
        """
        snapshot = await self.redis_repo.get_snapshot(conversation_id)
        if snapshot:
            return snapshot

        logger.debug(f"Redis cache miss for snapshot {conversation_id}. Fetching from Cassandra.")
        snapshot = self.cassandra_repo.get_snapshot(conversation_id)
        if snapshot:
            await self.redis_repo.set_snapshot(snapshot)
        return snapshot

    async def save_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Saves a conversation snapshot to Cassandra and updates the Redis hot cache.
        """
        self.cassandra_repo.upsert_snapshot(snapshot)
        await self.redis_repo.set_snapshot(snapshot)

    # ─── Summary Operations ──────────────────────────────────────────────────

    async def get_summary(self, conversation_id: str) -> Optional[str]:
        """
        Reads summary text from Redis cache, falling back to Cassandra and caching on miss.
        """
        summary_text = await self.redis_repo.get_summary(conversation_id)
        if summary_text:
            return summary_text

        logger.debug(f"Redis cache miss for summary {conversation_id}. Fetching from Cassandra.")
        summary = self.cassandra_repo.get_summary(conversation_id)
        if summary:
            summary_text = summary["summary_text"]
            await self.redis_repo.set_summary(conversation_id, summary_text)
            return summary_text
        return None

    async def save_summary(self, summary_record: Dict[str, Any]) -> None:
        """
        Persists summary record to Cassandra and caches the text in Redis.
        """
        self.cassandra_repo.upsert_summary(summary_record)
        await self.redis_repo.set_summary(
            summary_record["conversation_id"],
            summary_record["summary_text"]
        )

    # ─── Recent Messages Operations ──────────────────────────────────────────

    async def get_recent_messages(self, conversation_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Reads the recent messages sliding window from Redis, falling back to Cassandra
        and backfilling the sliding list on cache miss.
        """
        messages = await self.redis_repo.get_recent_messages(conversation_id)
        if messages:
            # Redis stores them in order: newest first. Limit if needed.
            return messages[:limit]

        logger.debug(f"Redis cache miss for recent messages {conversation_id}. Fetching from Cassandra.")
        messages = self.cassandra_repo.get_recent_messages(conversation_id, limit=limit)
        if messages:
            # Re-hydrate the Redis sliding list cache using Cassandra's records
            await self.redis_repo.set_recent_messages(conversation_id, messages)
        return messages

    async def append_recent_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """
        Appends a message to Cassandra durable backup and pushes it to Redis sliding list.
        """
        self.cassandra_repo.append_recent_message(conversation_id, message)
        await self.redis_repo.push_recent_message(conversation_id, message)

    # ─── Invalidation ────────────────────────────────────────────────────────

    async def invalidate_conversation(self, conversation_id: str) -> None:
        """
        Evicts all Redis hot cache keys related to a conversation.
        """
        await self.redis_repo.invalidate_conversation(conversation_id)
