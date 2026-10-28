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

    async def update_snapshot_summary_fields(
        self, conversation_id: str, summary_version: int, last_summary_msg_id: Optional[str]
    ) -> None:
        """Column-scoped write of the committed summary's version/watermark; the cached snapshot is evicted so the
        next read sees the counters as stored (never a stale copy)."""
        self.cassandra_repo.update_snapshot_summary_fields(conversation_id, summary_version, last_summary_msg_id)
        await self.redis_repo.invalidate_conversation(conversation_id)

    async def transition_snapshot_state(
        self, conversation_id: str, state: str, new_version: int, expected_version: int
    ) -> bool:
        """Compare-and-set of the state column. True when applied; the cache is evicted either way."""
        applied = self.cassandra_repo.update_snapshot_state_if_current(conversation_id, state, new_version, expected_version)
        await self.redis_repo.invalidate_conversation(conversation_id)
        return applied

    # ─── Summary Operations ──────────────────────────────────────────────────

    async def get_summary_record(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Summary text, version and watermark as ONE object, from one source.

        The cache entry holds all of them together, so a reader can never combine the text of one version with the
        watermark of another. Entries written before this change (plain text, no metadata) are treated as a miss and
        replaced by a record read from Cassandra."""
        cached = await self.redis_repo.get_summary_record(conversation_id)
        if cached:
            return cached
        row = self.cassandra_repo.get_summary(conversation_id)
        if not row or not (row.get("summary_text") or "").strip():
            return None
        record = {
            "text": row["summary_text"],
            "summary_version": int(row.get("summary_version") or 0),
            "through_message_id": row.get("through_message_id"),
            "through_created_at": row.get("through_created_at"),
        }
        await self.redis_repo.set_summary_record(conversation_id, record)
        return record

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
