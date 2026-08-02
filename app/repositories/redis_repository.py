"""
app/repositories/redis_repository.py

Hot cache repository layer utilizing Redis for sub-millisecond retrieval.
All summaries are zstd-compressed, snapshots are stored as hashes, and recent
messages are stored as sliding JSON lists.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis

from app.core.config import settings
from app.utils.compression import compress_string, decompress_to_string
from app.utils.serialization import from_json, to_json

logger = logging.getLogger("memory_service.repositories.redis_repository")


class RedisRepository:
    """
    Handles high-performance caching for conversation context.
    All cached objects automatically apply SNAPSHOT_TTL_SECONDS.
    """

    def __init__(self, client: aioredis.Redis):
        self.client = client
        self.ttl = settings.SNAPSHOT_TTL_SECONDS
        self.message_limit = settings.SHORT_TERM_MESSAGE_LIMIT

    def _snapshot_key(self, conversation_id: str) -> str:
        return f"snapshot:{conversation_id}"

    def _summary_key(self, conversation_id: str) -> str:
        return f"summary:{conversation_id}"

    def _recent_key(self, conversation_id: str) -> str:
        return f"recent:{conversation_id}"

    # ─── Snapshot Cache ──────────────────────────────────────────────────────

    async def get_snapshot(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the conversation snapshot hash from cache and casts values back to Python types.
        """
        key = self._snapshot_key(conversation_id)
        try:
            data = await self.client.hgetall(key)
            if not data:
                return None

            return {
                "conversation_id": data["conversation_id"],
                "user_id": data["user_id"],
                "message_count": int(data["message_count"]),
                "state": data["state"],
                "summary_version": int(data["summary_version"]),
                "fact_version": int(data["fact_version"]),
                "snapshot_version": int(data["snapshot_version"]),
                "last_summary_msg_id": data.get("last_summary_msg_id") or None,
                "updated_at": datetime.fromisoformat(data["updated_at"])
            }
        except Exception as e:
            logger.error(f"Error fetching snapshot cache for {conversation_id}: {e}")
            return None

    async def set_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Caches a conversation snapshot as a Redis Hash and updates its TTL.
        """
        conversation_id = snapshot["conversation_id"]
        key = self._snapshot_key(conversation_id)

        # Convert datetimes and None/missing fields to strings
        flat_hash = {
            "conversation_id": snapshot["conversation_id"],
            "user_id": snapshot["user_id"],
            "message_count": str(snapshot["message_count"]),
            "state": snapshot["state"],
            "summary_version": str(snapshot["summary_version"]),
            "fact_version": str(snapshot["fact_version"]),
            "snapshot_version": str(snapshot["snapshot_version"]),
            "last_summary_msg_id": snapshot.get("last_summary_msg_id") or "",
            "updated_at": (
                snapshot.get("updated_at") or datetime.now(timezone.utc)
            ).isoformat()
        }

        try:
            async with self.client.pipeline(transaction=True) as pipe:
                pipe.hset(key, mapping=flat_hash)
                pipe.expire(key, self.ttl)
                await pipe.execute()
        except Exception as e:
            logger.error(f"Error caching snapshot for {conversation_id}: {e}")

    # ─── Summary Cache ───────────────────────────────────────────────────────

    async def get_summary(self, conversation_id: str) -> Optional[str]:
        """
        Retrieves and decompresses (zstd) the conversation summary text from cache.
        """
        key = self._summary_key(conversation_id)
        try:
            compressed = await self.client.get(key)
            if not compressed:
                return None
            # Return decompressed string (Redis client returns bytes when set without decode_responses)
            # Since decode_responses is True on the connection pool, it might return string if we
            # stored it as string, but we compress it to bytes.
            # When bytes are retrieved from a connection pool with decode_responses=True, the driver
            # tries to decode it as utf-8, which would crash on binary zstd payloads!
            # CRITICAL CHECK: To store binary bytes, we must use a separate client or raw bytes.
            # However, the connection pool has decode_responses=True.
            # Let's verify: does hget/get crash on binary bytes if decode_responses=True?
            # Yes, if the bytes are not valid UTF-8.
            # To avoid this, we can store the zstd bytes as a hex string or base64 string,
            # OR we can retrieve a separate client with decode_responses=False.
            # Let's get a separate raw client for compressed binary operations, OR store it base64 encoded.
            # Storing as base64 is clean and works on any connection pool!
            # Let's store base64 encoded compressed bytes as a string. That is 100% safe.
            import base64
            decoded = base64.b64decode(compressed.encode("utf-8"))
            return decompress_to_string(decoded)
        except Exception as e:
            logger.error(f"Error fetching summary cache for {conversation_id}: {e}")
            return None

    async def set_summary(self, conversation_id: str, summary_text: str) -> None:
        """
        Compresses (zstd), base64-encodes, and caches the summary text.
        """
        key = self._summary_key(conversation_id)
        try:
            compressed = compress_string(summary_text)
            import base64
            encoded = base64.b64encode(compressed).decode("utf-8")
            await self.client.set(key, encoded, ex=self.ttl)
        except Exception as e:
            logger.error(f"Error caching summary for {conversation_id}: {e}")

    # ─── Recent Messages Cache ───────────────────────────────────────────────

    async def get_recent_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the time-ordered sliding window list from Redis (newest first).
        """
        key = self._recent_key(conversation_id)
        try:
            data = await self.client.lrange(key, 0, -1)
            return [from_json(item) for item in data]
        except Exception as e:
            logger.error(f"Error fetching recent messages cache for {conversation_id}: {e}")
            return []

    async def push_recent_message(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """
        Appends a message to the head of the Redis sliding list (newest first).
        Enforces the limit using LPUSH + LTRIM.
        """
        key = self._recent_key(conversation_id)
        serialized = to_json(message)
        try:
            async with self.client.pipeline(transaction=True) as pipe:
                pipe.lpush(key, serialized)
                pipe.ltrim(key, 0, self.message_limit - 1)
                pipe.expire(key, self.ttl)
                await pipe.execute()
        except Exception as e:
            logger.error(f"Error pushing recent message for {conversation_id}: {e}")

    async def set_recent_messages(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Hydrates the cache with a full list of messages.
        Useful on cache misses to reload historical rows from Cassandra.
        """
        key = self._recent_key(conversation_id)
        try:
            # We insert in reverse order so that LPUSH results in the correct order:
            # messages list is ordered newest first.
            # If we push from last to first (newest to oldest), the final Redis list
            # index 0 will be the first item pushed last (the newest message).
            # Let's verify:
            # messages = [msg_newest, msg_middle, msg_oldest]
            # If we push msg_oldest, then msg_middle, then msg_newest:
            # List state after push: [msg_newest, msg_middle, msg_oldest]
            # Yes! Pushing in reverse order of the list correctly reconstructs it!
            serialized_list = [to_json(m) for m in reversed(messages[:self.message_limit])]
            
            async with self.client.pipeline(transaction=True) as pipe:
                pipe.delete(key)
                if serialized_list:
                    pipe.rpush(key, *reversed(serialized_list)) # Rpush keeps the exact list order
                pipe.expire(key, self.ttl)
                await pipe.execute()
        except Exception as e:
            logger.error(f"Error setting recent messages cache for {conversation_id}: {e}")

    # ─── Cache Invalidation ──────────────────────────────────────────────────

    async def invalidate_conversation(self, conversation_id: str) -> None:
        """
        Deletes all cached keys associated with a conversation.
        Forces the next read to perform a read-through hydration from Cassandra.
        """
        snap_key = self._snapshot_key(conversation_id)
        sum_key = self._summary_key(conversation_id)
        rec_key = self._recent_key(conversation_id)
        try:
            await self.client.delete(snap_key, sum_key, rec_key)
            logger.debug(f"Invalidated Redis cache keys for conversation '{conversation_id}'.")
        except Exception as e:
            logger.error(f"Error invalidating cache for {conversation_id}: {e}")

    async def invalidate_context_cache(self, conversation_id: str) -> None:
        """Alias for invalidate_conversation matching low-level design specification."""
        await self.invalidate_conversation(conversation_id)
