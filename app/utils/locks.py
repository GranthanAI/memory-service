"""
app/utils/locks.py

Distributed Redis locking utility with ownership validation and watchdog heartbeats.

Features:
  1. Token-based Locks: set(nx=True, ex=ttl) with a unique owner_token (UUID).
     Prevents accidental deletion of locks by slow workers who exceeded their TTL.
  2. Safe Releases: Lua script compare-and-delete ensures a client only deletes
     a lock if the token in Redis matches their own owner_token.
  3. Watchdog Heartbeat: RedisLockWatchdog runs as a background task to extend the
     TTL of the lock while the client is executing slow operations (e.g. LLM inference).
"""

import asyncio
import logging
import uuid
from typing import Optional

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger("memory_service.utils.locks")

# Lua script to release the lock atomically only if the value matches the owner token.
# Returns 1 if lock was successfully released, 0 otherwise.
UNLOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


async def acquire_redis_lock(
    client: aioredis.Redis,
    lock_key: str,
    ttl_seconds: Optional[int] = None,
    owner_token: Optional[str] = None,
) -> Optional[str]:
    """
    Acquires a distributed Redis lock using SETNX with a UUID ownership token.

    Args:
        client: The active async Redis client.
        lock_key: The string identifier for the resource lock.
        ttl_seconds: The lock TTL. Defaults to REDIS_LOCK_TTL_SECONDS.
        owner_token: Optional custom token string. Defaults to a new random UUID.

    Returns:
        The owner_token string if the lock was successfully acquired, None otherwise.
        The caller must keep this token to safely release the lock later.
    """
    ttl = ttl_seconds or settings.REDIS_LOCK_TTL_SECONDS
    token = owner_token or str(uuid.uuid4())
    try:
        # Set key if not exists (nx) with expiration (ex)
        acquired = await client.set(lock_key, token, ex=ttl, nx=True)
        if acquired:
            logger.debug(f"Acquired Redis lock: {lock_key} (owner={token})")
            return token
        return None
    except Exception as e:
        logger.error(f"Error acquiring Redis lock for key '{lock_key}': {e}")
        return None


async def release_redis_lock(
    client: aioredis.Redis,
    lock_key: str,
    owner_token: str,
) -> bool:
    """
    Releases a distributed Redis lock only if the owner_token matches.
    Uses an atomic Lua script compare-and-delete.

    Args:
        client: The active async Redis client.
        lock_key: The string identifier for the resource lock.
        owner_token: The ownership token returned when the lock was acquired.

    Returns:
        True if the lock was successfully deleted, False otherwise (e.g. lock expired
        or owned by another client).
    """
    try:
        result = await client.eval(UNLOCK_LUA, 1, lock_key, owner_token)
        released = bool(result)
        if released:
            logger.debug(f"Released Redis lock: {lock_key} (owner={owner_token})")
        else:
            logger.warning(
                f"Failed to release Redis lock '{lock_key}': owner token mismatch or expired."
            )
        return released
    except Exception as e:
        logger.error(f"Error releasing Redis lock for key '{lock_key}': {e}")
        return False


class RedisLockWatchdog:
    """
    Background worker that runs a heartbeat loop to extend the expiration TTL
    of an acquired Redis lock. Useful for preventing lock timeouts during
    unpredictable or long-running computations like LLM generations.
    """

    def __init__(
        self,
        client: aioredis.Redis,
        lock_key: str,
        owner_token: str,
        interval: Optional[float] = None,
        extend_by: Optional[int] = None,
    ):
        self.client = client
        self.lock_key = lock_key
        self.owner_token = owner_token
        self.interval = interval or settings.REDIS_LOCK_WATCHDOG_INTERVAL
        self.extend_by = extend_by or settings.REDIS_LOCK_TTL_SECONDS
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Starts the background watchdog loop."""
        self._task = asyncio.create_task(self._loop())
        logger.debug(f"Started Redis watchdog heartbeat for lock '{self.lock_key}'.")

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.interval)
                # Verify we still own the lock before extending it
                current_holder = await self.client.get(self.lock_key)
                if current_holder == self.owner_token:
                    await self.client.expire(self.lock_key, self.extend_by)
                    logger.debug(
                        f"Watchdog extended TTL for lock '{self.lock_key}' by {self.extend_by}s."
                    )
                else:
                    logger.warning(
                        f"Watchdog: lock '{self.lock_key}' ownership lost or expired. Stopping."
                    )
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error on key '{self.lock_key}': {e}")
                break

    async def stop(self) -> None:
        """Gracefully stops the background watchdog loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.debug(f"Stopped Redis watchdog heartbeat for lock '{self.lock_key}'.")
