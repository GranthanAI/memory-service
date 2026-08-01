import logging
import redis.asyncio as aioredis

logger = logging.getLogger("memory_service.utils.locks")

async def acquire_redis_lock(client: aioredis.Redis, lock_key: str, ttl_seconds: int = 5) -> bool:
    """
    Acquires a non-blocking distributed lock using SETNX.
    Returns True if successfully acquired, False otherwise.
    """
    try:
        # Set key if not exists (nx) with expiration (ex)
        res = await client.set(lock_key, "1", ex=ttl_seconds, nx=True)
        return bool(res)
    except Exception as e:
        logger.error(f"Error acquiring Redis lock for key '{lock_key}': {str(e)}")
        return False

async def release_redis_lock(client: aioredis.Redis, lock_key: str) -> None:
    """
    Releases a distributed lock by deleting the lock key.
    """
    try:
        await client.delete(lock_key)
    except Exception as e:
        logger.error(f"Error releasing Redis lock for key '{lock_key}': {str(e)}")
