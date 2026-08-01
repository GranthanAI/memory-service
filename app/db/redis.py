import logging
from typing import Optional
import redis.asyncio as aioredis

logger = logging.getLogger("memory_service.db.redis")

# Global pool reference
redis_pool: Optional[aioredis.ConnectionPool] = None

def init_redis_pool(redis_url: str, max_connections: int = 100) -> aioredis.ConnectionPool:
    """
    Initializes the global Redis connection pool.
    """
    global redis_pool
    if redis_pool is None:
        logger.info(f"Initializing Redis connection pool on {redis_url}")
        redis_pool = aioredis.ConnectionPool.from_url(
            redis_url,
            decode_responses=True,
            max_connections=max_connections
        )
    return redis_pool

async def close_redis_pool() -> None:
    """
    Closes the global Redis connection pool gracefully.
    """
    global redis_pool
    if redis_pool is not None:
        logger.info("Closing Redis connection pool...")
        await redis_pool.disconnect()
        redis_pool = None

def get_redis_client() -> aioredis.Redis:
    """
    Creates an async Redis client from the global connection pool.
    """
    global redis_pool
    if redis_pool is None:
        raise RuntimeError("Redis connection pool is not initialized. Call init_redis_pool() first.")
    return aioredis.Redis(connection_pool=redis_pool)
