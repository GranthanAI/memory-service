import logging
import asyncio
from app.core.config import settings
from app.db.redis import init_redis_pool, close_redis_pool, get_redis_client
from app.db.milvus import connect_milvus, disconnect_milvus, check_milvus_ready

logger = logging.getLogger("memory_service.db.session")

async def initialize_db_sessions() -> None:
    """
    Initializes and verifies Redis and Milvus connection pools during application startup.
    Raises RuntimeError if a critical connection check fails.
    """
    logger.info("Initializing database connections...")
    
    # 1. Initialize Redis Pool
    init_redis_pool(settings.REDIS_URL)
    
    # Verify Redis connectivity
    redis_ready = False
    try:
        client = get_redis_client()
        await client.ping()
        redis_ready = True
        logger.info("Redis connection check successful.")
    except Exception as e:
        logger.critical(f"Redis connection failed: {str(e)}")
        
    # 2. Initialize Milvus Connection
    milvus_ready = False
    try:
        connect_milvus(settings.MILVUS_HOST, settings.MILVUS_PORT)
        milvus_ready = check_milvus_ready()
        if milvus_ready:
            logger.info("Milvus connection check successful.")
    except Exception as e:
        logger.critical(f"Milvus connection failed: {str(e)}")

    if not redis_ready or not milvus_ready:
        # Close anything that was partially initialized
        await close_db_sessions()
        raise RuntimeError(
            f"Database session initialization failed. Redis: {'ready' if redis_ready else 'failed'}, "
            f"Milvus: {'ready' if milvus_ready else 'failed'}"
        )

    logger.info("Database session initialization complete.")

async def close_db_sessions() -> None:
    """
    Tears down Redis and Milvus connection pools during application shutdown.
    """
    logger.info("Closing database sessions...")
    
    # 1. Close Redis Pool
    try:
        await close_redis_pool()
    except Exception as e:
        logger.error(f"Error closing Redis pool: {str(e)}")
        
    # 2. Disconnect Milvus
    try:
        disconnect_milvus()
    except Exception as e:
        logger.error(f"Error disconnecting Milvus: {str(e)}")
        
    logger.info("Database sessions closed.")
