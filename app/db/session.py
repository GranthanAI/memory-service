"""
app/db/session.py

Startup and shutdown orchestrator for all database connection pools.

Initialization Order (CRITICAL — do not reorder):
  1. Cassandra  — primary source of truth. Must be healthy first.
  2. Redis      — hot cache. Fails safe (cache misses fallback to Cassandra).
  3. Milvus     — vector index. Must be healthy for semantic retrieval.

If any pool fails its health check during startup, all pools are cleaned up
and a RuntimeError is raised with the name of the failing service, preventing
the application from entering a partially-initialized state.

Shutdown: Disconnects all three pools gracefully, in reverse order.
"""

import logging

from app.core.config import settings
from app.db.cassandra import (
    check_cassandra_ready,
    connect_cassandra,
    disconnect_cassandra,
)
from app.db.milvus import check_milvus_ready, connect_milvus, disconnect_milvus
from app.db.redis import close_redis_pool, get_redis_client, init_redis_pool

logger = logging.getLogger("memory_service.db.session")


async def initialize_db_sessions() -> None:
    """
    Initializes and health-checks all three database pools:
      Cassandra (primary store) → Redis (hot cache) → Milvus (vector index).

    Raises:
        RuntimeError: If any pool fails its health check. Includes the name
                      of the failing service for fast diagnosis.
    """
    logger.info("=== Database session initialization starting ===")

    cassandra_ready = False
    redis_ready = False
    milvus_ready = False

    # ── Step 1: Cassandra ────────────────────────────────────────────────────
    try:
        connect_cassandra()
        cassandra_ready = check_cassandra_ready()
        if cassandra_ready:
            logger.info("✓ Cassandra: connection OK")
        else:
            logger.critical("✗ Cassandra: health check returned False")
    except Exception as e:
        logger.critical(f"✗ Cassandra: connection failed — {e}")

    # ── Step 2: Redis ─────────────────────────────────────────────────────────
    try:
        init_redis_pool(settings.REDIS_URL)
        client = get_redis_client()
        await client.ping()
        redis_ready = True
        logger.info("✓ Redis: PING OK")
    except Exception as e:
        logger.critical(f"✗ Redis: connection failed — {e}")

    # ── Step 3: Milvus ────────────────────────────────────────────────────────
    try:
        connect_milvus(settings.MILVUS_HOST, settings.MILVUS_PORT)
        milvus_ready = check_milvus_ready()
        if milvus_ready:
            logger.info("✓ Milvus: connection OK")
        else:
            logger.critical("✗ Milvus: health check returned False")
    except Exception as e:
        logger.critical(f"✗ Milvus: connection failed — {e}")

    # ── Fail fast if any critical store is unavailable ────────────────────────
    if not cassandra_ready or not redis_ready or not milvus_ready:
        failed = [
            name
            for name, ready in [
                ("Cassandra", cassandra_ready),
                ("Redis", redis_ready),
                ("Milvus", milvus_ready),
            ]
            if not ready
        ]
        await close_db_sessions()
        raise RuntimeError(
            f"Database session initialization failed. "
            f"Failing services: {', '.join(failed)}"
        )

    logger.info("=== All database sessions initialized successfully ===")


async def close_db_sessions() -> None:
    """
    Gracefully disconnects all database pools in reverse initialization order:
      Milvus → Redis → Cassandra.

    Errors in individual pool shutdowns are logged but do not abort others.
    """
    logger.info("=== Closing database sessions ===")

    # ── Step 1: Milvus ────────────────────────────────────────────────────────
    try:
        disconnect_milvus()
        logger.info("✓ Milvus: disconnected")
    except Exception as e:
        logger.error(f"Error disconnecting Milvus: {e}")

    # ── Step 2: Redis ─────────────────────────────────────────────────────────
    try:
        await close_redis_pool()
        logger.info("✓ Redis: pool closed")
    except Exception as e:
        logger.error(f"Error closing Redis pool: {e}")

    # ── Step 3: Cassandra ─────────────────────────────────────────────────────
    try:
        disconnect_cassandra()
        logger.info("✓ Cassandra: disconnected")
    except Exception as e:
        logger.error(f"Error disconnecting Cassandra: {e}")

    logger.info("=== Database sessions closed ===")
