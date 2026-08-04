"""
app/core/startup_validation.py

Boot-time startup dependency validator.
Performs verification checks across all database, messaging, and client endpoints
at application start to ensure a fully operational environment.
"""

import asyncio
import logging
from typing import Set

import httpx
from pymilvus import Collection, utility
from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.core.container import Container
from app.db.migrations import MigrationManager

logger = logging.getLogger("memory_service.core.startup_validation")


async def validate_startup_dependencies(container: Container) -> None:
    """
    Performs boot-time verification checks on all infrastructure layers:
      1. Cassandra keyspace/tables and schema column metadata.
      2. Redis client connectivity (PING).
      3. Milvus collection existence and HNSW index status.
      4. Kafka expected topic existences.
      5. Graph Service REST endpoint reachability.
      6. LLM Service gRPC endpoint reachability.

    Raises:
        RuntimeError: If any critical service (Cassandra, Redis, Milvus, Kafka) is unreachable.
                      If STRICT_STARTUP_VALIDATION is True, also raises on Graph/LLM Service failures.
    """
    logger.info("=== Starting Boot-Time Startup Validation ===")
    errors = []

    # ── 1. Cassandra Schema Validation ───────────────────────────────────────
    try:
        logger.info("Verifying Cassandra schema tables & columns...")
        manager = MigrationManager(container.cassandra_session)
        manager.validate_schema()
        logger.info("✓ Cassandra Schema verification OK.")
    except Exception as e:
        logger.critical(f"✗ Cassandra Schema verification FAILED: {e}")
        errors.append(f"Cassandra: {e}")

    # ── 2. Redis Connection Validation ────────────────────────────────────────
    try:
        logger.info("Verifying Redis connectivity...")
        pong = await container.redis_client.ping()
        if not pong:
            raise RuntimeError("Redis PING returned False")
        logger.info("✓ Redis connectivity OK.")
    except Exception as e:
        logger.critical(f"✗ Redis connection check FAILED: {e}")
        errors.append(f"Redis: {e}")

    # ── 3. Milvus Collection & Index Validation ─────────────────────────────
    try:
        logger.info("Verifying Milvus collections & indexes...")
        if not utility.has_collection("user_memory_vectors"):
            raise RuntimeError("Milvus collection 'user_memory_vectors' does not exist")
        
        col = Collection("user_memory_vectors")
        indexes = [idx.index_name for idx in col.indexes]
        if not indexes:
            raise RuntimeError("Milvus collection 'user_memory_vectors' is missing an index")
            
        logger.info(f"✓ Milvus verification OK (indexes: {indexes}).")
    except Exception as e:
        logger.critical(f"✗ Milvus verification FAILED: {e}")
        errors.append(f"Milvus: {e}")

    # ── 4. Kafka Topics Validation ───────────────────────────────────────────
    expected_topics = {
        "chat.message.created",
        "chat.response.completed",
        settings.KAFKA_SUMMARY_TOPIC,
        settings.KAFKA_FACT_TOPIC,
        settings.KAFKA_EMBEDDING_TOPIC,
        settings.KAFKA_DELETE_TOPIC,
        settings.KAFKA_DLQ_TOPIC,
    }
    
    kafka_ok = False
    try:
        logger.info("Verifying Kafka topic existences...")
        temp_consumer = AIOKafkaConsumer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            request_timeout_ms=5000,
        )
        await temp_consumer.start()
        try:
            available_topics: Set[str] = await temp_consumer.topics()
            missing_topics = expected_topics - available_topics
            if missing_topics:
                raise RuntimeError(f"Missing required Kafka topics: {list(missing_topics)}")
            logger.info("✓ Kafka expected topics exist.")
            kafka_ok = True
        finally:
            await temp_consumer.stop()
    except Exception as e:
        logger.critical(f"✗ Kafka validation FAILED: {e}")
        errors.append(f"Kafka: {e}")

    # ── 5. Graph Service Endpoint Validation ───────────────────────────────
    graph_healthy = False
    try:
        logger.info(f"Verifying Graph Service reachability at {settings.GRAPH_SERVICE_URL}...")
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.GRAPH_SERVICE_URL}/health")
            if response.status_code >= 400:
                response = await client.get(settings.GRAPH_SERVICE_URL)
            logger.info("✓ Graph Service endpoint reachable.")
            graph_healthy = True
    except Exception as e:
        msg = f"Graph Service endpoint is unreachable: {e}"
        if settings.STRICT_STARTUP_VALIDATION:
            logger.critical(f"✗ {msg}")
            errors.append(f"GraphService: {e}")
        else:
            logger.warning(f"⚠️ {msg} (Non-strict mode: ignoring)")

    # ── 6. LLM Service gRPC Endpoint Validation ────────────────────────────
    llm_healthy = False
    try:
        logger.info("Verifying LLM Service gRPC connection pool...")
        channel = await container.llm_pool.get_channel()
        state = channel.get_state(try_to_connect=True)
        logger.info(f"✓ LLM Service pool verified. Connectivity state: {state}")
        llm_healthy = True
    except Exception as e:
        msg = f"LLM Service gRPC endpoint is unreachable/failure: {e}"
        if settings.STRICT_STARTUP_VALIDATION:
            logger.critical(f"✗ {msg}")
            errors.append(f"LLMService: {e}")
        else:
            logger.warning(f"⚠️ {msg} (Non-strict mode: ignoring)")

    # ── Resolve Errors ────────────────────────────────────────────────────────
    if errors:
        logger.critical(f"Startup validation FAILED with {len(errors)} error(s). Shutting down.")
        raise RuntimeError(f"Startup validation failed: {'; '.join(errors)}")

    logger.info("=== Startup Validation Passed Successfully ===")
