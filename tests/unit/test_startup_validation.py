"""
tests/unit/test_startup_validation.py

Unit tests for app/core/startup_validation.py.
All connection pools, external services, and clients are mocked.
Verifies correct validation behaviour under various status states and configuration modes.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import settings
from app.core.container import Container
from app.core.startup_validation import validate_startup_dependencies


# ─── Mock Helpers ─────────────────────────────────────────────────────────────

def make_mock_container():
    container = MagicMock(spec=Container)
    
    # Cassandra mock
    container.cassandra_session = MagicMock()
    
    # Redis mock
    container.redis_client = AsyncMock()
    container.redis_client.ping.return_value = True
    
    # LLM pool mock
    container.llm_pool = AsyncMock()
    mock_channel = MagicMock()
    mock_channel.get_state.return_value = "READY"
    container.llm_pool.get_channel.return_value = mock_channel
    
    return container


@pytest.fixture(autouse=True)
def reset_strict_setting():
    original = settings.STRICT_STARTUP_VALIDATION
    yield
    settings.STRICT_STARTUP_VALIDATION = original


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("app.core.startup_validation.MigrationManager")
@patch("app.core.startup_validation.utility")
@patch("app.core.startup_validation.Collection")
@patch("app.core.startup_validation.AIOKafkaConsumer")
@patch("app.core.startup_validation.httpx.AsyncClient")
async def test_validate_all_healthy(
    mock_http_client,
    mock_consumer_cls,
    mock_collection_cls,
    mock_utility,
    mock_migration_manager_cls
):
    """Verify happy path: all checks pass and no exception is raised."""
    container = make_mock_container()
    
    # Cassandra mock OK
    mock_manager = MagicMock()
    mock_migration_manager_cls.return_value = mock_manager
    
    # Milvus mock OK
    mock_utility.has_collection.return_value = True
    mock_col = MagicMock()
    mock_idx = MagicMock()
    mock_idx.index_name = "vector_index"
    mock_col.indexes = [mock_idx]
    mock_collection_cls.return_value = mock_col
    
    # Kafka mock OK
    mock_consumer = AsyncMock()
    mock_consumer.topics.return_value = {
        "chat.message.created",
        "chat.response.completed",
        settings.KAFKA_SUMMARY_TOPIC,
        settings.KAFKA_FACT_TOPIC,
        settings.KAFKA_EMBEDDING_TOPIC,
        settings.KAFKA_DELETE_TOPIC,
        settings.KAFKA_DLQ_TOPIC,
    }
    mock_consumer_cls.return_value = mock_consumer
    
    # Graph Service mock OK
    mock_client_instance = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client_instance.get.return_value = mock_resp
    mock_http_client.return_value.__aenter__.return_value = mock_client_instance
    
    # Run validator
    await validate_startup_dependencies(container)
    
    # Assertions
    mock_manager.validate_schema.assert_called_once()
    container.redis_client.ping.assert_called_once()
    mock_utility.has_collection.assert_called_once_with("user_memory_vectors")
    mock_consumer.topics.assert_called_once()


@pytest.mark.asyncio
@patch("app.core.startup_validation.MigrationManager")
@patch("app.core.startup_validation.utility")
@patch("app.core.startup_validation.Collection")
@patch("app.core.startup_validation.AIOKafkaConsumer")
@patch("app.core.startup_validation.httpx.AsyncClient")
async def test_validate_cassandra_fails(
    mock_http_client,
    mock_consumer_cls,
    mock_collection_cls,
    mock_utility,
    mock_migration_manager_cls
):
    """Verify that a Cassandra validation failure raises a RuntimeError."""
    container = make_mock_container()
    
    mock_manager = MagicMock()
    mock_manager.validate_schema.side_effect = RuntimeError("Missing table 'outbox_jobs'")
    mock_migration_manager_cls.return_value = mock_manager
    
    mock_utility.has_collection.return_value = True
    mock_col = MagicMock()
    mock_col.indexes = [MagicMock()]
    mock_collection_cls.return_value = mock_col
    
    mock_consumer = AsyncMock()
    mock_consumer.topics.return_value = {"chat.message.created"}
    mock_consumer_cls.return_value = mock_consumer
    
    with pytest.raises(RuntimeError, match="Startup validation failed: Cassandra: Missing table"):
        await validate_startup_dependencies(container)


@pytest.mark.asyncio
@patch("app.core.startup_validation.MigrationManager")
@patch("app.core.startup_validation.utility")
@patch("app.core.startup_validation.Collection")
@patch("app.core.startup_validation.AIOKafkaConsumer")
@patch("app.core.startup_validation.httpx.AsyncClient")
async def test_validate_redis_fails(
    mock_http_client,
    mock_consumer_cls,
    mock_collection_cls,
    mock_utility,
    mock_migration_manager_cls
):
    """Verify that a Redis ping check failure raises a RuntimeError."""
    container = make_mock_container()
    container.redis_client.ping.return_value = False
    
    mock_utility.has_collection.return_value = True
    mock_col = MagicMock()
    mock_col.indexes = [MagicMock()]
    mock_collection_cls.return_value = mock_col
    
    mock_consumer = AsyncMock()
    mock_consumer.topics.return_value = {"chat.message.created"}
    mock_consumer_cls.return_value = mock_consumer
    
    with pytest.raises(RuntimeError, match="Startup validation failed: Redis: Redis PING returned False"):
        await validate_startup_dependencies(container)


@pytest.mark.asyncio
@patch("app.core.startup_validation.MigrationManager")
@patch("app.core.startup_validation.utility")
@patch("app.core.startup_validation.Collection")
@patch("app.core.startup_validation.AIOKafkaConsumer")
@patch("app.core.startup_validation.httpx.AsyncClient")
async def test_validate_milvus_collection_missing(
    mock_http_client,
    mock_consumer_cls,
    mock_collection_cls,
    mock_utility,
    mock_migration_manager_cls
):
    """Verify that a missing Milvus collection raises a RuntimeError."""
    container = make_mock_container()
    
    mock_utility.has_collection.return_value = False
    
    mock_consumer = AsyncMock()
    mock_consumer.topics.return_value = {"chat.message.created"}
    mock_consumer_cls.return_value = mock_consumer
    
    with pytest.raises(RuntimeError, match="Milvus collection 'user_memory_vectors' does not exist"):
        await validate_startup_dependencies(container)


@pytest.mark.asyncio
@patch("app.core.startup_validation.MigrationManager")
@patch("app.core.startup_validation.utility")
@patch("app.core.startup_validation.Collection")
@patch("app.core.startup_validation.AIOKafkaConsumer")
@patch("app.core.startup_validation.httpx.AsyncClient")
async def test_validate_milvus_index_missing(
    mock_http_client,
    mock_consumer_cls,
    mock_collection_cls,
    mock_utility,
    mock_migration_manager_cls
):
    """Verify that a missing vector index raises a RuntimeError."""
    container = make_mock_container()
    
    mock_utility.has_collection.return_value = True
    mock_col = MagicMock()
    mock_col.indexes = []  # No index created
    mock_collection_cls.return_value = mock_col
    
    mock_consumer = AsyncMock()
    mock_consumer.topics.return_value = {"chat.message.created"}
    mock_consumer_cls.return_value = mock_consumer
    
    with pytest.raises(RuntimeError, match="user_memory_vectors' is missing an index"):
        await validate_startup_dependencies(container)


@pytest.mark.asyncio
@patch("app.core.startup_validation.MigrationManager")
@patch("app.core.startup_validation.utility")
@patch("app.core.startup_validation.Collection")
@patch("app.core.startup_validation.AIOKafkaConsumer")
@patch("app.core.startup_validation.httpx.AsyncClient")
async def test_validate_kafka_topics_missing(
    mock_http_client,
    mock_consumer_cls,
    mock_collection_cls,
    mock_utility,
    mock_migration_manager_cls
):
    """Verify that missing Kafka topics raise a RuntimeError."""
    container = make_mock_container()
    
    mock_utility.has_collection.return_value = True
    mock_col = MagicMock()
    mock_col.indexes = [MagicMock()]
    mock_collection_cls.return_value = mock_col
    
    mock_consumer = AsyncMock()
    # Kafka only has chat.message.created, missing others
    mock_consumer.topics.return_value = {"chat.message.created"}
    mock_consumer_cls.return_value = mock_consumer
    
    with pytest.raises(RuntimeError, match="Missing required Kafka topics"):
        await validate_startup_dependencies(container)


@pytest.mark.asyncio
@patch("app.core.startup_validation.MigrationManager")
@patch("app.core.startup_validation.utility")
@patch("app.core.startup_validation.Collection")
@patch("app.core.startup_validation.AIOKafkaConsumer")
@patch("app.core.startup_validation.httpx.AsyncClient")
async def test_validate_downstream_fails_non_strict(
    mock_http_client,
    mock_consumer_cls,
    mock_collection_cls,
    mock_utility,
    mock_migration_manager_cls
):
    """Verify that in non-strict mode (default), downstream failures are log-only and do not raise."""
    container = make_mock_container()
    settings.STRICT_STARTUP_VALIDATION = False
    
    mock_utility.has_collection.return_value = True
    mock_col = MagicMock()
    mock_col.indexes = [MagicMock()]
    mock_collection_cls.return_value = mock_col
    
    mock_consumer = AsyncMock()
    mock_consumer.topics.return_value = {
        "chat.message.created",
        "chat.response.completed",
        settings.KAFKA_SUMMARY_TOPIC,
        settings.KAFKA_FACT_TOPIC,
        settings.KAFKA_EMBEDDING_TOPIC,
        settings.KAFKA_DELETE_TOPIC,
        settings.KAFKA_DLQ_TOPIC,
    }
    mock_consumer_cls.return_value = mock_consumer
    
    # Graph Service fails
    mock_client_instance = AsyncMock()
    mock_client_instance.get.side_effect = Exception("Graph service down")
    mock_http_client.return_value.__aenter__.return_value = mock_client_instance
    
    # LLM pool fails
    container.llm_pool.get_channel.side_effect = Exception("LLM service unreachable")
    
    # Should not raise exception
    await validate_startup_dependencies(container)


@pytest.mark.asyncio
@patch("app.core.startup_validation.MigrationManager")
@patch("app.core.startup_validation.utility")
@patch("app.core.startup_validation.Collection")
@patch("app.core.startup_validation.AIOKafkaConsumer")
@patch("app.core.startup_validation.httpx.AsyncClient")
async def test_validate_downstream_fails_strict(
    mock_http_client,
    mock_consumer_cls,
    mock_collection_cls,
    mock_utility,
    mock_migration_manager_cls
):
    """Verify that in strict mode, downstream service failures raise a RuntimeError."""
    container = make_mock_container()
    settings.STRICT_STARTUP_VALIDATION = True
    
    mock_utility.has_collection.return_value = True
    mock_col = MagicMock()
    mock_col.indexes = [MagicMock()]
    mock_collection_cls.return_value = mock_col
    
    mock_consumer = AsyncMock()
    mock_consumer.topics.return_value = {
        "chat.message.created",
        "chat.response.completed",
        settings.KAFKA_SUMMARY_TOPIC,
        settings.KAFKA_FACT_TOPIC,
        settings.KAFKA_EMBEDDING_TOPIC,
        settings.KAFKA_DELETE_TOPIC,
        settings.KAFKA_DLQ_TOPIC,
    }
    mock_consumer_cls.return_value = mock_consumer
    
    # Graph Service fails
    mock_client_instance = AsyncMock()
    mock_client_instance.get.side_effect = Exception("Graph service down")
    mock_http_client.return_value.__aenter__.return_value = mock_client_instance
    
    with pytest.raises(RuntimeError, match="Startup validation failed: GraphService: Graph service down"):
        await validate_startup_dependencies(container)
