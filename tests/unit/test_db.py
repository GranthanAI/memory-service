import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.db.redis import init_redis_pool, close_redis_pool, get_redis_client
from app.db.milvus import connect_milvus, disconnect_milvus, check_milvus_ready
from app.db.session import initialize_db_sessions, close_db_sessions

@pytest.mark.asyncio
async def test_db_session_initialization_success():
    """
    Verifies database session initialization when both Redis and Milvus are healthy.
    """
    mock_redis = MagicMock()
    mock_redis.ping = AsyncMock(return_value=True)
    
    with patch("app.db.redis.aioredis.Redis", return_value=mock_redis), \
         patch("app.db.redis.aioredis.ConnectionPool.from_url") as mock_redis_pool, \
         patch("app.db.milvus.connections.connect") as mock_milvus_connect, \
         patch("app.db.milvus.utility.list_collections", return_value=[]) as mock_list_col:
         
        # Execute initialization
        await initialize_db_sessions()
        
        # Verify Redis assertions
        mock_redis_pool.assert_called_once()
        mock_redis.ping.assert_called_once()
        
        # Verify Milvus assertions
        mock_milvus_connect.assert_called_once_with(
            alias="default",
            host="localhost",
            port=19530
        )
        mock_list_col.assert_called_once()
        
        # Close sessions
        with patch("app.db.redis.redis_pool.disconnect") as mock_redis_disconnect, \
             patch("app.db.milvus.connections.disconnect") as mock_milvus_disconnect:
            await close_db_sessions()
            mock_redis_disconnect.assert_called_once()
            mock_milvus_disconnect.assert_called_once_with("default")

@pytest.mark.asyncio
async def test_db_session_initialization_redis_failure():
    """
    Verifies that a Redis ping failure triggers cleanup and raises a RuntimeError.
    """
    mock_redis = MagicMock()
    mock_redis.ping = AsyncMock(side_effect=Exception("Redis connection timed out"))
    
    with patch("app.db.redis.aioredis.Redis", return_value=mock_redis), \
         patch("app.db.redis.aioredis.ConnectionPool.from_url"), \
         patch("app.db.milvus.connections.connect"), \
         patch("app.db.milvus.utility.list_collections", return_value=[]), \
         patch("app.db.session.close_db_sessions") as mock_cleanup:
         
        with pytest.raises(RuntimeError) as exc:
            await initialize_db_sessions()
            
        assert "Database session initialization failed" in str(exc.value)
        mock_cleanup.assert_called_once()

@pytest.mark.asyncio
async def test_db_session_initialization_milvus_failure():
    """
    Verifies that a Milvus connection failure triggers cleanup and raises a RuntimeError.
    """
    mock_redis = MagicMock()
    mock_redis.ping = AsyncMock(return_value=True)
    
    with patch("app.db.redis.aioredis.Redis", return_value=mock_redis), \
         patch("app.db.redis.aioredis.ConnectionPool.from_url"), \
         patch("app.db.milvus.connections.connect"), \
         patch("app.db.milvus.utility.list_collections", side_effect=Exception("Milvus service unavailable")), \
         patch("app.db.session.close_db_sessions") as mock_cleanup:
         
        with pytest.raises(RuntimeError) as exc:
            await initialize_db_sessions()
            
        assert "Database session initialization failed" in str(exc.value)
        mock_cleanup.assert_called_once()
