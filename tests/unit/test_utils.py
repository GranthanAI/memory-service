import asyncio
import json
import pytest
from datetime import datetime
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel

from app.utils.compression import compress_string, decompress_to_string
from app.utils.locks import acquire_redis_lock, release_redis_lock
from app.utils.serialization import to_json, from_json
from app.utils.timers import Timer

# Mock Pydantic Model for testing serialization
class MockUser(BaseModel):
    id: UUID
    name: str
    created_at: datetime

def test_compression_and_decompression():
    """
    Verifies that text compression reduces space and decompresses correctly.
    """
    raw_text = "This is a long sample text to test zlib compression ratios. " * 50
    compressed = compress_string(raw_text)
    
    # Assert size is reduced
    assert len(compressed) < len(raw_text)
    
    # Assert decompressed text matches the original raw text
    decompressed = decompress_to_string(compressed)
    assert decompressed == raw_text
    
    # Empty inputs
    assert compress_string("") == b""
    assert decompress_to_string(b"") == ""

@pytest.mark.asyncio
async def test_redis_locks():
    """
    Asserts distributed Redis locks invoke NX keys and delete calls.
    """
    mock_redis = MagicMock()
    
    # Mock successful acquire
    mock_redis.set = AsyncMock(return_value=True)
    res_acquire = await acquire_redis_lock(mock_redis, "lock:123", ttl_seconds=10)
    
    assert res_acquire is True
    mock_redis.set.assert_called_once_with("lock:123", "1", ex=10, nx=True)
    
    # Mock release
    mock_redis.delete = AsyncMock(return_value=1)
    await release_redis_lock(mock_redis, "lock:123")
    mock_redis.delete.assert_called_once_with("lock:123")

def test_json_serialization():
    """
    Asserts JSON custom encoding processes complex datatypes successfully.
    """
    user_id = uuid4()
    now = datetime.now()
    user = MockUser(id=user_id, name="Alice", created_at=now)
    
    # Test serialization of pydantic model, datetime, uuid
    json_str = to_json(user)
    parsed = json.loads(json_str)
    
    assert parsed["id"] == str(user_id)
    assert parsed["name"] == "Alice"
    assert parsed["created_at"] == now.isoformat()
    
    # Test deserialization
    deserialized = from_json(json_str)
    assert deserialized["name"] == "Alice"
    assert deserialized["id"] == str(user_id)

@pytest.mark.asyncio
async def test_timer_context_manager():
    """
    Asserts execution timing captures correct delta duration metrics.
    """
    with Timer() as t:
        await asyncio.sleep(0.1)
    
    assert t.elapsed > 0.05
    assert t.elapsed < 0.2
