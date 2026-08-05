"""
tests/unit/test_long_memory_service.py

Unit tests for Phase 14 User Fact Merging Logic (Fact Merge Policy).
Mocks repository layers to assert correct skipping of exact statement matches,
ignores of lower importance facts, and supersedes of higher importance facts.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from app.services.long_memory_service import LongMemoryService
from app.core.config import settings


@pytest.fixture
def mock_repos():
    """Constructs mock Cassandra and Milvus repositories."""
    cassandra_repo = MagicMock()
    milvus_repo = MagicMock()

    cassandra_repo.get_facts = MagicMock(return_value=[])
    cassandra_repo.upsert_fact = MagicMock()
    cassandra_repo.delete_fact = MagicMock()

    milvus_repo.search_facts = MagicMock(return_value=[])
    milvus_repo.insert_facts = MagicMock(return_value=[])
    milvus_repo.delete_fact = MagicMock()

    return cassandra_repo, milvus_repo


@pytest.mark.asyncio
async def test_merge_user_facts_skips_on_exact_match(mock_repos):
    """Asserts that facts with exact statement matches are skipped immediately."""
    cassandra_repo, milvus_repo = mock_repos
    service = LongMemoryService(cassandra_repo, milvus_repo)

    # Seed existing Cassandra facts
    existing_fact = {
        "user_id": "user-123",
        "category": "preferences",
        "fact_id": uuid.uuid4(),
        "statement": "User likes espresso."
    }
    cassandra_repo.get_facts.return_value = [existing_fact]

    incoming_facts = [
        {
            "statement": "user likes espresso.",  # case-insensitive match
            "category": "preferences",
            "importance": 0.9,
            "vector": [0.1] * settings.VECTOR_DIMENSION
        }
    ]

    stats = await service.merge_user_facts(
        user_id="user-123",
        conversation_id="conv-1",
        incoming_facts=incoming_facts
    )

    assert stats["skipped"] == 1
    assert stats["inserted"] == 0
    assert stats["superseded"] == 0

    # No writes should have been made
    cassandra_repo.upsert_fact.assert_not_called()
    milvus_repo.insert_facts.assert_not_called()


@pytest.mark.asyncio
async def test_merge_user_facts_inserts_on_low_similarity(mock_repos):
    """Asserts that facts with low similarity are directly inserted as new facts."""
    cassandra_repo, milvus_repo = mock_repos
    service = LongMemoryService(cassandra_repo, milvus_repo)

    cassandra_repo.get_facts.return_value = []
    
    # Mock Milvus search returning a distant match (distance < threshold of 0.85)
    milvus_repo.search_facts.return_value = [
        {
            "fact_id": str(uuid.uuid4()),
            "statement": "User plays tennis.",
            "importance": 0.5,
            "fact_version": 1,
            "distance": 0.6  # Low similarity
        }
    ]

    incoming_facts = [
        {
            "statement": "User likes espresso.",
            "category": "preferences",
            "importance": 0.8,
            "vector": [0.1] * settings.VECTOR_DIMENSION
        }
    ]

    stats = await service.merge_user_facts(
        user_id="user-123",
        conversation_id="conv-1",
        incoming_facts=incoming_facts
    )

    assert stats["inserted"] == 1
    assert stats["skipped"] == 0
    assert stats["superseded"] == 0

    # Assert writes are triggered with version = 1
    cassandra_repo.upsert_fact.assert_called_once()
    cassandra_record = cassandra_repo.upsert_fact.call_args[0][0]
    assert cassandra_record["statement"] == "User likes espresso."
    assert cassandra_record["fact_version"] == 1

    milvus_repo.insert_facts.assert_called_once()


@pytest.mark.asyncio
async def test_merge_user_facts_ignores_on_lower_importance(mock_repos):
    """Asserts that high similarity matches are ignored if new importance is lower."""
    cassandra_repo, milvus_repo = mock_repos
    service = LongMemoryService(cassandra_repo, milvus_repo)

    # Mock Milvus search returning a close match with HIGHER importance
    existing_fact_id = str(uuid.uuid4())
    milvus_repo.search_facts.return_value = [
        {
            "fact_id": existing_fact_id,
            "statement": "User drinks dark roast coffee.",
            "importance": 0.9,  # Higher importance
            "fact_version": 1,
            "distance": 0.92  # High similarity
        }
    ]

    incoming_facts = [
        {
            "statement": "User likes dark roast coffee.",
            "category": "preferences",
            "importance": 0.7,  # Lower importance
            "vector": [0.1] * settings.VECTOR_DIMENSION
        }
    ]

    stats = await service.merge_user_facts(
        user_id="user-123",
        conversation_id="conv-1",
        incoming_facts=incoming_facts
    )

    assert stats["ignored"] == 1
    assert stats["inserted"] == 0
    assert stats["superseded"] == 0

    # No writes/deletes should be executed
    cassandra_repo.upsert_fact.assert_not_called()
    cassandra_repo.delete_fact.assert_not_called()
    milvus_repo.delete_fact.assert_not_called()


@pytest.mark.asyncio
async def test_merge_user_facts_supersedes_on_higher_importance(mock_repos):
    """Asserts that close matches with higher/equal importance supersede the old facts."""
    cassandra_repo, milvus_repo = mock_repos
    service = LongMemoryService(cassandra_repo, milvus_repo)

    # Mock Milvus search returning a close match with LOWER importance
    existing_fact_id = uuid.uuid4()
    milvus_repo.search_facts.return_value = [
        {
            "fact_id": str(existing_fact_id),
            "statement": "User lives in a flat in London.",
            "importance": 0.6,  # Lower importance
            "fact_version": 2,
            "distance": 0.88  # High similarity
        }
    ]

    incoming_facts = [
        {
            "statement": "User lives in a house in London.",
            "category": "preferences",
            "importance": 0.9,  # Higher importance
            "vector": [0.1] * settings.VECTOR_DIMENSION
        }
    ]

    stats = await service.merge_user_facts(
        user_id="user-123",
        conversation_id="conv-1",
        incoming_facts=incoming_facts
    )

    assert stats["superseded"] == 1
    assert stats["inserted"] == 0

    # Verify old fact deletion
    cassandra_repo.delete_fact.assert_called_once_with("user-123", "preferences", existing_fact_id)
    milvus_repo.delete_fact.assert_called_once_with("user-123", str(existing_fact_id))

    # Verify new fact insertion with version incremented (2 + 1 = 3)
    cassandra_repo.upsert_fact.assert_called_once()
    new_record = cassandra_repo.upsert_fact.call_args[0][0]
    assert new_record["statement"] == "User lives in a house in London."
    assert new_record["fact_version"] == 3
    assert new_record["fact_id"] != existing_fact_id

    milvus_repo.insert_facts.assert_called_once()
