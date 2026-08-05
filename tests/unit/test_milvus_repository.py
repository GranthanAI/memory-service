"""
tests/unit/test_milvus_repository.py

Unit tests for Phase 10: Milvus Repository Layer.
Mocks pymilvus Collections and utilities to verify schema setup, inserts, search expressions,
and deletions.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.repositories.milvus_repository import MilvusRepository
from app.core.config import settings


@pytest.fixture
def mock_milvus():
    """Mocks standard PyMilvus utility and Collection classes."""
    with patch("app.repositories.milvus_repository.utility") as mock_utility, \
         patch("app.repositories.milvus_repository.Collection") as mock_collection_class:
        
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection
        yield mock_utility, mock_collection_class, mock_collection


def test_milvus_repository_initialization_existing_collection(mock_milvus):
    """Verifies that if the collection already exists, it is loaded to RAM without recreation."""
    mock_utility, mock_collection_class, mock_collection = mock_milvus
    mock_utility.has_collection.return_value = True

    repo = MilvusRepository(collection_name="test_collection")

    mock_utility.has_collection.assert_called_once_with("test_collection")
    mock_collection_class.assert_called_once_with("test_collection")
    mock_collection.load.assert_called_once()


def test_milvus_repository_initialization_new_collection(mock_milvus):
    """Verifies that a new collection schema and HNSW indexes are created if not existing."""
    mock_utility, mock_collection_class, mock_collection = mock_milvus
    mock_utility.has_collection.return_value = False

    repo = MilvusRepository(collection_name="new_collection")

    mock_utility.has_collection.assert_called_once_with("new_collection")
    mock_collection_class.assert_called_once()
    mock_collection.create_index.assert_called_once_with(
        field_name="vector",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 256}
        }
    )
    mock_collection.load.assert_called_once()


def test_milvus_insert_facts(mock_milvus):
    """Verifies bulk inserts partition data into schema-ordered column lists and splits into batches."""
    mock_utility, mock_collection_class, mock_collection = mock_milvus
    mock_utility.has_collection.return_value = True

    # Setup insert return value
    mock_res = MagicMock()
    mock_res.primary_keys = ["id-1", "id-2"]
    mock_collection.insert.return_value = mock_res

    repo = MilvusRepository(collection_name="test_collection")

    records = [
        {
            "fact_id": "id-1",
            "user_id": "user-a",
            "conversation_id": "conv-1",
            "category": "preferences",
            "statement": "Likes code",
            "importance": 0.9,
            "fact_version": 1,
            "embedding_ver": "v1.0",
            "created_at": 12345.67,
            "vector": [0.1] * settings.VECTOR_DIMENSION
        },
        {
            "fact_id": "id-2",
            "user_id": "user-a",
            "conversation_id": "conv-1",
            "category": "preferences",
            "statement": "Likes python",
            "importance": 0.85,
            "fact_version": 2,
            "embedding_ver": "v1.0",
            "created_at": 12345.68,
            "vector": [0.2] * settings.VECTOR_DIMENSION
        }
    ]

    inserted_ids = repo.insert_facts(records)

    assert inserted_ids == ["id-1", "id-2"]
    mock_collection.insert.assert_called_once()
    
    # Check that data list sent to insert is formatted as column lists
    inserted_args = mock_collection.insert.call_args[0][0]
    assert len(inserted_args) == 10  # 10 columns in schema
    assert inserted_args[0] == ["id-1", "id-2"]  # Primary key list
    assert inserted_args[1] == ["user-a", "user-a"]  # Partition key list
    mock_collection.flush.assert_called_once()


def test_milvus_search_facts(mock_milvus):
    """Verifies nearest-neighbor search sets correct parameters, filters, and dynamic partitions."""
    mock_utility, mock_collection_class, mock_collection = mock_milvus
    mock_utility.has_collection.return_value = True

    # Mock search response
    mock_hit1 = MagicMock()
    mock_hit1.score = 0.92
    mock_hit1.entity.get.side_effect = lambda f: {
        "fact_id": "id-1",
        "statement": "Likes code",
        "user_id": "user-a"
    }.get(f)

    mock_collection.search.return_value = [[mock_hit1]]

    repo = MilvusRepository(collection_name="test_collection")

    query_vec = [0.1] * settings.VECTOR_DIMENSION
    results = repo.search_facts(
        user_id="user-a",
        query_vector=query_vec,
        limit=5,
        category="preferences"
    )

    assert len(results) == 1
    assert results[0]["fact_id"] == "id-1"
    assert results[0]["distance"] == 0.92

    mock_collection.search.assert_called_once_with(
        data=[query_vec],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=5,
        expr="user_id == 'user-a' && category == 'preferences'",
        output_fields=[
            "fact_id",
            "user_id",
            "conversation_id",
            "category",
            "statement",
            "importance",
            "fact_version",
            "embedding_ver",
            "created_at"
        ],
        consistency_level="Bounded"
    )


def test_milvus_deletions(mock_milvus):
    """Verifies that deletions construct correct expressions conforming to Milvus primary-key delete constraints."""
    mock_utility, mock_collection_class, mock_collection = mock_milvus
    mock_utility.has_collection.return_value = True

    repo = MilvusRepository(collection_name="test_collection")

    # 1. Delete single fact - calls delete with pk expression
    repo.delete_fact(user_id="user-a", fact_id="id-1")
    mock_collection.delete.assert_called_with("fact_id in ['id-1']")

    # 2. Delete all user facts - queries user_id first, then deletes by pk list
    mock_collection.query.return_value = [{"fact_id": "id-1"}, {"fact_id": "id-2"}]
    repo.delete_user_facts(user_id="user-a")
    mock_collection.query.assert_called_once_with(
        expr="user_id == 'user-a'",
        output_fields=["fact_id"],
        consistency_level="Strong"
    )
    mock_collection.delete.assert_called_with('fact_id in ["id-1", "id-2"]')
