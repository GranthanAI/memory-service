"""
tests/unit/test_migrations_and_container.py

Unit tests for Phase 21 Database Migration System and Phase 22 Dependency Injection Container.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.db.migrations import MigrationManager, Migration
from app.core.container import Container


class MockRow:
    def __init__(self, version):
        self.version = version


# ─── Migration System Tests ──────────────────────────────────────────────────

def test_migration_manager_get_current_version_empty():
    """get_current_version returns 0 if no migrations have been applied."""
    session = MagicMock()
    session.execute.return_value = []
    
    manager = MigrationManager(session)
    assert manager.get_current_version() == 0


def test_migration_manager_get_current_version_existing():
    """get_current_version returns the highest version applied."""
    session = MagicMock()
    session.execute.return_value = [MockRow(1), MockRow(2)]
    
    manager = MigrationManager(session)
    assert manager.get_current_version() == 2


def test_migration_manager_run_migrations_all_pending():
    """run_migrations applies all migrations when database is fresh."""
    session = MagicMock()
    
    # Simple side effect that avoids StopIteration for write statements
    def execute_side_effect(query, *args, **kwargs):
        if "SELECT version" in str(query):
            return []
        return []
    session.execute.side_effect = execute_side_effect
    
    manager = MigrationManager(session)
    
    # Track upgrades executed
    upgrades_called = []
    
    class TempMigration1(Migration):
        version = 1
        description = "Test 1"
        def upgrade(self, s) -> None:
            upgrades_called.append(1)
            
    class TempMigration2(Migration):
        version = 2
        description = "Test 2"
        def upgrade(self, s) -> None:
            upgrades_called.append(2)
            
    with patch("app.db.migrations.MIGRATIONS", [TempMigration1(), TempMigration2()]):
        manager.run_migrations()
        
    assert upgrades_called == [1, 2]
    # Check that execute was called at least 3 times (1 tracking table create, 2 run upgrades, 2 write applied entries)
    assert session.execute.call_count >= 3


def test_migration_manager_run_migrations_subset_pending():
    """run_migrations applies only pending migrations starting from the current version."""
    session = MagicMock()
    
    def execute_side_effect(query, *args, **kwargs):
        if "SELECT version" in str(query):
            return [MockRow(1)]
        return []
    session.execute.side_effect = execute_side_effect
    
    manager = MigrationManager(session)
    upgrades_called = []
    
    class TempMigration1(Migration):
        version = 1
        description = "Test 1"
        def upgrade(self, s) -> None:
            upgrades_called.append(1)
            
    class TempMigration2(Migration):
        version = 2
        description = "Test 2"
        def upgrade(self, s) -> None:
            upgrades_called.append(2)
            
    with patch("app.db.migrations.MIGRATIONS", [TempMigration1(), TempMigration2()]):
        manager.run_migrations()
        
    # Only V2 should be run
    assert upgrades_called == [2]


def test_validate_schema_success():
    """validate_schema passes if all expected tables and columns are present."""
    session = MagicMock()
    keyspace = MagicMock()
    
    # Configure mock tables
    tables = [
        "conversation_snapshots",
        "conversation_summaries",
        "processed_events",
        "outbox_jobs",
        "outbox_processing_index",
        "retry_jobs",
        "user_facts",
        "conversation_recent_messages"
    ]
    
    keyspace_tables = {}
    for t_name in tables:
        t_mock = MagicMock()
        t_mock.columns = [
            "conversation_id", "state", "message_count", "snapshot_metadata",
            "summary_text", "summary_version", "event_id", "processed_at",
            "job_id", "status", "topic", "payload", "claimed_date", "claimed_at",
            "next_retry", "user_id", "category", "fact_id", "statement", "importance", 
            "content", "message_id", "created_at"
        ]
        keyspace_tables[t_name] = t_mock
        
    keyspace.tables = keyspace_tables
    session.cluster.metadata.keyspaces = {"graphgpt_memory": keyspace}
    
    manager = MigrationManager(session)
    # Should not raise any error
    manager.validate_schema()


def test_validate_schema_missing_table():
    """validate_schema raises RuntimeError if a table is missing."""
    session = MagicMock()
    keyspace = MagicMock()
    
    # Omit "conversation_snapshots"
    keyspace.tables = {"conversation_summaries": MagicMock()}
    session.cluster.metadata.keyspaces = {"graphgpt_memory": keyspace}
    
    manager = MigrationManager(session)
    with pytest.raises(RuntimeError) as exc:
        manager.validate_schema()
    assert "Expected table" in str(exc.value)


def test_validate_schema_missing_column():
    """validate_schema raises RuntimeError if a table is missing a required column."""
    session = MagicMock()
    keyspace = MagicMock()
    
    t_mock = MagicMock()
    # Missing snapshot_metadata
    t_mock.columns = ["conversation_id", "state", "message_count"]
    
    keyspace.tables = {"conversation_snapshots": t_mock}
    session.cluster.metadata.keyspaces = {"graphgpt_memory": keyspace}
    
    manager = MigrationManager(session)
    with pytest.raises(RuntimeError) as exc:
        manager.validate_schema()
    assert "Column 'snapshot_metadata' is missing" in str(exc.value)


# ─── DI Container Tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_di_container_lifecycle():
    """DI container successfully connects, registers, wires providers, and shuts down resources."""
    container = Container()

    # Mock DB clients resolution
    with patch("app.core.container.get_session") as mock_cass, \
         patch("app.core.container.get_redis_client") as mock_redis, \
         patch("app.core.container.MilvusRepository") as mock_milvus:

        # Execute container bootstrap
        await container.init_resources()

        # Assert all singletons and wire connections exist
        assert container.cassandra_session is not None
        assert container.redis_client is not None
        assert container.milvus_repo is not None
        assert container.llm_service is not None
        assert container.graph_client is not None
        assert container.memory_service is not None
        assert container.summary_service is not None
        assert container.long_memory_service is not None
        assert container.retrieval_service is not None
        assert container.context_builder is not None

        # Execute container teardown
        await container.shutdown_resources()
