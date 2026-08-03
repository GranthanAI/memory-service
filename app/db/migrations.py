"""
app/db/migrations.py

Cassandra database migration system.
Handles schema version tracking, incremental schema updates, and startup validation.
"""

import logging
from datetime import datetime, timezone
from cassandra.query import SimpleStatement

from app.core.config import settings

logger = logging.getLogger("memory_service.db.migrations")

# Schema version table creation statement
_CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INT,
    description TEXT,
    applied_at TIMESTAMP,
    PRIMARY KEY (version)
);
"""


class Migration:
    """Base migration interface."""
    version: int
    description: str

    def upgrade(self, session) -> None:
        raise NotImplementedError()


class Migration1_InitialSchema(Migration):
    """Migration 1: Sets up the initial 8 core tables of the Memory Service."""
    version = 1
    description = "Initial core 8 tables creation"

    def upgrade(self, session) -> None:
        # Import the DDL statements from cassandra.py
        from app.db.cassandra import _DDL_STATEMENTS
        for i, ddl in enumerate(_DDL_STATEMENTS, start=1):
            logger.info(f"Applying initial core schema DDL statement {i}/{len(_DDL_STATEMENTS)}...")
            session.execute(SimpleStatement(ddl.strip()))


class Migration2_AddSnapshotMetadata(Migration):
    """Migration 2: Incremental update adding a snapshot_metadata column to conversation_snapshots."""
    version = 2
    description = "Add snapshot_metadata column to conversation_snapshots"

    def upgrade(self, session) -> None:
        logger.info("Applying Migration 2: Adding 'snapshot_metadata' column to conversation_snapshots table...")
        alter_statement = "ALTER TABLE conversation_snapshots ADD snapshot_metadata map<text, text>;"
        session.execute(SimpleStatement(alter_statement))


# List of all migrations in chronological order
MIGRATIONS: list[Migration] = [
    Migration1_InitialSchema(),
    Migration2_AddSnapshotMetadata(),
]


class MigrationManager:
    """Manages tracking, executing, and validating migrations in Cassandra."""

    def __init__(self, session):
        self.session = session

    def initialize_schema_version_table(self) -> None:
        """Bootstraps the schema version tracking table."""
        logger.info("Initializing schema version tracker table...")
        self.session.execute(SimpleStatement(_CREATE_SCHEMA_VERSION_TABLE))

    def get_current_version(self) -> int:
        """Retrieves the highest applied migration version from Cassandra."""
        query = "SELECT version FROM schema_version;"
        rows = self.session.execute(SimpleStatement(query))
        versions = [row.version for row in rows]
        return max(versions) if versions else 0

    def run_migrations(self) -> None:
        """Executes any pending migrations sequentially and records application status."""
        self.initialize_schema_version_table()
        current_version = self.get_current_version()
        logger.info(f"Current database schema version: V{current_version}")

        for migration in MIGRATIONS:
            if migration.version > current_version:
                logger.info(f"Running V{migration.version} migration: {migration.description}")
                try:
                    migration.upgrade(self.session)
                    
                    # Record the migration application status
                    record_statement = """
                    INSERT INTO schema_version (version, description, applied_at)
                    VALUES (%s, %s, %s);
                    """
                    self.session.execute(
                        SimpleStatement(record_statement),
                        (migration.version, migration.description, datetime.now(timezone.utc))
                    )
                    logger.info(f"✓ Migration V{migration.version} applied successfully.")
                except Exception as e:
                    logger.critical(f"✗ Migration V{migration.version} failed: {e}")
                    raise RuntimeError(f"Migration V{migration.version} failed: {e}") from e

    def validate_schema(self) -> None:
        """Validates that keyspaces, all tables, and core columns exist in the active Cassandra cluster."""
        logger.info("Starting schema verification and bootstrap validation...")
        keyspace_name = settings.CASSANDRA_KEYSPACE
        metadata = self.session.cluster.metadata
        
        keyspace = metadata.keyspaces.get(keyspace_name)
        if not keyspace:
            raise RuntimeError(f"Keyspace '{keyspace_name}' does not exist in cluster metadata.")

        # Map of expected table names and a list of crucial columns that must be present
        expected_tables = {
            "conversation_snapshots": ["conversation_id", "state", "message_count", "snapshot_metadata"],
            "conversation_summaries": ["conversation_id", "summary_text", "summary_version"],
            "processed_events": ["event_id", "conversation_id", "processed_at"],
            "outbox_jobs": ["job_id", "status", "topic", "payload"],
            "outbox_processing_index": ["claimed_date", "claimed_at", "job_id"],
            "retry_jobs": ["status", "next_retry", "job_id", "payload"],
            "user_facts": ["user_id", "category", "fact_id", "statement", "importance"],
            "conversation_recent_messages": ["conversation_id", "message_id", "content"]
        }

        for table_name, columns in expected_tables.items():
            table = keyspace.tables.get(table_name)
            if not table:
                raise RuntimeError(f"Validation Error: Expected table '{table_name}' is missing.")
            
            for col_name in columns:
                if col_name not in table.columns:
                    raise RuntimeError(
                        f"Validation Error: Column '{col_name}' is missing in table '{table_name}'."
                    )
        
        logger.info("✓ Schema validation check succeeded: all tables and columns verified.")
