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


class Migration3_AddCreatedAtToProcessingIndex(Migration):
    """Migration 3: Add created_at column to outbox_processing_index.

    The CleanupWorker needs created_at from the index to issue targeted
    LWT UPDATE on outbox_jobs (which has PRIMARY KEY (status, created_at, job_id))
    without needing ALLOW FILTERING on the main table.
    """
    version = 3
    description = "Add created_at column to outbox_processing_index"

    def upgrade(self, session) -> None:
        logger.info(
            "Applying Migration 3: Adding 'created_at' column to outbox_processing_index..."
        )
        session.execute(SimpleStatement(
            "ALTER TABLE outbox_processing_index ADD created_at TIMESTAMP;"
        ))


class Migration4_UserMemoryControls(Migration):
    """Migration 4: user-facing memory controls - per-user memory on/off
    (user_memory_settings) and tombstones for deleted/corrected memories
    (user_fact_tombstones). Both CREATE TABLE IF NOT EXISTS, so re-running is safe."""
    version = 4
    description = "Add user_memory_settings and user_fact_tombstones tables"

    def upgrade(self, session) -> None:
        from app.db.cassandra import _USER_CONTROLS_DDL
        for ddl in _USER_CONTROLS_DDL:
            session.execute(SimpleStatement(ddl.strip()))


class Migration5_SummaryWatermark(Migration):
    """Migration 5: the summary row carries its own watermark (the last message it incorporates:
    through_message_id + through_created_at). Summary text, version and watermark are then written by ONE
    conditional update, so they can never disagree. Existing rows get NULL watermarks (= unknown: the next
    summary treats all retained messages as new once, as before). Re-running is safe."""
    version = 5
    description = "Add through_message_id / through_created_at to conversation_summaries"

    def upgrade(self, session) -> None:
        for column, cql_type in (("through_message_id", "text"), ("through_created_at", "timestamp")):
            try:
                session.execute(SimpleStatement(f"ALTER TABLE conversation_summaries ADD {column} {cql_type}"))
            except Exception as exc:  # column already present (re-run / fresh keyspace created with it)
                if "already exist" not in str(exc).lower() and "conflicts with an existing column" not in str(exc).lower():
                    raise


class Migration6_SummaryMetadata(Migration):
    """Migration 6: summaries record which prompt produced them (`prompt_version`) and which provider served the model
    (`llm_provider`); `model_version` used to hold the provider name by mistake. Additive and nullable: old rows read as
    "unknown". Re-running is safe."""
    version = 6
    description = "Add prompt_version / llm_provider to conversation_summaries"

    def upgrade(self, session) -> None:
        for column, cql_type in (("prompt_version", "text"), ("llm_provider", "text")):
            try:
                session.execute(SimpleStatement(f"ALTER TABLE conversation_summaries ADD {column} {cql_type}"))
            except Exception as exc:  # column already present (re-run / fresh keyspace created with it)
                if "already exist" not in str(exc).lower() and "conflicts with an existing column" not in str(exc).lower():
                    raise


class Migration7_SummaryVersions(Migration):
    """Migration 7: immutable history of conversation summaries. `conversation_summaries` keeps ONE row per conversation
    (the current summary); once a parent was summarized past a branch point its children lost the parent summary for
    good. Every superseded summary is archived here, keyed (conversation_id, summary_version), and never updated.
    New table only: nothing existing is touched. Re-running is safe."""
    version = 7
    description = "Create conversation_summary_versions"

    def upgrade(self, session) -> None:
        session.execute(SimpleStatement("""
            CREATE TABLE IF NOT EXISTS conversation_summary_versions (
                conversation_id    TEXT,
                summary_version    INT,
                summary_text       TEXT,
                through_message_id TEXT,
                through_created_at TIMESTAMP,
                model_name         TEXT,
                model_version      TEXT,
                llm_provider       TEXT,
                prompt_version     TEXT,
                generated_at       TIMESTAMP,
                PRIMARY KEY (conversation_id, summary_version)
            ) WITH CLUSTERING ORDER BY (summary_version DESC)
        """))


class Migration8_FactLifecycle(Migration):
    """Migration 8: facts are retired, not deleted. `status` (NULL / 'ACTIVE' = active, 'SUPERSEDED'), `superseded_by`
    (the fact that replaced it), `valid_to` (when it stopped being true), `last_confirmed` (last time the user said it again)
    and `last_used` (last time it was injected into an answer). All nullable and additive: existing rows are active facts
    with unknown confirmation time. Re-running is safe."""
    version = 8
    description = "Add status/superseded_by/valid_to/last_confirmed/last_used to user_facts"

    def upgrade(self, session) -> None:
        for column, cql_type in (("status", "text"), ("superseded_by", "uuid"), ("valid_to", "timestamp"),
                                 ("last_confirmed", "timestamp"), ("last_used", "timestamp")):
            try:
                session.execute(SimpleStatement(f"ALTER TABLE user_facts ADD {column} {cql_type}"))
            except Exception as exc:  # column already present (re-run / fresh keyspace created with it)
                if "already exist" not in str(exc).lower() and "conflicts with an existing column" not in str(exc).lower():
                    raise


class Migration9_AnchorLedger(Migration):
    """Migration 9: the anchor ledger. Facts that must survive summarization (identifiers, decisions with reasons, constraints,
    open items, corrections), one row per anchor with the message it came from. New table only; nothing existing is touched."""
    version = 9
    description = "Create conversation_anchors"

    def upgrade(self, session) -> None:
        session.execute(SimpleStatement("""
            CREATE TABLE IF NOT EXISTS conversation_anchors (
                conversation_id   TEXT,
                anchor_id         TEXT,
                kind              TEXT,
                anchor_text       TEXT,
                source_message_id TEXT,
                status            TEXT,
                created_at        TIMESTAMP,
                PRIMARY KEY (conversation_id, anchor_id)
            )
        """))


# List of all migrations in chronological order
MIGRATIONS: list[Migration] = [
    Migration1_InitialSchema(),
    Migration2_AddSnapshotMetadata(),
    Migration3_AddCreatedAtToProcessingIndex(),
    Migration4_UserMemoryControls(),
    Migration5_SummaryWatermark(),
    Migration6_SummaryMetadata(),
    Migration7_SummaryVersions(),
    Migration8_FactLifecycle(),
    Migration9_AnchorLedger(),
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
            "conversation_summaries": ["conversation_id", "summary_text", "summary_version", "through_message_id", "through_created_at", "prompt_version", "llm_provider"],
            "conversation_summary_versions": ["conversation_id", "summary_version", "summary_text", "through_message_id"],
            "conversation_anchors": ["conversation_id", "anchor_id", "kind", "anchor_text", "status"],
            "processed_events": ["event_id", "conversation_id", "processed_at"],
            "outbox_jobs": ["job_id", "status", "topic", "payload"],
            "outbox_processing_index": ["claimed_date", "claimed_at", "job_id", "created_at"],
            "retry_jobs": ["status", "next_retry", "job_id", "payload"],
            "user_facts": ["user_id", "category", "fact_id", "statement", "importance"],
            "conversation_recent_messages": ["conversation_id", "message_id", "content"],
            "user_memory_settings": ["user_id", "memory_enabled"],
            "user_fact_tombstones": ["user_id", "statement_hash"],
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
