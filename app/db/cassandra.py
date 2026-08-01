"""
app/db/cassandra.py

Cassandra connection lifecycle and keyspace/table initialization.

Cassandra is the PRIMARY SOURCE OF TRUTH for this service.
All persistent state (snapshots, summaries, facts, outbox, idempotency,
recent messages) is written to Cassandra first. Redis is a hot cache only.

Connection Strategy:
  - Uses cassandra-driver's Cluster object with configurable contact points.
  - Connection pool is synchronous (cassandra-driver uses its own thread pool internally).
  - Session is bound to the configured keyspace.
  - On first connect, all 8 CQL tables are created (idempotent DDL with IF NOT EXISTS).

Tables Created (matching LLD §3):
  1. conversation_snapshots        — lightweight state metadata
  2. conversation_summaries        — versioned AI-generated summaries
  3. processed_events              — idempotency registry (7-day TTL)
  4. outbox_jobs                   — async task queue for Kafka publishing
  5. outbox_processing_index       — efficient stale PROCESSING row reaping
  6. retry_jobs                    — failed job retry scheduling
  7. user_facts                    — long-term structured user memory
  8. conversation_recent_messages  — durable short-term message window backup
"""

import logging
from typing import List, Optional

from cassandra.cluster import Cluster, Session
from cassandra.policies import DCAwareRoundRobinPolicy, RetryPolicy
from cassandra.query import SimpleStatement

from app.core.config import settings

logger = logging.getLogger("memory_service.db.cassandra")

# Global references — initialized once during startup
_cluster: Optional[Cluster] = None
_session: Optional[Session] = None


# ─── CQL DDL Statements ───────────────────────────────────────────────────────

_CREATE_KEYSPACE = """
CREATE KEYSPACE IF NOT EXISTS {keyspace}
WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 3}}
AND durable_writes = true;
"""

_DDL_STATEMENTS = [
    # Table 1: Lightweight Conversation Snapshots
    # Stores metadata state only — no raw message content.
    # Recent messages are in conversation_recent_messages (below).
    """
    CREATE TABLE IF NOT EXISTS conversation_snapshots (
        conversation_id     TEXT,
        user_id             TEXT,
        message_count       INT,
        state               TEXT,
        summary_version     INT,
        fact_version        INT,
        snapshot_version    INT,
        last_summary_msg_id TEXT,
        updated_at          TIMESTAMP,
        PRIMARY KEY (conversation_id)
    );
    """,

    # Table 2: Conversation Summaries (versioned)
    """
    CREATE TABLE IF NOT EXISTS conversation_summaries (
        conversation_id     TEXT,
        summary_text        TEXT,
        summary_version     INT,
        model_name          TEXT,
        model_version       TEXT,
        generated_at        TIMESTAMP,
        PRIMARY KEY (conversation_id)
    );
    """,

    # Table 3: Idempotency Registry
    # 7-day TTL covers Kafka replay window. Prevents duplicate event processing.
    """
    CREATE TABLE IF NOT EXISTS processed_events (
        event_id            TEXT,
        conversation_id     TEXT,
        processed_at        TIMESTAMP,
        PRIMARY KEY (event_id)
    ) WITH default_time_to_live = 604800;
    """,

    # Table 4: Outbox Jobs
    # Partitioned by status for efficient PENDING row polling.
    # LWT is used to transition PENDING → PROCESSING atomically.
    """
    CREATE TABLE IF NOT EXISTS outbox_jobs (
        job_id              UUID,
        status              TEXT,
        topic               TEXT,
        conversation_id     TEXT,
        payload             TEXT,
        attempt_count       INT,
        last_error          TEXT,
        created_at          TIMESTAMP,
        claimed_at          TIMESTAMP,
        PRIMARY KEY (status, created_at, job_id)
    ) WITH CLUSTERING ORDER BY (created_at ASC, job_id ASC);
    """,

    # Table 5: Outbox Processing Index
    # Avoids ALLOW FILTERING on outbox_jobs for stale PROCESSING reaping.
    # Cleanup worker queries this table by (claimed_date, claimed_at) range.
    """
    CREATE TABLE IF NOT EXISTS outbox_processing_index (
        claimed_date        TEXT,
        claimed_at          TIMESTAMP,
        job_id              UUID,
        PRIMARY KEY ((claimed_date), claimed_at, job_id)
    ) WITH CLUSTERING ORDER BY (claimed_at ASC, job_id ASC);
    """,

    # Table 6: Retry Jobs
    # Tracks failed background jobs for retry scheduling and DLQ.
    # max_retry is per-job-type configurable (summary=5, embedding=2, etc.)
    """
    CREATE TABLE IF NOT EXISTS retry_jobs (
        status              TEXT,
        next_retry          TIMESTAMP,
        job_id              UUID,
        job_type            TEXT,
        payload             TEXT,
        retry_count         INT,
        max_retry           INT,
        last_error          TEXT,
        created_at          TIMESTAMP,
        PRIMARY KEY ((status), next_retry, job_id)
    ) WITH CLUSTERING ORDER BY (next_retry ASC, job_id ASC);
    """,

    # Table 7: User Facts (Long-Term Memory)
    # Partitioned by (user_id, category) for fast category-scoped lookups.
    """
    CREATE TABLE IF NOT EXISTS user_facts (
        user_id             TEXT,
        category            TEXT,
        fact_id             UUID,
        conversation_id     TEXT,
        statement           TEXT,
        importance          FLOAT,
        fact_version        INT,
        embedding_version   TEXT,
        created_at          TIMESTAMP,
        updated_at          TIMESTAMP,
        PRIMARY KEY ((user_id, category), fact_id)
    ) WITH CLUSTERING ORDER BY (fact_id ASC);
    """,

    # Table 8: Recent Message Window (Short-Term Memory Durable Backup)
    # Redis is the hot cache. This is the durable fallback.
    # Addresses the Kafka retention gap: Kafka retention = 7 days,
    # but conversations can be months old.
    # On Redis miss → read last N rows here → repopulate Redis.
    """
    CREATE TABLE IF NOT EXISTS conversation_recent_messages (
        conversation_id     TEXT,
        message_id          TEXT,
        role                TEXT,
        content             TEXT,
        created_at          TIMESTAMP,
        PRIMARY KEY (conversation_id, created_at, message_id)
    ) WITH CLUSTERING ORDER BY (created_at DESC, message_id ASC);
    """,
]


# ─── Lifecycle Functions ──────────────────────────────────────────────────────

def connect_cassandra() -> None:
    """
    Creates the global Cassandra Cluster + Session.
    Parses CASSANDRA_HOSTS (comma-separated for multi-node).
    Creates keyspace and all tables on first connect (idempotent DDL).
    """
    global _cluster, _session

    contact_points: List[str] = [
        h.strip() for h in settings.CASSANDRA_HOSTS.split(",") if h.strip()
    ]
    logger.info(f"Connecting to Cassandra at {contact_points}:{settings.CASSANDRA_PORT}")

    _cluster = Cluster(
        contact_points=contact_points,
        port=settings.CASSANDRA_PORT,
        load_balancing_policy=DCAwareRoundRobinPolicy(),
        default_retry_policy=RetryPolicy(),
        connect_timeout=settings.CASSANDRA_TIMEOUT_SECONDS,
        protocol_version=4,
    )

    # Connect without keyspace to create it if it doesn't exist
    system_session = _cluster.connect()
    try:
        logger.info(f"Creating keyspace '{settings.CASSANDRA_KEYSPACE}' if not exists...")
        system_session.execute(
            SimpleStatement(
                _CREATE_KEYSPACE.format(keyspace=settings.CASSANDRA_KEYSPACE)
            )
        )
    finally:
        system_session.shutdown()

    # Bind session to the keyspace
    _session = _cluster.connect(settings.CASSANDRA_KEYSPACE)
    logger.info(f"Cassandra session bound to keyspace '{settings.CASSANDRA_KEYSPACE}'.")

    _apply_schema(_session)
    logger.info("Cassandra schema initialization complete.")


def _apply_schema(session: Session) -> None:
    """
    Executes all DDL statements to create tables if they don't exist.
    Safe to run on every startup (IF NOT EXISTS makes it idempotent).
    """
    for i, ddl in enumerate(_DDL_STATEMENTS, start=1):
        logger.debug(f"Applying DDL statement {i}/{len(_DDL_STATEMENTS)}...")
        session.execute(SimpleStatement(ddl.strip()))


def disconnect_cassandra() -> None:
    """
    Gracefully shuts down the Cassandra session and cluster connection.
    """
    global _cluster, _session

    if _session is not None:
        logger.info("Shutting down Cassandra session...")
        _session.shutdown()
        _session = None

    if _cluster is not None:
        logger.info("Shutting down Cassandra cluster connection...")
        _cluster.shutdown()
        _cluster = None


def get_session() -> Session:
    """
    Returns the global Cassandra Session for repository injection.
    Raises RuntimeError if connect_cassandra() has not been called.
    """
    if _session is None:
        raise RuntimeError(
            "Cassandra session is not initialized. "
            "Call connect_cassandra() during application startup."
        )
    return _session


def check_cassandra_ready() -> bool:
    """
    Performs a lightweight health check by querying system.local.
    Used by session.py during startup to verify connectivity.
    """
    try:
        if _session is None:
            return False
        _session.execute("SELECT now() FROM system.local")
        return True
    except Exception as e:
        logger.error(f"Cassandra health check failed: {e}")
        return False
