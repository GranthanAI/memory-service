# Task Update 5: Database Migration System & Dependency Injection Container

This update documents the technical implementation of **Phase 21: Database Migration System** and **Phase 22: Dependency Injection Container** for the GraphGPT Memory Service. It covers the Cassandra version tracker, incremental schema scripts, metadata validators, IOC container wiring, FastAPI lifespan integration, and unit/integration test results.

---

## 1. Executive Summary

With the database repositories, state machine, pipeline workers, and retry daemons implemented, establishing robust database version control and centralized dependency wiring is critical to support local, staging, and production deployments. 

This phase delivers:
* An incremental, tracking-based **Cassandra schema migration manager** that records schema updates in a dedicated metadata table.
* A boot-time **schema validator** using Cassandra cluster metadata to verify tables and columns.
* A native **Inversion of Control (IOC) container** providing registered singletons and client pools to decouple database drivers from business services.
* Core FastAPI **lifespan startup/shutdown orchestrations** in `app/lifespan.py` and `app/main.py`.

The entire suite of 133 tests passed successfully. No regressions were introduced.

---

## 2. Phase 21: Database Migration System

To avoid untracked DDL updates and simplify environment bootstraps, we designed a programmatic Cassandra migration engine located in [migrations.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/db/migrations.py).

### 2.1 Schema Version Tracker Table
We introduced a tracking table `schema_version` which durably logs applied migrations:
```sql
CREATE TABLE IF NOT EXISTS schema_version (
    version INT,
    description TEXT,
    applied_at TIMESTAMP,
    PRIMARY KEY (version)
);
```

### 2.2 Incremental Migration Scripts
Migrations are represented as instances of the `Migration` class containing an incremental version index and description.

1. **Migration V1 (Initial Schema)**:
   Migrates the fresh keyspace and builds the 8 core tables:
   * `conversation_snapshots`: Holds conversation metadata and versions.
   * `conversation_summaries`: Stores generated text summaries of conversations.
   * `processed_events`: Keeps record of event IDs processed to prevent duplicates.
   * `outbox_jobs`: Queue of outbox jobs pending delivery to Kafka.
   * `outbox_processing_index`: Index mapping of claimed outbox jobs for outbox worker claiming.
   * `retry_jobs`: Jobs needing automatic retry under failure.
   * `user_facts`: Durable user-specific long-term cognitive statements.
   * `conversation_recent_messages`: Rolling window of raw recent conversation messages.
   
2. **Migration V2 (Incremental Update)**:
   Alters the `conversation_snapshots` table to add a map metadata column:
   ```sql
   ALTER TABLE conversation_snapshots ADD snapshot_metadata map<text, text>;
   ```

### 2.3 Bootstrap Schema Metadata Validator
On startup, after applying any pending migrations, the migration manager checks cluster metadata to ensure all tables exist and possess the required keys:
```python
keyspace = metadata.keyspaces.get(keyspace_name)
for table_name, columns in expected_tables.items():
    table = keyspace.tables.get(table_name)
    for col_name in columns:
        if col_name not in table.columns:
            raise RuntimeError(f"Column '{col_name}' missing in '{table_name}'")
```

---

## 3. Phase 22: Dependency Injection Container

We introduced a native, thread-safe, and asynchronous Dependency Injection (DI) Container under [container.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/core/container.py) to centralize all singleton providers, client pools, and business wiring.

### 3.1 Wiring Map Diagram
```text
                       [FastAPI lifespan]
                               │
                               ▼
                        [IOC Container]
                               │
      ┌────────────────────────┼────────────────────────┐
      ▼                        ▼                        ▼
[gRPC LLM Pool]         [Redis Client]         [Cassandra Session]
      │                        │                        │
      ▼                        └──────────┬─────────────┘
  LLMClient                               │
                                          ▼
                                   MemoryRepository
                                          │
                                          ▼
                                    MemoryService
```

### 3.2 Registered Components
The DI container orchestrates dependencies into three main groups: Connection clients, repository abstractions, and core business services.
* **Connection Clients**:
  * `AsyncGRPCConnectionPool` (LLM pool)
  * `LLMClient`
  * `GraphClient` (Graph Database service communication)
  * `Redis` connection client (hot cache)
  * Cassandra session (primary source of truth)
* **Repositories**:
  * `CassandraRepository`: Handles low-level Cassandra DML read and write operations.
  * `RedisRepository`: Handles Hot Cache read, write, expiration, and lock management.
  * `ProcessedEventRepository`: Manages idempotency markers.
  * `MemoryRepository`: Wraps Cassandra and Redis repositories for read-through lookup.
  * `MilvusRepository`: Handles semantic vector storage and similarities.
* **Services**:
  * `SnapshotService`: Sequenced batch updates to snapshots, windows, events, and outbox.
  * `MemoryService`: Coordinates state machine transition boundaries.
  * `SummaryService`: Orchestrates incremental text summarization via LLM gRPC calls.
  * `LongMemoryService`: Implements Fact Merge Policy and updates vector indexes.
  * `RankingService`: Scores retrieved facts using cosine similarity, recency decay, and importance.
  * `RetrievalService`: Handles cache read-through and DB fallback queries.
  * `ContextBuilder`: Gathers short-term rolling messages, summaries, lineage, and scored long-term facts.

---

## 4. Lifespan and API Entry Points

The dependency injection and migration manager are bootstrapped via the FastAPI application lifespan.

### 4.1 Application Lifespan (`lifespan.py`)
Wired in [lifespan.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/lifespan.py):
* **Startup Hook**:
  1. Calls `initialize_db_sessions()` to bootstrap Cassandra cluster, connect Redis, and Milvus.
  2. Runs migrations using `MigrationManager` and validates schema column integrity.
  3. Spawns `Container` and runs `container.init_resources()` to configure connections, LLM channel pools, and wire services.
  4. Saves container singleton on `app.state.container`.
* **Shutdown Hook**:
  1. Invokes `container.shutdown_resources()` to close active gRPC channel pools.
  2. Calls `close_db_sessions()` to release Cassandra, Redis, and Milvus pools.

### 4.2 Application Main Entry (`main.py`)
Instantiates FastAPI in [main.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/app/main.py):
```python
app = FastAPI(
    title="GraphGPT Memory Service",
    description="Derived Cognitive AI Memory Engine",
    version="4.0",
    lifespan=lifespan
)
```

---

## 5. Verification & Test Logs

To verify the migration manager and DI container, we wrote unit tests in [test_migrations_and_container.py](file:///c:/Users/hp/Desktop/Granthan/memory-service/tests/unit/test_migrations_and_container.py):
* **V1 Fresh Migrations**: Asserts all core tables are created correctly.
* **V2 Incremental Updates**: Asserts alter queries run correctly and write the V2 applied tracking marker.
* **Schema Column Validator**: Mocks cluster metadata to check column mismatch and missing tables.
* **Container Lifecycle**: Asserts resources bootstrap and wire up correctly without failing.

### 5.1 Test Outputs
All tests pass cleanly:
```text
tests/unit/test_migrations_and_container.py::test_migration_manager_get_current_version_empty PASSED
tests/unit/test_migrations_and_container.py::test_migration_manager_get_current_version_existing PASSED
tests/unit/test_migrations_and_container.py::test_migration_manager_run_migrations_all_pending PASSED
tests/unit/test_migrations_and_container.py::test_migration_manager_run_migrations_subset_pending PASSED
tests/unit/test_validate_schema_success PASSED
tests/unit/test_validate_schema_missing_table PASSED
tests/unit/test_validate_schema_missing_column PASSED
tests/unit/test_di_container_lifecycle PASSED

============================== 8 passed in 4.77s ==============================
```

Additionally, we ran full regression test suite covering all 133 tests:
* Unit tests (state machine, workers, outbox, scoring, repositories): **Passed**
* Integration tests (Cassandra, Redis snapshot cache, Milvus index, worker pipelines): **Passed**

```text
================ 133 passed, 111 warnings in 131.62s (0:02:12) =================
```

---

## 6. Table Layout Definitions

To provide a complete layout of Cassandra schemas, we outline table columns expected under the version tracker:

### 6.1 `conversation_snapshots`
Used to manage short-term context states, count rolling thresholds, and track versions.
* `conversation_id`: `TEXT` (Partition Key)
* `user_id`: `TEXT`
* `message_count`: `INT`
* `state`: `TEXT`
* `summary_version`: `INT`
* `fact_version`: `INT`
* `snapshot_version`: `INT`
* `last_summary_msg_id`: `TEXT`
* `updated_at`: `TIMESTAMP`
* `snapshot_metadata`: `map<text, text>` (Added in Migration V2)

### 6.2 `conversation_summaries`
Used to persist generated text summaries.
* `conversation_id`: `TEXT` (Partition Key)
* `summary_text`: `TEXT`
* `summary_version`: `INT`
* `model_name`: `TEXT`
* `model_version`: `TEXT`
* `generated_at`: `TIMESTAMP`

### 6.3 `processed_events`
Stores event IDs processed by consumers to guarantee idempotency.
* `event_id`: `TEXT` (Partition Key)
* `conversation_id`: `TEXT`
* `processed_at`: `TIMESTAMP`

### 6.4 `outbox_jobs`
Durable jobs scheduled to publish to Kafka using transactional outbox pattern.
* `job_id`: `UUID` (Partition Key)
* `conversation_id`: `TEXT`
* `topic`: `TEXT`
* `payload`: `TEXT`
* `status`: `TEXT`
* `created_at`: `TIMESTAMP`
* `error_message`: `TEXT`

### 6.5 `outbox_processing_index`
Indexes pending outbox jobs by bucket keys to prevent scanning entire jobs table.
* `claimed_date`: `TEXT` (Partition Key)
* `claimed_at`: `TIMESTAMP` (Clustering Key)
* `job_id`: `UUID` (Clustering Key)

### 6.6 `retry_jobs`
Failed Kafka payload jobs marked for retry.
* `status`: `TEXT` (Partition Key)
* `next_retry`: `TIMESTAMP` (Clustering Key)
* `job_id`: `UUID` (Clustering Key)
* `payload`: `TEXT`
* `retry_count`: `INT`
* `max_retry`: `INT`
* `last_error`: `TEXT`
* `created_at`: `TIMESTAMP`

### 6.7 `user_facts`
Extracted cognitive facts of users for long-term memory.
* `user_id`: `TEXT` (Partition Key)
* `category`: `TEXT` (Clustering Key)
* `fact_id`: `UUID` (Clustering Key)
* `statement`: `TEXT`
* `importance`: `FLOAT`
* `created_at`: `TIMESTAMP`

### 6.8 `conversation_recent_messages`
Rolling message window of active conversations.
* `conversation_id`: `TEXT` (Partition Key)
* `created_at`: `TIMESTAMP` (Clustering Key)
* `message_id`: `TEXT` (Clustering Key)
* `role`: `TEXT`
* `content`: `TEXT`

---

## 7. Client & Pool Configurations

To ensure highly reliable service routing under load, we configure client pools inside the container:

### 7.1 gRPC Connection Pool
The LLM Inference client uses `AsyncGRPCConnectionPool` to persist multiple persistent channels, ensuring low-latency communication with the LLM Service:
* Maintains a pool of active gRPC channels.
* Validates channel health at regular configurable intervals.
* Closes channels gracefully on teardown inside `shutdown_resources()`.

### 7.2 Redis Client & Cache Expirations
The Redis cache is registered as a singleton inside the container to provide extremely fast, temporary cache lookups:
* Connects using connection pool string.
* Rebuilds snapshots and recent sliding windows automatically under cache miss.
* Invalidates the keys via cache invalidation hooks.

### 7.3 Milvus Collection & Index Setup
Milvus handles long-term memory semantic search queries:
* Collection schema contains columns: `fact_id`, `user_id`, `category`, and `vector`.
* Utilizes L2 metric distance or IP distance index.
* Loads collection data partitions into active query nodes.

---

## 8. Operational Runbook

When booting the application inside a fresh environment:
1. Ensure the Cassandra nodes are active.
2. Initialize environment settings (hosts, ports, keyspace names) in `.env` file.
3. The lifespan manager automatically connects to the cluster on boot.
4. `MigrationManager` executes `initialize_schema_version_table()` to track applied migration scripts.
5. If table has missing rows, pending scripts upgrade database to match version `2`.
6. Validator runs checks to confirm table columns.
7. Any schema anomalies trigger a startup crash to avoid corrupt states.

All changes have been successfully committed. The service is ready for deployment.
