# Cassandra replication and consistency (memory-service)

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CASSANDRA_REPLICATION_STRATEGY` | `NetworkTopologyStrategy` | `NetworkTopologyStrategy` or `SimpleStrategy` |
| `CASSANDRA_LOCAL_DC` | `datacenter1` | Datacenter name; must match `nodetool status` / `SELECT data_center FROM system.local` |
| `CASSANDRA_REPLICATION_FACTOR` | unset | Unset = **3 in production**, 1 otherwise |
| `CASSANDRA_CONSISTENCY_READ` | `LOCAL_QUORUM` | Applied to every `SELECT` |
| `CASSANDRA_CONSISTENCY_WRITE` | `LOCAL_QUORUM` | Applied to writes and batches |
| `CASSANDRA_SERIAL_CONSISTENCY` | `LOCAL_SERIAL` | Lightweight transactions (`IF ...`) |

With `ENVIRONMENT=production` the service **refuses to start** unless the strategy is `NetworkTopologyStrategy` and the
RF is at least 3. Single-node environments must use `ENVIRONMENT=staging` or `development`.

The service creates the keyspace and tables on startup from this configuration (`python -m app.db.keyspace` prints the
statement). `CREATE KEYSPACE IF NOT EXISTS` never changes an existing keyspace, so clusters that
were created earlier with `SimpleStrategy` / RF=1 must be migrated by hand:

## Migrating an existing cluster (SimpleStrategy RF=1 to NetworkTopologyStrategy RF=3)

Prerequisites: at least 3 nodes in the datacenter, and capacity for 3x the data on disk. Do this in a quiet period.

1. Find the datacenter name: `nodetool status` (or `cqlsh -e "SELECT data_center FROM system.local"`).
2. Alter the keyspace (repeat for each application keyspace; `system_auth` should also be RF=3 if auth is enabled):
   ```cql
   ALTER KEYSPACE graphgpt_memory
     WITH replication = {'class': 'NetworkTopologyStrategy', '<dc-name>': 3};
   ```
3. **Repair so the new replicas receive the existing data.** On every node, one node at a time:
   ```
   nodetool repair -full graphgpt_memory
   ```
   Until the repair finishes, reads at `LOCAL_QUORUM` can miss data that so far exists only on the old replica, so run it
   promptly (and do not lower the read consistency to `LOCAL_ONE`, which can hit a still-empty new replica).
4. Verify: `DESCRIBE KEYSPACE graphgpt_memory` shows the new replication, and `nodetool status graphgpt_memory`
   shows "Owns (effective)" of about 300% across the datacenter.
5. Run `nodetool repair` regularly afterwards (at least once within `gc_grace_seconds`, default 10 days).

Rollback: `ALTER KEYSPACE ... {'<dc-name>': 1}` followed by `nodetool cleanup` on each node.
