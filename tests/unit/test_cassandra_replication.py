"""Keyspace replication/consistency configuration: generated CQL, production fail-fast, read/write consistency."""
import pytest
from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement

from app.core.config import SystemSettings as Settings
from app.db.cassandra import ConsistencySession
from app.db.keyspace import effective_replication_factor, keyspace_cql

STRONG = "k9Qx2mV7pLr4Zt8WyB3nHd6Fs1JcUe5A"
PROD = dict(ENVIRONMENT="production", API_KEY=STRONG, JWT_SECRET_KEY=STRONG, USER_JWT_SECRET_KEY=STRONG, DEBUG=False)


def build(**over):
    return Settings(_env_file=None, **over)


# ── generated CQL ────────────────────────────────────────────────────────────

def test_network_topology_cql_uses_the_datacenter_name():
    cql = keyspace_cql("graphgpt_memory", "NetworkTopologyStrategy", "dc-eu-1", 3)
    assert "'class': 'NetworkTopologyStrategy'" in cql and "'dc-eu-1': 3" in cql and "SimpleStrategy" not in cql


def test_simple_strategy_cql_for_development():
    assert "'replication_factor': 1" in keyspace_cql("ks", "SimpleStrategy", "dc", 1)


def test_keyspace_name_is_validated():
    with pytest.raises(ValueError):
        keyspace_cql("bad name; DROP", "NetworkTopologyStrategy", "dc", 3)


def test_default_rf_is_3_in_production_and_1_elsewhere():
    assert effective_replication_factor(None, "production") == 3
    assert effective_replication_factor(None, "development") == 1
    assert effective_replication_factor(2, "production") == 2


# ── fail-fast ────────────────────────────────────────────────────────────────

def test_production_defaults_are_valid():
    build(**PROD)  # NetworkTopologyStrategy, RF unset -> 3


@pytest.mark.parametrize("over", [
    dict(CASSANDRA_REPLICATION_STRATEGY="SimpleStrategy", CASSANDRA_REPLICATION_FACTOR=3),
    dict(CASSANDRA_REPLICATION_FACTOR=1),
    dict(CASSANDRA_REPLICATION_FACTOR=2),
])
def test_production_refuses_simple_strategy_or_rf_below_3(over):
    with pytest.raises(ValueError) as exc:
        build(**PROD, **over)
    assert "ENVIRONMENT=staging or development" in str(exc.value)  # says how to fix it


@pytest.mark.parametrize("env", ["development", "staging", "test"])
def test_single_node_settings_are_fine_outside_production(env):
    build(ENVIRONMENT=env, CASSANDRA_REPLICATION_STRATEGY="SimpleStrategy", CASSANDRA_REPLICATION_FACTOR=1)


def test_invalid_consistency_level_is_rejected():
    with pytest.raises(ValueError):
        build(CASSANDRA_CONSISTENCY_READ="QUORUMM")


def test_defaults_are_local_quorum():
    s = build()
    assert s.CASSANDRA_CONSISTENCY_READ == s.CASSANDRA_CONSISTENCY_WRITE == "LOCAL_QUORUM"


# ── read/write consistency ───────────────────────────────────────────────────

class FakePrepared:
    consistency_level = None

    def __init__(self, q):
        self.query_string = q


class FakeSession:
    def __init__(self):
        self.executed = []

    def prepare(self, q):
        return FakePrepared(q)

    def execute(self, query, *a, **k):
        self.executed.append(query)
        return "rs"

    def execute_async(self, query, *a, **k):
        self.executed.append(query)
        return "fut"

    keyspace = "ks"


def test_selects_get_the_read_consistency_and_writes_keep_the_profile_default():
    raw = FakeSession()
    s = ConsistencySession(raw, ConsistencyLevel.LOCAL_ONE)
    s.execute("SELECT * FROM t")
    s.execute("INSERT INTO t (a) VALUES (1)")
    select, insert = raw.executed
    assert isinstance(select, SimpleStatement) and select.consistency_level == ConsistencyLevel.LOCAL_ONE
    assert insert == "INSERT INTO t (a) VALUES (1)"  # untouched: runs at the default (write) profile consistency


def test_prepared_selects_carry_the_read_consistency():
    s = ConsistencySession(FakeSession(), ConsistencyLevel.LOCAL_QUORUM)
    assert s.prepare("select a from t where id=?").consistency_level == ConsistencyLevel.LOCAL_QUORUM
    assert s.prepare("UPDATE t SET a=? WHERE id=?").consistency_level is None


def test_other_attributes_are_delegated_and_async_is_covered():
    raw = FakeSession()
    s = ConsistencySession(raw, ConsistencyLevel.LOCAL_QUORUM)
    assert s.keyspace == "ks" and s.execute_async("SELECT 1") == "fut"
    assert raw.executed[0].consistency_level == ConsistencyLevel.LOCAL_QUORUM
