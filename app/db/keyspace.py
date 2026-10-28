"""Cassandra keyspace / consistency helpers. Pure functions (no settings import) so Settings can validate with them.

The same logic lives in conversation-service (app/db/keyspace.py); keep the two in step.

CLI: prints the CREATE KEYSPACE statement for the current configuration.
"""
import re

PRODUCTION_ENVIRONMENTS = frozenset({"production", "prod"})
STRATEGIES = ("NetworkTopologyStrategy", "SimpleStrategy")
CONSISTENCY_LEVELS = (
    "ANY", "ONE", "TWO", "THREE", "QUORUM", "ALL", "LOCAL_QUORUM", "EACH_QUORUM", "LOCAL_ONE",
)
SERIAL_LEVELS = ("SERIAL", "LOCAL_SERIAL")
PRODUCTION_MIN_RF = 3
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def effective_replication_factor(rf: int | None, environment: str) -> int:
    """Explicit RF wins; otherwise 3 in production and 1 elsewhere (single-node development / staging)."""
    if rf is not None:
        return rf
    return PRODUCTION_MIN_RF if environment.strip().lower() in PRODUCTION_ENVIRONMENTS else 1


def validate_cassandra_settings(
    environment: str, strategy: str, rf: int | None, keyspace: str, local_dc: str,
    read_cl: str, write_cl: str, serial_cl: str,
) -> list[str]:
    """Return a list of problems (empty = fine)."""
    problems: list[str] = []
    if strategy not in STRATEGIES:
        problems.append(f"CASSANDRA_REPLICATION_STRATEGY must be one of {', '.join(STRATEGIES)}")
    if not _IDENTIFIER.match(keyspace):
        problems.append("CASSANDRA_KEYSPACE must be a plain identifier (letters, digits, underscore)")
    if strategy == "NetworkTopologyStrategy" and not local_dc.strip():
        problems.append("CASSANDRA_LOCAL_DC must name the datacenter for NetworkTopologyStrategy")
    if rf is not None and rf < 1:
        problems.append("CASSANDRA_REPLICATION_FACTOR must be >= 1")
    for name, value, allowed in (
        ("CASSANDRA_CONSISTENCY_READ", read_cl, CONSISTENCY_LEVELS),
        ("CASSANDRA_CONSISTENCY_WRITE", write_cl, CONSISTENCY_LEVELS),
        ("CASSANDRA_SERIAL_CONSISTENCY", serial_cl, SERIAL_LEVELS),
    ):
        if value not in allowed:
            problems.append(f"{name} must be one of {', '.join(allowed)}")
    if environment.strip().lower() in PRODUCTION_ENVIRONMENTS:
        eff = effective_replication_factor(rf, environment)
        if strategy != "NetworkTopologyStrategy" or eff < PRODUCTION_MIN_RF:
            problems.append(
                f"production requires CASSANDRA_REPLICATION_STRATEGY=NetworkTopologyStrategy and "
                f"CASSANDRA_REPLICATION_FACTOR>={PRODUCTION_MIN_RF} (got {strategy}, RF={eff}). "
                "Single-node environments must set ENVIRONMENT=staging or development instead"
            )
    return problems


def keyspace_cql(keyspace: str, strategy: str, local_dc: str, rf: int, durable_writes: bool = True) -> str:
    """CREATE KEYSPACE statement for the configured replication."""
    if not _IDENTIFIER.match(keyspace):
        raise ValueError("invalid keyspace name")
    if strategy == "NetworkTopologyStrategy":
        dc = local_dc.replace("'", "''")
        replication = f"{{'class': 'NetworkTopologyStrategy', '{dc}': {int(rf)}}}"
    elif strategy == "SimpleStrategy":
        replication = f"{{'class': 'SimpleStrategy', 'replication_factor': {int(rf)}}}"
    else:
        raise ValueError(f"unknown replication strategy: {strategy}")
    return (
        f"CREATE KEYSPACE IF NOT EXISTS {keyspace}\n"
        f"WITH replication = {replication}\n"
        f"AND durable_writes = {'true' if durable_writes else 'false'};"
    )


def is_read_query(query_text: str) -> bool:
    return query_text.lstrip().upper().startswith("SELECT")


if __name__ == "__main__":  # pragma: no cover - exercised via `make schema`
    from app.core.config import settings

    print(keyspace_cql(
        settings.CASSANDRA_KEYSPACE,
        settings.CASSANDRA_REPLICATION_STRATEGY,
        settings.CASSANDRA_LOCAL_DC,
        effective_replication_factor(settings.CASSANDRA_REPLICATION_FACTOR, settings.ENVIRONMENT),
    ))
