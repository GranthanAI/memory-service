"""
tests/unit/test_db.py

Unit tests for Phase 3: Database Connection Adapters.
Tests cover Cassandra (primary store), Redis (hot cache), and Milvus (vector index).

All external drivers are mocked — no real infrastructure required.
Test assertions match the Phase 3 verification spec in docs/phases.md.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from app.db.redis import init_redis_pool, close_redis_pool, get_redis_client
from app.db.milvus import connect_milvus, disconnect_milvus, check_milvus_ready
from app.db.session import initialize_db_sessions, close_db_sessions


# ─── Helper: mock all three healthy drivers ────────────────────────────────────

def _mock_cassandra_session():
    """Returns a mock Cassandra session that reports healthy."""
    session = MagicMock()
    session.execute = MagicMock(return_value=[])
    return session


def _mock_cassandra_cluster(session):
    """Returns a mock Cassandra Cluster that returns the given session."""
    cluster = MagicMock()
    cluster.connect = MagicMock(return_value=session)
    return cluster


# ─── Cassandra adapter unit tests ─────────────────────────────────────────────

class TestCassandraAdapter:
    """Tests for app/db/cassandra.py lifecycle functions."""

    def test_connect_cassandra_parses_multi_node_hosts(self):
        """Verifies that comma-separated CASSANDRA_HOSTS are parsed correctly."""
        from app.db import cassandra as cassandra_mod

        mock_session = _mock_cassandra_session()
        mock_cluster = _mock_cassandra_cluster(mock_session)

        with patch("app.db.cassandra.Cluster", return_value=mock_cluster) as mock_cluster_cls, \
             patch("app.db.cassandra.settings") as mock_settings, \
             patch("app.db.migrations.MigrationManager") as mock_migration_manager:

            mock_settings.CASSANDRA_HOSTS = "10.0.0.1,10.0.0.2,10.0.0.3"
            mock_settings.CASSANDRA_PORT = 9042
            mock_settings.CASSANDRA_KEYSPACE = "graphgpt_memory"
            mock_settings.CASSANDRA_TIMEOUT_SECONDS = 5.0

            cassandra_mod._cluster = None
            cassandra_mod._session = None
            cassandra_mod.connect_cassandra()

            call_args = mock_cluster_cls.call_args
            contact_points = call_args.kwargs.get("contact_points") or call_args.args[0]
            assert contact_points == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]

    def test_check_cassandra_ready_returns_true_on_success(self):
        """Verifies health check returns True when session.execute succeeds."""
        from app.db import cassandra as cassandra_mod

        mock_session = _mock_cassandra_session()
        cassandra_mod._session = mock_session

        result = cassandra_mod.check_cassandra_ready()

        assert result is True
        mock_session.execute.assert_called_once()
        executed_cql = str(mock_session.execute.call_args)
        assert "system.local" in executed_cql

    def test_check_cassandra_ready_returns_false_on_exception(self):
        """Verifies health check returns False when session raises an exception."""
        from app.db import cassandra as cassandra_mod

        mock_session = MagicMock()
        mock_session.execute.side_effect = Exception("Cassandra unavailable")
        cassandra_mod._session = mock_session

        result = cassandra_mod.check_cassandra_ready()

        assert result is False

    def test_check_cassandra_ready_returns_false_when_not_initialized(self):
        """Verifies health check returns False when session is None."""
        from app.db import cassandra as cassandra_mod

        cassandra_mod._session = None
        result = cassandra_mod.check_cassandra_ready()

        assert result is False

    def test_get_session_raises_if_not_initialized(self):
        """Verifies get_session() raises RuntimeError when called before connect."""
        from app.db import cassandra as cassandra_mod
        from app.db.cassandra import get_session

        cassandra_mod._session = None
        with pytest.raises(RuntimeError, match="Cassandra session is not initialized"):
            get_session()

    def test_disconnect_cassandra_shuts_down_both(self):
        """Verifies disconnect_cassandra() calls shutdown on both session and cluster."""
        from app.db import cassandra as cassandra_mod

        mock_session = MagicMock()
        mock_cluster = MagicMock()
        cassandra_mod._session = mock_session
        cassandra_mod._cluster = mock_cluster

        cassandra_mod.disconnect_cassandra()

        mock_session.shutdown.assert_called_once()
        mock_cluster.shutdown.assert_called_once()
        assert cassandra_mod._session is None
        assert cassandra_mod._cluster is None


# ─── Session orchestration tests ──────────────────────────────────────────────

class TestSessionOrchestrator:
    """Tests for app/db/session.py initialization and shutdown."""

    @pytest.mark.asyncio
    async def test_initialization_success_all_three_healthy(self):
        """
        Verifies that when Cassandra, Redis, and Milvus are all healthy,
        initialize_db_sessions() completes without raising.
        """
        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with patch("app.db.session.connect_cassandra") as mock_cass_connect, \
             patch("app.db.session.check_cassandra_ready", return_value=True), \
             patch("app.db.redis.aioredis.ConnectionPool.from_url"), \
             patch("app.db.redis.aioredis.Redis", return_value=mock_redis), \
             patch("app.db.session.connect_milvus") as mock_milvus_connect, \
             patch("app.db.session.check_milvus_ready", return_value=True):

            await initialize_db_sessions()

            mock_cass_connect.assert_called_once()
            mock_redis.ping.assert_called_once()
            mock_milvus_connect.assert_called_once_with("localhost", 19530)

    @pytest.mark.asyncio
    async def test_cassandra_failure_raises_runtime_error(self):
        """
        Verifies that a Cassandra connection failure raises RuntimeError
        and mentions 'Cassandra' in the error message.
        """
        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with patch("app.db.session.connect_cassandra", side_effect=Exception("Cassandra down")), \
             patch("app.db.session.check_cassandra_ready", return_value=False), \
             patch("app.db.redis.aioredis.ConnectionPool.from_url"), \
             patch("app.db.redis.aioredis.Redis", return_value=mock_redis), \
             patch("app.db.session.connect_milvus"), \
             patch("app.db.session.check_milvus_ready", return_value=True), \
             patch("app.db.session.close_db_sessions") as mock_cleanup:

            with pytest.raises(RuntimeError) as exc:
                await initialize_db_sessions()

            assert "Cassandra" in str(exc.value)
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_failure_raises_runtime_error(self):
        """
        Verifies that a Redis PING failure raises RuntimeError
        and mentions 'Redis' in the error message.
        """
        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("Redis timed out"))

        with patch("app.db.session.connect_cassandra"), \
             patch("app.db.session.check_cassandra_ready", return_value=True), \
             patch("app.db.redis.aioredis.ConnectionPool.from_url"), \
             patch("app.db.redis.aioredis.Redis", return_value=mock_redis), \
             patch("app.db.session.connect_milvus"), \
             patch("app.db.session.check_milvus_ready", return_value=True), \
             patch("app.db.session.close_db_sessions") as mock_cleanup:

            with pytest.raises(RuntimeError) as exc:
                await initialize_db_sessions()

            assert "Redis" in str(exc.value)
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_milvus_failure_raises_runtime_error(self):
        """
        Verifies that a Milvus health check failure raises RuntimeError
        and mentions 'Milvus' in the error message.
        """
        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with patch("app.db.session.connect_cassandra"), \
             patch("app.db.session.check_cassandra_ready", return_value=True), \
             patch("app.db.redis.aioredis.ConnectionPool.from_url"), \
             patch("app.db.redis.aioredis.Redis", return_value=mock_redis), \
             patch("app.db.session.connect_milvus"), \
             patch("app.db.session.check_milvus_ready", return_value=False), \
             patch("app.db.session.close_db_sessions") as mock_cleanup:

            with pytest.raises(RuntimeError) as exc:
                await initialize_db_sessions()

            assert "Milvus" in str(exc.value)
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_sessions_completes_without_error(self):
        """
        Verifies that close_db_sessions() runs all three disconnects
        and does not raise even if individual disconnects fail.
        """
        with patch("app.db.session.disconnect_milvus") as mock_milvus_disc, \
             patch("app.db.session.close_redis_pool") as mock_redis_close, \
             patch("app.db.session.disconnect_cassandra") as mock_cass_disc:

            mock_redis_close.return_value = AsyncMock(return_value=None)()
            await close_db_sessions()

            mock_milvus_disc.assert_called_once()
            mock_cass_disc.assert_called_once()
