"""
tests/unit/test_api_endpoints.py

Unit tests for REST API endpoints using FastAPI TestClient with dependency overrides.

Tests:
  - GET /health              — liveness probe
  - GET /internal/health/ready — readiness probe (all pass / partial fail / all fail)
  - POST /internal/memory/context — context retrieval (success, builder error)
  - GET /metrics             — Prometheus endpoint
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies import (
    get_cassandra_health_session,
    get_redis_health_client,
    get_llm_pool,
    get_context_builder,
)


# ─── Mock Factories ───────────────────────────────────────────────────────────

def make_mock_cassandra_session(release_version: str = "4.1.0", fail: bool = False):
    session = MagicMock()
    if fail:
        session.execute.side_effect = Exception("Cassandra connection refused")
    else:
        row = MagicMock()
        row.release_version = release_version
        result = MagicMock()
        result.one.return_value = row
        session.execute.return_value = result
    return session


def make_mock_redis_client(ping_ok: bool = True, fail: bool = False):
    client = AsyncMock()
    if fail:
        client.ping.side_effect = Exception("Redis NOAUTH")
    else:
        client.ping.return_value = ping_ok
    return client


def make_mock_llm_pool(healthy: bool = True):
    pool = AsyncMock()
    pool.pool_size = 3
    if healthy:
        pool.get_channel.return_value = MagicMock()
    else:
        pool.get_channel.side_effect = RuntimeError("No healthy gRPC channels available")
    return pool


def make_mock_context_builder(payload: dict = None, fail: bool = False):
    builder = AsyncMock()
    if fail:
        builder.build_context.side_effect = RuntimeError("Cassandra timeout")
    else:
        builder.build_context.return_value = payload or {
            "conversation_id": "conv-test-001",
            "user_id": "user-test-001",
            "current_summary": "Test summary",
            "short_term_messages": [
                {
                    "message_id": "msg-1",
                    "role": "user",
                    "content": "Hello",
                    "created_at": datetime.now(timezone.utc),
                }
            ],
            "parent_summaries": [],
            "relevant_facts": [],
            "metadata": {"parent_summaries_available": True, "facts_retrieved_count": 0},
        }
    return builder


# ─── Helpers ──────────────────────────────────────────────────────────────────

def override_all_healthy():
    """Apply dependency overrides for all-healthy infrastructure."""
    app.dependency_overrides[get_cassandra_health_session] = lambda: make_mock_cassandra_session()
    app.dependency_overrides[get_redis_health_client] = lambda: make_mock_redis_client()
    app.dependency_overrides[get_llm_pool] = lambda: make_mock_llm_pool()


def clear_overrides():
    app.dependency_overrides.clear()


# ─── Liveness probe ───────────────────────────────────────────────────────────

class TestLivenessProbe:
    def test_health_returns_200(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


# ─── Readiness probe ──────────────────────────────────────────────────────────

class TestReadinessProbe:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        clear_overrides()

    def test_ready_all_healthy(self):
        override_all_healthy()
        with patch("app.api.internal.health.check_milvus_ready", return_value=True):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/internal/health/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["checks"]["cassandra"]["status"] == "ok"
        assert body["checks"]["redis"]["status"] == "ok"
        assert body["checks"]["milvus"]["status"] == "ok"
        assert body["checks"]["llm_grpc"]["status"] == "ok"
        assert body["checks"]["llm_grpc"]["pool_size"] == 3

    def test_ready_cassandra_fails(self):
        app.dependency_overrides[get_cassandra_health_session] = lambda: make_mock_cassandra_session(fail=True)
        app.dependency_overrides[get_redis_health_client] = lambda: make_mock_redis_client()
        app.dependency_overrides[get_llm_pool] = lambda: make_mock_llm_pool()
        with patch("app.api.internal.health.check_milvus_ready", return_value=True):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/internal/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["cassandra"]["status"] == "error"
        assert body["checks"]["redis"]["status"] == "ok"

    def test_ready_redis_fails(self):
        app.dependency_overrides[get_cassandra_health_session] = lambda: make_mock_cassandra_session()
        app.dependency_overrides[get_redis_health_client] = lambda: make_mock_redis_client(fail=True)
        app.dependency_overrides[get_llm_pool] = lambda: make_mock_llm_pool()
        with patch("app.api.internal.health.check_milvus_ready", return_value=True):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/internal/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["redis"]["status"] == "error"

    def test_ready_milvus_fails(self):
        override_all_healthy()
        with patch("app.api.internal.health.check_milvus_ready", return_value=False):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/internal/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["milvus"]["status"] == "error"

    def test_ready_grpc_fails(self):
        app.dependency_overrides[get_cassandra_health_session] = lambda: make_mock_cassandra_session()
        app.dependency_overrides[get_redis_health_client] = lambda: make_mock_redis_client()
        app.dependency_overrides[get_llm_pool] = lambda: make_mock_llm_pool(healthy=False)
        with patch("app.api.internal.health.check_milvus_ready", return_value=True):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/internal/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["checks"]["llm_grpc"]["status"] == "error"

    def test_ready_all_fail_returns_503(self):
        app.dependency_overrides[get_cassandra_health_session] = lambda: make_mock_cassandra_session(fail=True)
        app.dependency_overrides[get_redis_health_client] = lambda: make_mock_redis_client(fail=True)
        app.dependency_overrides[get_llm_pool] = lambda: make_mock_llm_pool(healthy=False)
        with patch("app.api.internal.health.check_milvus_ready", return_value=False):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.get("/internal/health/ready")
        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"


# ─── Memory context endpoint ──────────────────────────────────────────────────

class TestMemoryContextEndpoint:
    @pytest.fixture(autouse=True)
    def setup_security_override(self):
        from app.core.security import verify_service_auth
        app.dependency_overrides[verify_service_auth] = lambda: {"identity": "test"}
        yield
        clear_overrides()

    def test_context_success(self):
        mock_builder = make_mock_context_builder()
        app.dependency_overrides[get_context_builder] = lambda: mock_builder
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/internal/memory/context",
                json={
                    "conversation_id": "conv-test-001",
                    "user_id": "user-test-001",
                    "query": "What are my food preferences?",
                    "top_k_facts": 5,
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["summary"] == "Test summary"
        assert len(body["short_term"]) == 1
        assert body["short_term"][0]["role"] == "user"
        assert body["short_term"][0]["content"] == "Hello"
        assert isinstance(body["context_version"], int)
        assert "built_at" in body

    def test_context_unauthorized_missing_credentials(self):
        from app.core.security import verify_service_auth
        if verify_service_auth in app.dependency_overrides:
            del app.dependency_overrides[verify_service_auth]
        
        mock_builder = make_mock_context_builder()
        app.dependency_overrides[get_context_builder] = lambda: mock_builder
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/internal/memory/context",
                json={
                    "conversation_id": "conv-test-001",
                    "user_id": "user-test-001",
                    "query": "What are my food preferences?",
                    "top_k_facts": 5,
                },
            )
        assert response.status_code == 401

    def test_context_authorized_with_api_key(self):
        from app.core.security import verify_service_auth
        if verify_service_auth in app.dependency_overrides:
            del app.dependency_overrides[verify_service_auth]
        
        mock_builder = make_mock_context_builder()
        app.dependency_overrides[get_context_builder] = lambda: mock_builder
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/internal/memory/context",
                headers={"X-API-Key": "graphgpt-memory-secret"},
                json={
                    "conversation_id": "conv-test-001",
                    "user_id": "user-test-001",
                    "query": "What are my food preferences?",
                    "top_k_facts": 5,
                },
            )
        assert response.status_code == 200

    def test_context_authorized_with_jwt(self):
        from app.core.security import verify_service_auth, generate_jwt
        if verify_service_auth in app.dependency_overrides:
            del app.dependency_overrides[verify_service_auth]
        
        token = generate_jwt({"sub": "test-service"}, secret_key="graphgpt-jwt-secret")
        mock_builder = make_mock_context_builder()
        app.dependency_overrides[get_context_builder] = lambda: mock_builder
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/internal/memory/context",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "conversation_id": "conv-test-001",
                    "user_id": "user-test-001",
                    "query": "What are my food preferences?",
                    "top_k_facts": 5,
                },
            )
        assert response.status_code == 200

    def test_context_missing_required_fields(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/internal/memory/context",
                json={"conversation_id": "conv-test-001"},  # missing user_id and query
            )
        assert response.status_code == 422

    def test_context_builder_error_returns_500(self):
        mock_builder = make_mock_context_builder(fail=True)
        app.dependency_overrides[get_context_builder] = lambda: mock_builder
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/internal/memory/context",
                json={
                    "conversation_id": "conv-fail-001",
                    "user_id": "user-fail-001",
                    "query": "query text",
                    "top_k_facts": 5,
                },
            )
        assert response.status_code == 500
        assert "Failed to assemble memory context" in response.json()["detail"]

    def test_context_top_k_facts_bounds(self):
        """top_k_facts must be between 1 and 50."""
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/internal/memory/context",
                json={
                    "conversation_id": "conv-test",
                    "user_id": "user-test",
                    "query": "test",
                    "top_k_facts": 0,  # Below minimum
                },
            )
        assert response.status_code == 422

    def test_context_empty_short_term(self):
        """Verify response still valid when no recent messages exist."""
        payload = {
            "conversation_id": "conv-empty",
            "user_id": "user-empty",
            "current_summary": None,
            "short_term_messages": [],
            "parent_summaries": [],
            "relevant_facts": [],
            "metadata": {"parent_summaries_available": False, "facts_retrieved_count": 0},
        }
        mock_builder = make_mock_context_builder(payload=payload)
        app.dependency_overrides[get_context_builder] = lambda: mock_builder
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/internal/memory/context",
                json={
                    "conversation_id": "conv-empty",
                    "user_id": "user-empty",
                    "query": "anything",
                    "top_k_facts": 5,
                },
            )
        assert response.status_code == 200
        body = response.json()
        assert body["short_term"] == []
        assert body["summary"] is None
        assert body["long_term_facts"] == []


# ─── Prometheus metrics endpoint ──────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self):
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus text format starts with # HELP or a metric name
        content = response.text
        assert "memory_redis_hit_total" in content or "# HELP" in content or "# TYPE" in content
