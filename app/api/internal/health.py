"""
app/api/internal/health.py

Deep Readiness Probe endpoint.

GET /ready — verified connection health across all four infrastructure
components required for full service operation:
  1. Cassandra  — runs `SELECT release_version FROM system.local`
  2. Redis      — runs PING
  3. Milvus     — calls utility.list_collections()
  4. LLM gRPC   — checks pool has at least one live channel

Returns HTTP 200 + {"status": "ready", "checks": {...}} if all pass.
Returns HTTP 503 + {"status": "not_ready", "checks": {...}} if any fail.

The /health endpoint (shallow liveness probe) is registered directly in
main.py and does not depend on infrastructure health.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, Response

from app.api.dependencies import (
    get_cassandra_health_session,
    get_llm_service,
    get_redis_health_client,
)
from app.core.config import settings
from app.db.milvus import check_milvus_ready

logger = logging.getLogger("memory_service.api.health")

router = APIRouter()


@router.get(
    "/ready",
    summary="Deep Readiness Probe",
    description=(
        "Verifies live connectivity to Cassandra, Redis, Milvus, and the LLM gRPC pool. "
        "Returns 200 if all pass, 503 if any fail."
    ),
    tags=["Observability"],
)
async def readiness_probe(
    response: Response,
    cassandra_session=Depends(get_cassandra_health_session),
    redis_client=Depends(get_redis_health_client),
    llm_service=Depends(get_llm_service),
) -> Dict[str, Any]:
    """
    Performs deep health checks on all four infrastructure components.
    Intended for Kubernetes readinessProbe — gates traffic until all checks pass.
    """
    checks: Dict[str, Any] = {}
    all_healthy = True

    # ── 1. Cassandra ──────────────────────────────────────────────────────────
    try:
        row = cassandra_session.execute(
            "SELECT release_version FROM system.local"
        ).one()
        checks["cassandra"] = {
            "status": "ok",
            "release_version": row.release_version if row else "unknown",
        }
    except Exception as e:
        logger.error(f"Cassandra readiness check failed: {e}")
        checks["cassandra"] = {"status": "error", "detail": str(e)}
        all_healthy = False

    # ── 2. Redis ──────────────────────────────────────────────────────────────
    try:
        pong = await redis_client.ping()
        checks["redis"] = {"status": "ok" if pong else "error"}
        if not pong:
            all_healthy = False
    except Exception as e:
        logger.error(f"Redis readiness check failed: {e}")
        checks["redis"] = {"status": "error", "detail": str(e)}
        all_healthy = False

    # ── 3. Milvus ─────────────────────────────────────────────────────────────
    try:
        milvus_ok = check_milvus_ready()
        checks["milvus"] = {"status": "ok" if milvus_ok else "error"}
        if not milvus_ok:
            all_healthy = False
    except Exception as e:
        logger.error(f"Milvus readiness check failed: {e}")
        checks["milvus"] = {"status": "error", "detail": str(e)}
        all_healthy = False

    # ── 4. Internal LLM Engine ────────────────────────────────────────────────
    try:
        llm_ok = await llm_service.manager.check_health()
        checks["llm_engine"] = {
            "status": "ok" if llm_ok else "error",
            "provider": settings.LLM_PROVIDER,
        }
        if not llm_ok:
            all_healthy = False
    except Exception as e:
        logger.warning(f"Internal LLM engine readiness check failed: {e}")
        checks["llm_engine"] = {"status": "error", "detail": str(e)}
        all_healthy = False

    # ── Response ──────────────────────────────────────────────────────────────
    if all_healthy:
        return {"status": "ready", "checks": checks}

    response.status_code = 503
    return {"status": "not_ready", "checks": checks}
