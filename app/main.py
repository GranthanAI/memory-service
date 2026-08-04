"""
app/main.py

FastAPI Application Entry Point.

Route map:
  GET  /health                    — Liveness probe (shallow — no infra checks)
  GET  /internal/health/ready     — Readiness probe (deep — all infra checks)
  POST /internal/memory/context   — Memory context retrieval (LLM Service)
  GET  /metrics                   — Prometheus metrics endpoint
"""

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.api.routers import api_router
from app.lifespan import lifespan

app = FastAPI(
    title="GraphGPT Memory Service",
    description=(
        "Derived Cognitive AI Memory Engine. "
        "Converts raw conversational data into structured, queryable memory "
        "representations that power personalized AI responses."
    ),
    version="4.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Internal API routes ───────────────────────────────────────────────────────
app.include_router(api_router)

# ── Prometheus metrics sub-app ────────────────────────────────────────────────
# Mounted as a separate ASGI sub-app at /metrics.
# Compatible with standard Prometheus scrape configs.
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ── Liveness probe ────────────────────────────────────────────────────────────
@app.get(
    "/health",
    summary="Liveness Probe",
    description=(
        "Shallow liveness probe — confirms the process is alive and the event loop is "
        "responding. Does NOT check infrastructure health. Use GET /internal/health/ready "
        "for full deep readiness verification."
    ),
    tags=["Observability"],
)
def health_check():
    """Process-level liveness check. Always returns 200 if the server is running."""
    return {"status": "healthy"}
