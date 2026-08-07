"""
app/api/routers.py

Central router registry for the Memory Service API.

All sub-routers are imported here and mounted onto a single top-level
api_router, which is then included in the FastAPI app in main.py.

Route map:
  POST /internal/memory/context   — context retrieval (LLM Service caller)
  GET  /internal/health/ready     — deep readiness probe (Kubernetes)
"""

from fastapi import APIRouter

from app.api.internal import health, memory, llm

# Top-level internal router — all routes are service-internal (not public)
api_router = APIRouter(prefix="/internal")

# Memory context retrieval
api_router.include_router(
    memory.router,
    prefix="/memory",
    tags=["Memory"],
)

# Health and readiness probes
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Observability"],
)

# LLM Engine summarization and fact extraction
api_router.include_router(
    llm.router,
    prefix="/llm",
    tags=["LLM Engine"],
)
