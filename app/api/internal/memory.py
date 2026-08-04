"""
app/api/internal/memory.py

Internal Memory Context REST endpoint.

POST /internal/memory/context

Called by the LLM Service to retrieve structured memory context for
enriching AI prompt assembly. Callers pass raw query text — embedding
generation is handled internally, decoupling callers from the model version.

Request: GetContextRequest  (conversation_id, user_id, query, top_k_facts)
Response: MemoryContextResponse (short_term, summary, parent_summaries,
                                  long_term_facts, semantic_results,
                                  context_version, built_at)

The endpoint:
  1. Validates the request body with Pydantic.
  2. Delegates to ContextBuilder.build_context() which concurrently fetches
     short-term messages, summary, parent summaries, and semantic facts.
  3. Records end-to-end latency in the CTX_BUILD histogram.
  4. Returns a structured MemoryContextResponse.

Note: query_vector is intentionally None here — the ContextBuilder currently
returns facts without semantic scoring when no vector is supplied. Future
versions will call the LLM gRPC pool internally to embed the query.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_context_builder
from app.core.metrics import CTX_BUILD
from app.schemas.requests import GetContextRequest
from app.schemas.responses import (
    FactResponse,
    MemoryContextResponse,
    RecentMessageResponse,
)
from app.services.context_builder import ContextBuilder

logger = logging.getLogger("memory_service.api.memory")

router = APIRouter()


def _to_recent_message(msg: Dict[str, Any]) -> RecentMessageResponse:
    """Convert a raw message dict from the retrieval layer to a response model."""
    created_at = msg.get("created_at")
    if not isinstance(created_at, datetime):
        # Cassandra driver returns datetime objects; guard against unexpected types
        try:
            created_at = datetime.fromisoformat(str(created_at))
        except Exception:
            created_at = datetime.now(timezone.utc)

    return RecentMessageResponse(
        message_id=str(msg.get("message_id", "")),
        role=str(msg.get("role", "user")),
        content=str(msg.get("content", "")),
        created_at=created_at,
    )


def _to_fact_response(fact: Dict[str, Any]) -> FactResponse:
    """Convert a raw fact dict from the ranking layer to a response model."""
    updated_at = fact.get("updated_at") or fact.get("created_at")
    if not isinstance(updated_at, datetime):
        try:
            updated_at = datetime.fromisoformat(str(updated_at))
        except Exception:
            updated_at = datetime.now(timezone.utc)

    return FactResponse(
        fact_id=str(fact.get("fact_id", "")),
        category=str(fact.get("category", "general")),
        statement=str(fact.get("statement", "")),
        importance=float(fact.get("importance", 0.5)),
        updated_at=updated_at,
    )


@router.post(
    "/context",
    response_model=MemoryContextResponse,
    summary="Retrieve Memory Context",
    description=(
        "Builds a structured memory context payload for the calling LLM Service. "
        "Concurrently fetches short-term messages, the active conversation summary, "
        "parent/ancestor summaries via Graph Service, and ranked long-term user facts "
        "from Milvus. Raw query text is passed by the caller — embedding is handled "
        "internally, decoupling callers from the embedding model version."
    ),
    tags=["Memory"],
    status_code=status.HTTP_200_OK,
)
async def get_memory_context(
    request: GetContextRequest,
    context_builder: ContextBuilder = Depends(get_context_builder),
) -> MemoryContextResponse:
    """
    Primary context retrieval endpoint.

    Delegates to ContextBuilder which concurrently gathers:
    - Short-term message sliding window (Redis hot cache → Cassandra fallback)
    - Active conversation summary (Redis → Cassandra fallback)
    - Parent/ancestor summaries via Graph Service (with timeout + graceful fallback)
    - Relevant long-term facts (Milvus HNSW search + Cassandra metadata)
    """
    start_ts = time.perf_counter()

    try:
        context_payload = await context_builder.build_context(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            query_vector=None,  # Future: embed request.query via LLM gRPC pool
        )
    except Exception as e:
        logger.error(
            f"Context assembly failed for conversation {request.conversation_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assemble memory context: {str(e)}",
        )
    finally:
        elapsed = time.perf_counter() - start_ts
        CTX_BUILD.observe(elapsed)
        logger.info(
            f"Context built for conversation={request.conversation_id} "
            f"user={request.user_id} latency={elapsed:.3f}s"
        )

    # ── Map raw context payload to response schema ─────────────────────────
    raw_messages: List[Dict[str, Any]] = context_payload.get("short_term_messages", [])
    raw_facts: List[Dict[str, Any]] = context_payload.get("relevant_facts", [])
    parent_summaries: List[str] = []

    for ps in context_payload.get("parent_summaries", []):
        if isinstance(ps, dict):
            parent_summaries.append(ps.get("summary_text", ""))
        else:
            parent_summaries.append(str(ps))

    # Determine context version from snapshot if available
    snapshot = context_payload.get("snapshot")
    context_version = (
        snapshot.get("snapshot_version", 0) if isinstance(snapshot, dict) else 0
    )

    return MemoryContextResponse(
        short_term=[_to_recent_message(m) for m in raw_messages],
        summary=context_payload.get("current_summary") or None,
        parent_summaries=parent_summaries,
        long_term_facts=[_to_fact_response(f) for f in raw_facts],
        semantic_results=raw_facts,  # Raw Milvus hit objects for downstream consumers
        context_version=context_version,
        built_at=datetime.now(timezone.utc),
    )
