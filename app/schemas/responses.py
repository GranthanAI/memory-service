"""
app/schemas/responses.py

Pydantic schemas for validated HTTP API responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RecentMessageResponse(BaseModel):
    """
    Durable sliding message window item representation.
    """
    message_id: str = Field(..., description="Message UUID")
    role: str = Field(..., description="Role of author ('user', 'assistant', 'system')")
    content: str = Field(..., description="Message text content")
    created_at: datetime = Field(..., description="Message generation timestamp")


class FactResponse(BaseModel):
    """
    Durable user fact representation.
    """
    fact_id: str = Field(..., description="Unique fact UUID string")
    category: str = Field(..., description="Fact category (e.g. preferences, habits, etc.)")
    statement: str = Field(..., description="Deduplicated fact statement text")
    importance: float = Field(..., description="Fact importance score")
    updated_at: datetime = Field(..., description="Fact update timestamp")


class MemoryContextResponse(BaseModel):
    """
    Response schema for the compiled retrieval context payload.
    Exposes structured content from all three memory layers.
    """
    short_term: List[RecentMessageResponse] = Field(
        ..., description="Recent sliding message window"
    )
    summary: Optional[str] = Field(
        None, description="Incremental conversation summary text"
    )
    parent_summaries: List[str] = Field(
        default_factory=list, description="Summaries of parent/ancestor conversations"
    )
    long_term_facts: List[FactResponse] = Field(
        default_factory=list, description="Ranked and decay-scored user facts"
    )
    semantic_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Direct Milvus search outputs"
    )
    context_version: int = Field(
        ..., description="Monotonically increasing context generation version"
    )
    built_at: datetime = Field(
        default_factory=datetime.utcnow, description="Retrieval compilation timestamp"
    )
