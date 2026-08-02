"""
app/schemas/requests.py

Pydantic schemas for validated HTTP API requests.
"""

from pydantic import BaseModel, Field


class GetContextRequest(BaseModel):
    """
    Validation schema for retrieving memory context.
    Hides embedding generation from clients; callers pass raw query string.
    """
    conversation_id: str = Field(..., description="ID of the conversation to build context for")
    user_id: str = Field(..., description="ID of the user owning the conversation")
    query: str = Field(..., description="Raw text search query for semantic memory retrieval")
    top_k_facts: int = Field(10, ge=1, le=50, description="Number of ranked user facts to retrieve")
