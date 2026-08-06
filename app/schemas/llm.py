"""
app/schemas/llm.py

Pydantic validation schemas for the internal LLM engine requests and responses.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """
    Standard chat completion message dictionary replica.
    """
    role: str = Field(
        ...,
        description="Role of the message author (e.g. 'user', 'assistant', 'system')"
    )
    content: str = Field(
        ...,
        description="Text content of the message"
    )


class SummarizeRequest(BaseModel):
    """
    Request schema for the conversation summary generation endpoint.
    """
    previous_summary: Optional[str] = Field(
        None,
        description="Previous summary text to build upon, if any exists."
    )
    new_messages: List[LLMMessage] = Field(
        ...,
        description="List of new chat messages since the last summary snapshot."
    )


class SummarizeResponse(BaseModel):
    """
    Response schema for the conversation summary generation endpoint.
    """
    summary: str = Field(
        ...,
        description="Generated/updated conversation summary text."
    )


class FactExtractRequest(BaseModel):
    """
    Request schema for the user fact extraction endpoint.
    """
    summary: str = Field(
        ...,
        description="Conversation summary text to extract facts from."
    )


class ExtractedFact(BaseModel):
    """
    Structured atomic representation of a single extracted fact.
    """
    category: str = Field(
        ...,
        description="Fact category (e.g. 'preferences', 'habits', 'plans', 'goals')."
    )
    importance: float = Field(
        ...,
        description="Fact importance/score (between 0.0 and 1.0)."
    )
    statement: str = Field(
        ...,
        description="Factual statement text."
    )


class FactExtractResponse(BaseModel):
    """
    Response schema for the user fact extraction endpoint.
    """
    facts: List[ExtractedFact] = Field(
        default_factory=list,
        description="List of extracted structured facts."
    )
