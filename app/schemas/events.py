"""
app/schemas/events.py

Pydantic schemas representing Kafka events consumed or produced by the service.
"""

from datetime import datetime
from typing import Literal, Union
from pydantic import BaseModel, Field


class MessageCreatedPayload(BaseModel):
    """
    Payload for chat.message.created Kafka events.
    """
    message_id: str = Field(..., description="Unique message UUID string")
    role: Literal["user", "assistant", "system"] = Field(..., description="Role of the message author")
    content: str = Field(..., description="Text content of the message")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when message was created")


class ResponseCompletedPayload(BaseModel):
    """
    Payload for chat.response.completed Kafka events.
    """
    message_id: str = Field(..., description="Unique message UUID string")
    role: Literal["assistant"] = Field("assistant", description="Assistant role")
    content: str = Field(..., description="Text content of the generated response")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when response was completed")


class MemoryEventEnvelope(BaseModel):
    """
    Envelope wrapping all incoming Kafka events for deduplication and routing.
    """
    event_id: str = Field(..., description="Unique event transaction ID for idempotency checks")
    event_type: str = Field(..., description="Type of the event, e.g. chat.message.created")
    conversation_id: str = Field(..., description="Conversation ID reference")
    user_id: str = Field(..., description="User ID reference")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event generation timestamp")
    payload: Union[MessageCreatedPayload, ResponseCompletedPayload] = Field(..., description="Inner typed payload")
