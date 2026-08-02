"""
tests/unit/test_schemas.py

Unit tests for Phase 5: Domain Models & Pydantic Schemas.
Verifies validation logic on event envelopes, payloads, REST requests, and REST responses.
Asserts that API request contract hides embedding vectors.
"""

from datetime import datetime
from uuid import uuid4
import pytest
from pydantic import ValidationError

from app.models.memory import MemoryState
from app.models.snapshot import ConversationSnapshot
from app.models.summary import SummaryRecord
from app.schemas.events import (
    MessageCreatedPayload,
    ResponseCompletedPayload,
    MemoryEventEnvelope,
)
from app.schemas.requests import GetContextRequest
from app.schemas.responses import (
    RecentMessageResponse,
    FactResponse,
    MemoryContextResponse,
)


def test_domain_models_initialization():
    """
    Ensures domain models initialize cleanly with correct types.
    """
    now = datetime.utcnow()
    # 1. Snapshot Model
    snap = ConversationSnapshot(
        conversation_id="conv-123",
        user_id="user-456",
        message_count=10,
        state=MemoryState.ACTIVE,
        summary_version=1,
        fact_version=2,
        snapshot_version=3,
        last_summary_msg_id="msg-001",
        updated_at=now,
    )
    assert snap.conversation_id == "conv-123"
    assert snap.state == MemoryState.ACTIVE

    # 2. Summary Model
    record = SummaryRecord(
        conversation_id="conv-123",
        summary_text="Quick summary",
        summary_version=1,
        model_name="test-llm",
        model_version="v1",
        generated_at=now,
    )
    assert record.summary_text == "Quick summary"
    assert record.generated_at == now


def test_kafka_event_payloads_validation():
    """
    Verifies validation rules on Kafka event payloads.
    """
    # 1. MessageCreatedPayload - Success
    payload = MessageCreatedPayload(
        message_id="msg-1",
        role="user",
        content="Hello world",
    )
    assert payload.role == "user"
    assert payload.content == "Hello world"

    # MessageCreatedPayload - Invalid Role
    with pytest.raises(ValidationError):
        MessageCreatedPayload(
            message_id="msg-1",
            role="invalid_role",  # Not in Literal
            content="Hello",
        )

    # 2. ResponseCompletedPayload - Success
    resp = ResponseCompletedPayload(
        message_id="msg-2",
        content="Hi there",
    )
    assert resp.role == "assistant"  # Defaults to assistant

    # 3. Envelope - Success
    envelope = MemoryEventEnvelope(
        event_id="evt-100",
        event_type="chat.message.created",
        conversation_id="conv-123",
        user_id="user-456",
        payload=payload,
    )
    assert envelope.event_id == "evt-100"
    assert envelope.payload.content == "Hello world"


def test_api_requests_hides_vectors():
    """
    Validates REST request schemas and verifies that query is raw text
    and no vector embeddings are exposed to clients.
    """
    # Valid Request
    req = GetContextRequest(
        conversation_id="conv-123",
        user_id="user-456",
        query="python coding",
        top_k_facts=5,
    )
    assert req.query == "python coding"
    assert req.top_k_facts == 5

    # Check schema fields to make sure no embedding or vector fields exist
    schema_properties = GetContextRequest.model_fields
    assert "query" in schema_properties
    assert "query_embedding" not in schema_properties
    assert "embedding" not in schema_properties
    assert "vector" not in schema_properties

    # Invalid Request (missing query)
    with pytest.raises(ValidationError):
        GetContextRequest(
            conversation_id="conv-123",
            user_id="user-456",
        )


def test_api_responses_serialization():
    """
    Validates REST response serialization.
    """
    msg = RecentMessageResponse(
        message_id=str(uuid4()),
        role="user",
        content="Show me python",
        created_at=datetime.utcnow(),
    )
    fact = FactResponse(
        fact_id=str(uuid4()),
        category="preferences",
        statement="User prefers Python.",
        importance=0.9,
        updated_at=datetime.utcnow(),
    )
    resp = MemoryContextResponse(
        short_term=[msg],
        summary="A python convo",
        parent_summaries=["Past convo 1"],
        long_term_facts=[fact],
        semantic_results=[{"raw_milvus": 1}],
        context_version=42,
    )
    assert resp.context_version == 42
    assert len(resp.short_term) == 1
    assert resp.short_term[0].content == "Show me python"
    assert len(resp.long_term_facts) == 1
    assert resp.long_term_facts[0].statement == "User prefers Python."
