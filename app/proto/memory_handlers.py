"""
app/proto/memory_handlers.py

gRPC handler serving the Memory Service context API consumed by the LLM
Service's ContextCollector (llm-service/proto/memory.proto). Delegates to the
existing ContextBuilder / RetrievalService — no new business logic, just the
missing transport wired to already-built short/medium/long-term memory tiers.
"""

import logging
import uuid
from datetime import datetime, timezone

import grpc

from app.clients.embedding_client import EmbeddingClient
from app.core.logging import clear_log_context, set_log_context
from app.proto import memory_pb2, memory_pb2_grpc
from app.services.context_builder import ContextBuilder

logger = logging.getLogger("memory_service.proto.memory_handlers")


def _to_epoch_millis(value) -> int:
    """
    Converts a datetime, ISO string, or numeric epoch timestamp to epoch
    milliseconds; 0 if unavailable. Milvus-sourced facts store created_at as
    a raw DOUBLE (epoch seconds) — see MilvusRepository's schema — while
    Cassandra-sourced messages store it as a real datetime, so both shapes
    must be handled here.
    """
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value * 1000)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return 0
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


def _extract_trace_id(context) -> str:
    for key, val in (context.invocation_metadata() or []):
        if key.lower() in ("x-trace-id", "x-request-id"):
            return val
    return str(uuid.uuid4())


def _messages_to_proto(messages: list[dict]) -> list["memory_pb2.Message"]:
    return [
        memory_pb2.Message(
            role=m.get("role", "user"),
            content=m.get("content", ""),
            timestamp=_to_epoch_millis(m.get("created_at")),
            message_id=str(m.get("message_id", "")),
        )
        for m in messages
    ]


def _facts_to_proto(facts: list[dict]) -> list["memory_pb2.Fact"]:
    return [
        memory_pb2.Fact(
            content=f.get("statement", ""),
            confidence=float(f.get("importance") or 0.0),
            category=f.get("category", ""),
            last_updated=_to_epoch_millis(f.get("created_at")),
        )
        for f in facts
    ]


class MemoryContextHandler(memory_pb2_grpc.MemoryServiceServicer):
    """
    gRPC handler for MemoryService.GetMemoryContext / GetShortTermMemory.
    Delegates to ContextBuilder (short/medium/long-term aggregation) and
    RetrievalService (recent-messages-only path), converting a text query
    into a vector via EmbeddingClient when semantic fact retrieval is needed.
    """

    def __init__(self, context_builder: ContextBuilder, embedding_client: EmbeddingClient):
        self.context_builder = context_builder
        self.embedding_client = embedding_client

    async def GetMemoryContext(self, request, context):
        trace_id = _extract_trace_id(context)
        set_log_context(trace_id=trace_id, conversation_id=request.conversation_id)
        try:
            logger.info(
                f"gRPC GetMemoryContext request received for conversation={request.conversation_id} user={request.user_id}"
            )

            query_vector = None
            if request.query:
                try:
                    query_vector = await self.embedding_client.generate_embedding(request.query)
                except Exception as e:
                    logger.warning(f"Embedding generation failed, proceeding without semantic facts: {e}")

            result = await self.context_builder.build_context(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                query_vector=query_vector,
            )

            return memory_pb2.GetMemoryContextResponse(
                short_term_messages=_messages_to_proto(result.get("short_term_messages", [])),
                medium_summary=result.get("current_summary") or "",
                long_term_facts=_facts_to_proto(result.get("relevant_facts", [])),
                semantic_snippets=[],
                total_tokens=0,
                request_id=trace_id,
            )
        except Exception as e:
            logger.error(f"gRPC GetMemoryContext failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise
        finally:
            clear_log_context()

    async def GetShortTermMemory(self, request, context):
        trace_id = _extract_trace_id(context)
        set_log_context(trace_id=trace_id, conversation_id=request.conversation_id)
        try:
            logger.info(f"gRPC GetShortTermMemory request received for conversation={request.conversation_id}")

            messages = await self.context_builder.retrieval_service.get_or_hydrate_recent_messages(
                request.conversation_id,
                limit=request.limit or 20,
            )

            return memory_pb2.GetShortTermMemoryResponse(
                messages=_messages_to_proto(messages),
                total_tokens=0,
                request_id=trace_id,
            )
        except Exception as e:
            logger.error(f"gRPC GetShortTermMemory failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise
        finally:
            clear_log_context()
