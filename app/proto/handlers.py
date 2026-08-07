"""
app/proto/handlers.py

gRPC handlers for internal LLMService operations.
Delegates calls to LLMService facade.
"""

import logging

import grpc

from app.proto import llm_pb2, llm_pb2_grpc
from app.schemas.llm import LLMMessage, FactExtractRequest, SummarizeRequest
from app.services.llm_service import LLMService

logger = logging.getLogger("memory_service.proto.handlers")


class LLMServiceHandler(llm_pb2_grpc.LLMServiceServicer):
    """
    gRPC handlers for LLMService operations.
    Delegates calls to LLMService.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def Summarize(self, request, context):
        try:
            logger.info("gRPC Summarize request received.")

            # Map request to Pydantic schema
            messages = [
                LLMMessage(role=msg.role, content=msg.content)
                for msg in request.new_messages
            ]
            summarize_request = SummarizeRequest(
                previous_summary=request.previous_summary, new_messages=messages
            )

            # Call service
            response_dto = await self.llm_service.summarize(summarize_request)

            # Map response to proto message
            return llm_pb2.SummaryResponse(summary=response_dto.summary)

        except Exception as e:
            logger.error(f"gRPC Summarize failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise e

    async def ExtractFacts(self, request, context):
        try:
            logger.info("gRPC ExtractFacts request received.")

            # Map request
            extract_request = FactExtractRequest(summary=request.summary)

            # Call service
            response_dto = await self.llm_service.extract_facts(extract_request)

            # Map response
            facts_proto = [
                llm_pb2.ExtractedFact(
                    category=f.category,
                    importance=f.importance,
                    statement=f.statement,
                )
                for f in response_dto.facts
            ]
            return llm_pb2.FactResponse(facts=facts_proto)

        except Exception as e:
            logger.error(f"gRPC ExtractFacts failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise e
