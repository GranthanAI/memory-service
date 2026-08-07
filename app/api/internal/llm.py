"""
app/api/internal/llm.py

Internal LLM REST API endpoints.
Provides manual summarization and fact extraction routes for testing,
development, and Swagger UI usage.
"""

import logging

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_llm_service
from app.core.security import verify_service_auth
from app.schemas.llm import (
    FactExtractRequest,
    FactExtractResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services.llm_service import LLMService

logger = logging.getLogger("memory_service.api.llm")

router = APIRouter(
    prefix="",
    tags=["internal-llm"],
    dependencies=[Depends(verify_service_auth)],
)


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate summary from chat history",
)
async def summarize(
    request: SummarizeRequest,
    llm_service: LLMService = Depends(get_llm_service),
) -> SummarizeResponse:
    """
    HTTP endpoint to incrementally summarize conversation history.
    """
    logger.info("Received HTTP request to summarize messages.")
    return await llm_service.summarize(request)


@router.post(
    "/facts",
    response_model=FactExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract structured user facts from summary",
)
async def extract_facts(
    request: FactExtractRequest,
    llm_service: LLMService = Depends(get_llm_service),
) -> FactExtractResponse:
    """
    HTTP endpoint to extract category-importance structured user facts.
    """
    logger.info("Received HTTP request to extract facts.")
    return await llm_service.extract_facts(request)
