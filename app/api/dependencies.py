"""
app/api/dependencies.py

FastAPI Depends() providers for the Memory Service API layer.

All singletons (services, repositories, clients) are resolved from
app.state.container, which is populated during the lifespan startup hook.
This keeps endpoints thin and fully testable — override these in tests
with dependency_overrides.
"""

import logging
from typing import AsyncGenerator

from fastapi import Request

from app.services.context_builder import ContextBuilder
from app.services.llm_service import LLMService
from app.db.cassandra import get_session
from app.db.redis import get_redis_client

logger = logging.getLogger("memory_service.api.dependencies")


def get_container(request: Request):
    """
    Resolves the DI container from FastAPI app state.
    The container is populated by the lifespan startup hook.
    """
    return request.app.state.container


def get_context_builder(request: Request) -> ContextBuilder:
    """
    Provides the ContextBuilder singleton for context retrieval endpoints.
    Pulls from the container set up during app lifespan startup.
    """
    container = get_container(request)
    return container.context_builder





def get_cassandra_health_session(request: Request):
    """
    Provides the Cassandra session for health probe queries.
    Uses the global session singleton initialized during startup.
    """
    return get_session()


async def get_redis_health_client(request: Request):
    """
    Provides the Redis async client for health probe PING.
    Uses the global client singleton initialized during startup.
    """
    return get_redis_client()


def get_llm_service(request: Request) -> LLMService:
    """
    Provides the LLMService singleton for summarization and fact extraction endpoints.
    """
    container = get_container(request)
    return container.llm_service

