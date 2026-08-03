"""
app/core/container.py

Dependency Injection Container for the Memory Service.
Centralizes resource allocation, singleton provider registration, and service wiring.
"""

import logging
from typing import Optional

from app.core.config import settings
from app.db.cassandra import get_session
from app.db.redis import get_redis_client
from app.clients.llm_client import AsyncGRPCConnectionPool, LLMClient
from app.clients.graph_client import GraphClient
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.redis_repository import RedisRepository
from app.repositories.processed_event_repository import ProcessedEventRepository
from app.repositories.memory_repository import MemoryRepository
from app.repositories.milvus_repository import MilvusRepository
from app.services.snapshot_service import SnapshotService
from app.services.memory_service import MemoryService
from app.services.summary_service import SummaryService
from app.services.long_memory_service import LongMemoryService
from app.services.ranking_service import RankingService
from app.services.retrieval_service import RetrievalService
from app.services.context_builder import ContextBuilder

logger = logging.getLogger("memory_service.core.container")


class Container:
    """
    IOC Container managing resources and dependency wiring for services,
    repositories, and clients.
    """

    def __init__(self):
        # Database Sessions & Clients
        self.cassandra_session = None
        self.redis_client = None
        self.milvus_repo = None
        self.llm_pool: Optional[AsyncGRPCConnectionPool] = None
        self.llm_client: Optional[LLMClient] = None
        self.graph_client: Optional[GraphClient] = None

        # Repositories
        self.cassandra_repo: Optional[CassandraRepository] = None
        self.redis_repo: Optional[RedisRepository] = None
        self.processed_event_repo: Optional[ProcessedEventRepository] = None
        self.memory_repo: Optional[MemoryRepository] = None

        # Services
        self.snapshot_service: Optional[SnapshotService] = None
        self.memory_service: Optional[MemoryService] = None
        self.summary_service: Optional[SummaryService] = None
        self.long_memory_service: Optional[LongMemoryService] = None
        self.ranking_service: Optional[RankingService] = None
        self.retrieval_service: Optional[RetrievalService] = None
        self.context_builder: Optional[ContextBuilder] = None

    async def init_resources(self) -> None:
        """
        Bootstraps and wires all providers.
        Assumes initialize_db_sessions() has been executed.
        """
        logger.info("Initializing dependency injection container providers...")

        # 1. Resolve connection clients from session pools
        self.cassandra_session = get_session()
        self.redis_client = get_redis_client()
        self.milvus_repo = MilvusRepository()

        # 2. Build gRPC clients
        self.llm_pool = AsyncGRPCConnectionPool(
            target=f"{settings.LLM_SERVICE_HOST}:{settings.LLM_SERVICE_PORT}",
            pool_size=settings.GRPC_POOL_SIZE
        )
        await self.llm_pool.connect()
        self.llm_client = LLMClient(self.llm_pool)

        self.graph_client = GraphClient(base_url=settings.GRAPH_SERVICE_URL)

        # 3. Setup Repositories
        self.cassandra_repo = CassandraRepository(self.cassandra_session)
        self.redis_repo = RedisRepository(self.redis_client)
        self.processed_event_repo = ProcessedEventRepository(self.cassandra_session)
        self.memory_repo = MemoryRepository(self.cassandra_repo, self.redis_repo)

        # 4. Setup Services
        self.snapshot_service = SnapshotService(self.cassandra_session, self.redis_repo)
        self.memory_service = MemoryService(self.memory_repo, self.cassandra_repo)
        self.summary_service = SummaryService(self.memory_repo, self.cassandra_repo, self.llm_client)
        self.long_memory_service = LongMemoryService(self.cassandra_repo, self.milvus_repo)
        self.ranking_service = RankingService()

        # 5. Setup Retrieval & Context services
        self.retrieval_service = RetrievalService(
            memory_repo=self.memory_repo,
            milvus_repo=self.milvus_repo
        )
        self.context_builder = ContextBuilder(
            retrieval_service=self.retrieval_service,
            graph_client=self.graph_client
        )

        logger.info("✓ Dependency injection container wired successfully.")

    async def shutdown_resources(self) -> None:
        """Gracefully tears down clients and active connection pools."""
        logger.info("Tearing down container clients...")
        if self.llm_pool:
            try:
                await self.llm_pool.close()
                logger.info("✓ LLM Connection Pool closed.")
            except Exception as e:
                logger.error(f"Error closing LLM pool: {e}")
        logger.info("Container teardown finished.")
