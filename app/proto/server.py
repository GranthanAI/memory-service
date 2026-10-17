"""
app/proto/server.py

gRPC Server wrapper for the internal LLM Service.
Allows async serve and clean shutdown sequences.
"""

from concurrent import futures
import logging
from typing import Optional

import grpc
from grpc import aio as grpc_aio

from app.proto import llm_pb2_grpc, memory_pb2_grpc
from app.proto.handlers import LLMServiceHandler
from app.proto.memory_handlers import MemoryContextHandler
from app.services.context_builder import ContextBuilder
from app.services.llm_service import LLMService
from app.clients.embedding_client import EmbeddingClient

logger = logging.getLogger("memory_service.proto.server")


class GRPCServer:
    """
    gRPC Server wrapper hosting both the internal LLM Service (summarization /
    fact-extraction, consumed only by this service's own background workers)
    and the Memory Context Service (consumed by llm-service's ContextCollector)
    on a single grpc.aio server instance.
    """

    def __init__(
        self,
        port: int,
        llm_service: LLMService,
        context_builder: Optional[ContextBuilder] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.port = port
        self.llm_service = llm_service
        self.context_builder = context_builder
        self.embedding_client = embedding_client
        self.server = None

    async def start(self) -> None:
        """
        Starts the gRPC server asynchronously.
        """
        self.server = grpc_aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_receive_message_length", 10 * 1024 * 1024),
                ("grpc.max_send_message_length", 10 * 1024 * 1024),
            ],
        )

        # Register internal LLM Service handler (summarization / fact extraction)
        handler = LLMServiceHandler(self.llm_service)
        llm_pb2_grpc.add_LLMServiceServicer_to_server(handler, self.server)

        # Register Memory Context Service handler (llm-service's baseline context provider)
        if self.context_builder is not None and self.embedding_client is not None:
            memory_handler = MemoryContextHandler(self.context_builder, self.embedding_client)
            memory_pb2_grpc.add_MemoryServiceServicer_to_server(memory_handler, self.server)
        else:
            logger.warning(
                "ContextBuilder/EmbeddingClient not provided — Memory Context Service gRPC "
                "endpoint (GetMemoryContext) will not be registered."
            )

        # Bind port
        bind_address = f"[::]:{self.port}"
        self.server.add_insecure_port(bind_address)
        logger.info(f"Starting gRPC server on {bind_address}...")

        await self.server.start()
        logger.info("gRPC server started successfully.")

    async def stop(self, grace: float = 5.0) -> None:
        """
        Stops the gRPC server gracefully.
        """
        if self.server:
            logger.info("Stopping gRPC server...")
            await self.server.stop(grace)
            logger.info("gRPC server stopped.")
