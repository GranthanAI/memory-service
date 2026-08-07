"""
app/proto/server.py

gRPC Server wrapper for the internal LLM Service.
Allows async serve and clean shutdown sequences.
"""

from concurrent import futures
import logging

import grpc
from grpc import aio as grpc_aio

from app.proto import llm_pb2_grpc
from app.proto.handlers import LLMServiceHandler
from app.services.llm_service import LLMService

logger = logging.getLogger("memory_service.proto.server")


class GRPCServer:
    """
    gRPC Server wrapper for the internal LLM Service.
    Allows async serve and clean shutdown sequences.
    """

    def __init__(self, port: int, llm_service: LLMService):
        self.port = port
        self.llm_service = llm_service
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

        # Register handler
        handler = LLMServiceHandler(self.llm_service)
        llm_pb2_grpc.add_LLMServiceServicer_to_server(handler, self.server)

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
