"""
app/workers/embedding_worker.py

Embedding worker daemon.
Polls embedding requests from Kafka, generates fact embeddings, and merges them using LongMemoryService.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import List, Tuple
from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.models.memory import MemoryState
from app.clients.embedding_client import EmbeddingClient
from app.services.memory_service import MemoryService
from app.services.long_memory_service import LongMemoryService

logger = logging.getLogger("memory_service.workers.embedding_worker")


def parse_fact_string(fact_str: str) -> Tuple[str, float, str]:
    """
    Parses a formatted fact string 'category:importance:statement' (e.g. 'preferences:0.85:Likes tea').
    Falls back to defaults if parsing fails.
    """
    parts = fact_str.split(":", 2)
    if len(parts) == 3:
        try:
            category = parts[0].strip().lower()
            importance = float(parts[1].strip())
            statement = parts[2].strip()
            return category, importance, statement
        except ValueError:
            pass
    return "general", 0.5, fact_str.strip()


class EmbeddingWorker:
    """
    Background worker that consumes memory.embedding.request events, generates
    vectors for each fact, and applies the Fact Merge Policy.
    """

    def __init__(
        self,
        cassandra_session,
        memory_service: MemoryService,
        long_memory_service: LongMemoryService,
        embedding_client: EmbeddingClient,
        bootstrap_servers: str = settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id: str = "embedding-worker-group"
    ):
        self.session = cassandra_session
        self.memory_service = memory_service
        self.long_memory_service = long_memory_service
        self.embedding_client = embedding_client
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer = None
        self.is_running = False
        self._task = None

    async def start(self) -> None:
        """Starts the EmbeddingWorker consumer loop."""
        logger.info(f"Starting EmbeddingWorker on topic {settings.KAFKA_EMBEDDING_TOPIC}")
        try:
            self._consumer = AIOKafkaConsumer(
                settings.KAFKA_EMBEDDING_TOPIC,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                enable_auto_commit=False,
                session_timeout_ms=settings.KAFKA_SESSION_TIMEOUT_MS,
                max_poll_interval_ms=settings.KAFKA_MAX_POLL_INTERVAL_MS
            )
            await self._consumer.start()
            self.is_running = True
            self._task = asyncio.create_task(self._loop())
            logger.info("EmbeddingWorker started successfully.")
        except Exception as e:
            logger.critical(f"Failed to start EmbeddingWorker: {e}")
            raise e

    async def _loop(self) -> None:
        """Primary message loop."""
        while self.is_running:
            try:
                msg_batch = await self._consumer.getmany(timeout_ms=1000)
                if not msg_batch:
                    continue

                for tp, messages in msg_batch.items():
                    for msg in messages:
                        payload = None
                        conversation_id = "unknown"
                        try:
                            payload = json.loads(msg.value.decode("utf-8"))
                            conversation_id = payload["conversation_id"]
                            user_id = payload.get("user_id", "unknown_user")
                            facts_list = payload.get("facts", [])

                            logger.info(f"Processing embedding requests for conversation: {conversation_id}")

                            incoming_facts = []
                            for fact_str in facts_list:
                                category, importance, statement = parse_fact_string(fact_str)
                                if not statement:
                                    continue

                                # Generate embedding via the abstract client
                                vector = await self.embedding_client.generate_embedding(statement)
                                
                                incoming_facts.append({
                                    "statement": statement,
                                    "category": category,
                                    "importance": importance,
                                    "vector": vector
                                })

                            # Merge facts with merge policy
                            if incoming_facts:
                                stats = await self.long_memory_service.merge_user_facts(
                                    user_id=user_id,
                                    conversation_id=conversation_id,
                                    incoming_facts=incoming_facts
                                )
                                logger.info(f"Fact merge statistics for user {user_id}: {stats}")

                            # Transition snapshot to READY -> ACTIVE
                            await self.memory_service.transition_state(
                                conversation_id, MemoryState.READY
                            )
                            await self.memory_service.transition_state(
                                conversation_id, MemoryState.ACTIVE
                            )

                        except Exception as e:
                            logger.error(f"Error in EmbeddingWorker processing job: {e}")
                            if payload and conversation_id != "unknown":
                                # Handle retry/DLQ scheduling
                                await self.memory_service.handle_failure(
                                    conversation_id=conversation_id,
                                    failed_state=MemoryState.EMBEDDING_PENDING,
                                    job_type="embedding",
                                    payload=payload,
                                    error_msg=str(e),
                                    attempt_count=payload.get("attempt_count", 0)
                                )

                    # Commit partition batch offsets
                    await self._consumer.commit()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"EmbeddingWorker polling loop error: {e}")
                await asyncio.sleep(2.0)

    async def stop(self) -> None:
        """Gracefully stops the worker connection and loops."""
        logger.info("Initiating graceful shutdown for EmbeddingWorker...")
        self.is_running = False
        if self._task:
            try:
                # Wait up to 10 seconds for the active message batch to finish processing
                await asyncio.wait_for(asyncio.shield(self._task), timeout=10.0)
                logger.info("EmbeddingWorker loop exited cleanly.")
            except asyncio.TimeoutError:
                logger.warning("Graceful shutdown timed out. Forcefully cancelling EmbeddingWorker loop.")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
            logger.info("EmbeddingWorker consumer stopped.")
