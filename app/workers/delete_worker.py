"""
app/workers/delete_worker.py

Durable memory deletion worker.
Polls deletion requests from Kafka, and deletes matching facts from Cassandra and Milvus.
"""

import asyncio
import json
import logging
import uuid
from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.milvus_repository import MilvusRepository

logger = logging.getLogger("memory_service.workers.delete_worker")


class DeleteWorker:
    """
    Background worker that consumes memory.delete.request events, deleting records
    durable from both Cassandra and the Milvus index.
    """

    def __init__(
        self,
        cassandra_session,
        milvus_repo: MilvusRepository,
        bootstrap_servers: str = settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id: str = "delete-worker-group"
    ):
        self.session = cassandra_session
        self.milvus_repo = milvus_repo
        self.cassandra_repo = CassandraRepository(self.session)
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer = None
        self.is_running = False
        self._task = None

    async def start(self) -> None:
        """Starts the DeleteWorker consumer loop."""
        logger.info(f"Starting DeleteWorker on topic {settings.KAFKA_DELETE_TOPIC}")
        try:
            self._consumer = AIOKafkaConsumer(
                settings.KAFKA_DELETE_TOPIC,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                enable_auto_commit=False,
                session_timeout_ms=settings.KAFKA_SESSION_TIMEOUT_MS,
                max_poll_interval_ms=settings.KAFKA_MAX_POLL_INTERVAL_MS
            )
            await self._consumer.start()
            self.is_running = True
            self._task = asyncio.create_task(self._loop())
            logger.info("DeleteWorker started successfully.")
        except Exception as e:
            logger.critical(f"Failed to start DeleteWorker: {e}")
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
                        try:
                            payload = json.loads(msg.value.decode("utf-8"))
                            user_id = payload.get("user_id")
                            if not user_id:
                                logger.error("Missing user_id in delete request. Skipping.")
                                continue

                            # Scenario 1: Deletion of specific fact_id
                            if "fact_id" in payload and "category" in payload:
                                fact_id_str = payload["fact_id"]
                                category = payload["category"]
                                fact_id = uuid.UUID(fact_id_str)

                                logger.info(f"Deleting specific fact {fact_id} under category '{category}' for user {user_id}")
                                self.cassandra_repo.delete_fact(user_id, category, fact_id)
                                self.milvus_repo.delete_fact(user_id, fact_id_str)

                            # Scenario 2: Deletion of facts associated with conversation_id
                            elif "conversation_id" in payload:
                                conversation_id = payload["conversation_id"]
                                logger.info(f"Deleting all facts for conversation {conversation_id} user {user_id}")
                                
                                # Iterate over standard categories to find and purge matching conversation facts
                                standard_categories = ["preferences", "habits", "hobbies", "general"]
                                for cat in standard_categories:
                                    facts = self.cassandra_repo.get_facts(user_id, cat)
                                    for fact in facts:
                                        if fact.get("conversation_id") == conversation_id:
                                            fact_id = fact["fact_id"]
                                            logger.info(f"Purging fact {fact_id} associated with conversation {conversation_id}")
                                            self.cassandra_repo.delete_fact(user_id, cat, fact_id)
                                            self.milvus_repo.delete_fact(user_id, str(fact_id))

                        except Exception as e:
                            logger.error(f"Error in DeleteWorker processing job: {e}")

                    # Commit partition batch offsets
                    await self._consumer.commit()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DeleteWorker polling loop error: {e}")
                await asyncio.sleep(2.0)

    async def stop(self) -> None:
        """Gracefully stops the worker connection and loops."""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
            logger.info("DeleteWorker consumer stopped.")
