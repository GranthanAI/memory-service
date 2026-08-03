"""
app/events/kafka_producer.py

AIOKafka Producer wrapper responsible for publishing outbox jobs to Kafka topics.
Preserves strict chronological ordering by routing messages using the conversation_id as the key.
"""

import json
import logging
from typing import Any, Dict
from aiokafka import AIOKafkaProducer

from app.core.config import settings

logger = logging.getLogger("memory_service.events.kafka_producer")


class KafkaProducer:
    """
    Wraps AIOKafkaProducer to provide safe async connection management
    and ordered message dispatching.
    """

    def __init__(self, bootstrap_servers: str = settings.KAFKA_BOOTSTRAP_SERVERS):
        self.bootstrap_servers = bootstrap_servers
        self._producer = None

    async def start(self) -> None:
        """Starts the underlying Kafka producer connection pool."""
        logger.info(f"Initializing Kafka Producer connecting to: {self.bootstrap_servers}")
        try:
            self._producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)
            await self._producer.start()
            logger.info("Kafka Producer started successfully.")
        except Exception as e:
            logger.critical(f"Failed to start Kafka Producer: {e}")
            raise e

    async def stop(self) -> None:
        """Stops the underlying Kafka producer gracefully."""
        if self._producer:
            logger.info("Stopping Kafka Producer...")
            await self._producer.stop()
            self._producer = None
            logger.info("Kafka Producer stopped.")

    async def publish_task(self, topic: str, conversation_id: str, payload: Dict[str, Any]) -> None:
        """
        Publishes a task payload to a Kafka topic.
        Ordering is strictly preserved by setting conversation_id as the partition key.
        """
        if not self._producer:
            raise RuntimeError("Kafka Producer is not started. Call start() before publishing.")

        key_bytes = conversation_id.encode("utf-8")
        value_bytes = json.dumps(payload).encode("utf-8")

        logger.debug(f"Publishing event to '{topic}' for conversation: {conversation_id}")
        await self._producer.send_and_wait(topic, key=key_bytes, value=value_bytes)
