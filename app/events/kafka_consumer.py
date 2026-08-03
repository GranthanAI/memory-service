"""
app/events/kafka_consumer.py

Kafka Event Consumer daemon wrapper.
Listens to conversation message topics and processes messages through EventDispatcher.
"""

import asyncio
import json
import logging
from typing import List, Optional
from aiokafka import AIOKafkaConsumer

from app.core.config import settings
from app.events.dispatcher import EventDispatcher

logger = logging.getLogger("memory_service.events.kafka_consumer")


class KafkaEventConsumer:
    """
    Polls Kafka for conversation events, passes them to EventDispatcher,
    and commits offsets manually to guarantee at-least-once processing semantics.
    """

    def __init__(
        self,
        dispatcher: EventDispatcher,
        bootstrap_servers: str = settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id: str = settings.KAFKA_GROUP_ID,
        topics: Optional[List[str]] = None
    ):
        self.dispatcher = dispatcher
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics or ["chat.message.created", "chat.response.completed"]
        self._consumer = None
        self.is_running = False
        self._task = None

    async def start(self) -> None:
        """Initializes and runs the consumer polling loop."""
        logger.info(f"Initializing Kafka Consumer: topics={self.topics}, group={self.group_id}")
        try:
            self._consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                enable_auto_commit=False,  # Enforce manual offset commits
                session_timeout_ms=settings.KAFKA_SESSION_TIMEOUT_MS,
                max_poll_interval_ms=settings.KAFKA_MAX_POLL_INTERVAL_MS
            )
            await self._consumer.start()
            self.is_running = True
            self._task = asyncio.create_task(self._loop())
            logger.info("Kafka Consumer daemon started successfully.")
        except Exception as e:
            logger.critical(f"Failed to start Kafka Consumer: {e}")
            raise e

    async def _loop(self) -> None:
        """Primary polling execution loop."""
        while self.is_running:
            try:
                # Poll batches from Kafka partitions
                msg_batch = await self._consumer.getmany(timeout_ms=1000)
                if not msg_batch:
                    continue

                for tp, messages in msg_batch.items():
                    for msg in messages:
                        try:
                            # Deserialize event payload
                            raw_event = json.loads(msg.value.decode("utf-8"))
                            # Route through idempotency gate and database commits
                            await self.dispatcher.dispatch(raw_event)
                        except Exception as e:
                            logger.error(
                                f"Failed to process message from topic {tp.topic} partition {tp.partition}: {e}"
                            )
                            # Do not stall on deserialization/payload errors.
                            # Standard Cassandra/network connection issues will raise out of consumer loop.
                            if isinstance(e, (json.JSONDecodeError, KeyError, ValueError)):
                                continue
                            raise e

                    # Commit partition offsets only after all messages in batch are written to Cassandra
                    await self._consumer.commit()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Kafka consumer polling loop encountered an error: {e}")
                # Backoff delay before reconnect attempt
                await asyncio.sleep(2.0)

    async def stop(self) -> None:
        """Gracefully halts the consumer loop and clean connection handles."""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._consumer:
            logger.info("Stopping Kafka Consumer connection...")
            await self._consumer.stop()
            self._consumer = None
            logger.info("Kafka Consumer stopped.")
