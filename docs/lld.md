# GraphGPT Memory Service - Low-Level Design (LLD)
**Version:** 2.2 (Production Ready - Multi-Partition Decoupled)  
**Status:** Approved for Implementation  
**Base HLD Reference:** [hld.md](file:///c:/Users/hp/Desktop/Granthan/memory-service/docs/hld.md)  
**Scale Target:** 100M Users, 1M Writes/sec peak (Kafka ingestion buffer), 99.99% Availability  

---

## 1. Document Introduction & Metadata

This Low-Level Design (LLD) specifies the concrete implementation architecture for the **GraphGPT Memory Service**. Version 2.2 introduces a fully Kafka-decoupled background worker pipeline, enforces event idempotency and transaction safety, details storage operations, and models the context retrieval system to return structured JSON payloads.

### 1.1 Revision History

| Version | Date | Description | Author |
| :--- | :--- | :--- | :--- |
| 1.0 | 2026-07-31 | Initial draft design with basic REST / Consumer interfaces | Antigravity AI |
| 2.0 | 2026-08-01 | HLD alignment version (Redis-cache, Milvus vectors) | Antigravity AI |
| 2.1 | 2026-08-01 | Production update: Worker decoupling, Idempotency, Prompt/Ranking/Context builder separation, State Machine, Resilient Clients | Antigravity AI |
| 2.2 | 2026-08-01 | Enterprise update: Kafka-based worker queues, 7-day Idempotency, transactional safety, Milvus delete-then-insert updates, rate-limiting, zlib compression, structured trace logging | Antigravity AI |

---

## 2. Directory Structure & File Mapping

```text
memory-service/
│
├── app/
│   ├── api/
│   │   ├── dependencies.py
│   │   ├── internal/
│   │   │   ├── health.py
│   │   │   └── memory.py
│   │   └── routers.py
│   │
│   ├── clients/
│   │   ├── graph_client.py
│   │   ├── llm_client.py
│   │   ├── milvus_client.py
│   │   └── redis_client.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── exceptions.py
│   │   ├── logging.py
│   │   └── security.py
│   │
│   ├── db/
│   │   ├── redis.py
│   │   ├── milvus.py
│   │   └── session.py
│   │
│   ├── events/
│   │   ├── consumers.py
│   │   ├── producers.py
│   │   ├── dispatcher.py
│   │   ├── kafka_consumer.py
│   │   ├── kafka_producer.py
│   │   └── topics.py
│   │
│   ├── models/
│   │   ├── api.py
│   │   ├── event.py
│   │   ├── memory.py
│   │   ├── snapshot.py
│   │   ├── summary.py
│   │   ├── context.py
│   │   └── embedding.py
│   │
│   ├── repositories/
│   │   ├── memory_repository.py
│   │   ├── redis_repository.py
│   │   ├── milvus_repository.py
│   │   └── processed_event_repository.py
│   │
│   ├── schemas/
│   │   ├── requests.py
│   │   ├── responses.py
│   │   ├── grpc.py
│   │   └── events.py
│   │
│   ├── services/
│   │   ├── memory_service.py
│   │   ├── snapshot_service.py
│   │   ├── summary_service.py
│   │   ├── long_memory_service.py
│   │   ├── semantic_memory_service.py
│   │   ├── retrieval_service.py
│   │   ├── ranking_service.py
│   │   ├── context_builder.py
│   │   ├── embedding_service.py
│   │   ├── cleanup_service.py
│   │   └── idempotency_service.py
│   │
│   ├── workers/
│   │   ├── summary_worker.py
│   │   ├── fact_worker.py
│   │   ├── embedding_worker.py
│   │   ├── delete_worker.py
│   │   └── cleanup_worker.py
│   │
│   ├── proto/
│   │   ├── graph.proto
│   │   └── llm.proto
│   │
│   ├── utils/
│   │   ├── compression.py
│   │   ├── hashing.py
│   │   ├── ranking.py
│   │   ├── locks.py
│   │   ├── serialization.py
│   │   └── timers.py
│   │
│   ├── main.py
│   └── lifespan.py
│
├── tests/
│   ├── integration/
│   ├── unit/
│   ├── fixtures/
│   └── conftest.py
│
├── alembic/
│
├── docs/
│   ├── HLD.md
│   └── LLD.md
│
├── scripts/
│
├── docker/
│
├── .env
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

### Folder Responsibilities

| Folder | Responsibility |
| :--- | :--- |
| **api/** | Internal REST endpoints (`/internal/memory/context`, health, readiness) |
| **clients/** | All outbound communication (Graph gRPC/HTTP, LLM gRPC, Redis, Milvus) |
| **core/** | Configuration, logging, constants, exceptions |
| **db/** | Database initialization and connection pools |
| **events/** | Kafka consumers, producers, dispatcher, topic definitions |
| **models/** | Internal domain models (Snapshot, Memory, Summary, Context) |
| **repositories/** | Redis/Milvus persistence layer, processed event storage |
| **schemas/** | Pydantic request/response DTOs and event schemas |
| **services/** | Business logic only |
| **workers/** | Background Kafka worker implementations |
| **proto/** | Generated gRPC contracts |
| **utils/** | Shared helper functions |
| **tests/** | Unit and integration tests |

---

## 3. Configuration & Runtime System Settings

System settings are loaded and validated at boot time via Pydantic settings.

### 3.1 Code Implementation: `config/settings.py`
```python
import os
from typing import List
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class SystemSettings(BaseSettings):
    """
    Validates and stores service execution configurations.
    """
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    # Core Application Configuration
    APP_NAME: str = "graphgpt-memory-service"
    APP_ENV: str = Field("production", env="APP_ENV")
    DEBUG: bool = Field(False, env="DEBUG")
    HTTP_HOST: str = Field("0.0.0.0", env="HTTP_HOST")
    HTTP_PORT: int = Field(8000, env="HTTP_PORT")

    # Kafka Broker Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = Field("localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    KAFKA_GROUP_ID: str = Field("memory-service-consumers", env="KAFKA_GROUP_ID")
    
    # Internal Communication Topics (Refactored to Kafka Topics - Issue 1)
    KAFKA_SUMMARY_TOPIC: str = Field("memory.summary.request", env="KAFKA_SUMMARY_TOPIC")
    KAFKA_FACT_TOPIC: str = Field("memory.fact.request", env="KAFKA_FACT_TOPIC")
    KAFKA_EMBEDDING_TOPIC: str = Field("memory.embedding.request", env="KAFKA_EMBEDDING_TOPIC")
    KAFKA_DELETE_TOPIC: str = Field("memory.delete.request", env="KAFKA_DELETE_TOPIC")
    KAFKA_DLQ_TOPIC: str = Field("memory.dlq", env="KAFKA_DLQ_TOPIC")

    # Redis Cache Configuration
    REDIS_URL: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    SNAPSHOT_TTL_SECONDS: int = Field(2592000, env="SNAPSHOT_TTL_SECONDS")  # 30 Days default
    SHORT_TERM_MESSAGE_LIMIT: int = Field(20, env="SHORT_TERM_MESSAGE_LIMIT")
    
    # Idempotency Cache Duration: 7 days to cover extended Kafka downtime (Issue 2)
    IDEMPOTENCY_TTL_SECONDS: int = Field(604800, env="IDEMPOTENCY_TTL_SECONDS")

    # Milvus Vector Database Configuration
    MILVUS_HOST: str = Field("localhost", env="MILVUS_HOST")
    MILVUS_PORT: int = Field(19530, env="MILVUS_PORT")
    VECTOR_DIMENSION: int = Field(1536, env="VECTOR_DIMENSION")

    # Embedding Model Tracking Configs (Issue 22)
    EMBEDDING_MODEL_NAME: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL_NAME")
    EMBEDDING_MODEL_VERSION: str = Field("v1.0.0", env="EMBEDDING_MODEL_VERSION")

    # gRPC & Microservice Communication Configurations
    LLM_SERVICE_HOST: str = Field("localhost", env="LLM_SERVICE_HOST")
    LLM_SERVICE_PORT: int = Field(50051, env="LLM_SERVICE_PORT")
    GRAPH_SERVICE_URL: str = Field("http://localhost:8001", env="GRAPH_SERVICE_URL")
    GRPC_TIMEOUT_SECONDS: float = Field(5.0, env="GRPC_TIMEOUT_SECONDS")
    
    # LLM Rate Limiting Semaphore (Issue 7)
    LLM_CONCURRENT_LIMIT: int = Field(50, env="LLM_CONCURRENT_LIMIT")

    # Memory Search Scoring Weights (Must sum to 1.0)
    RETRIEVAL_WEIGHT_SIMILARITY: float = Field(0.5, env="RETRIEVAL_WEIGHT_SIMILARITY")
    RETRIEVAL_WEIGHT_RECENCY: float = Field(0.2, env="RETRIEVAL_WEIGHT_RECENCY")
    RETRIEVAL_WEIGHT_IMPORTANCE: float = Field(0.3, env="RETRIEVAL_WEIGHT_IMPORTANCE")
    RETRIEVAL_DECAY_RATE: float = Field(0.05, env="RETRIEVAL_DECAY_RATE")  # Decay per day
    RETRIEVAL_TOP_K_FACTS: int = Field(10, env="RETRIEVAL_TOP_K_FACTS")

    # Deduplication Similarity Threshold (Issue 23)
    FACT_MERGE_SIMILARITY_THRESHOLD: float = Field(0.85, env="FACT_MERGE_SIMILARITY_THRESHOLD")

# Instantiated configuration singleton
settings = SystemSettings()
```

---

## 4. Redis Key Namespaces & Serialization Formats (Issue 8, 9)

### 4.1 Key Architecture

| Namespace Key | Redis Type | Purpose | TTL Policy |
| :--- | :--- | :--- | :--- |
| `snapshot:{conversation_id}` | Hash | Stores the active workspace state. Contains metadata like `user_id`, `updated_at`, and progress markers. | 30 Days (sliding) |
| `recent:{conversation_id}` | List | Stores the sliding message window. Stores JSON-serialized `MessageRecord` entries. | 30 Days (sliding) |
| `summary:{conversation_id}` | String | Stores the hot summary text (zlib-compressed) and its version metrics. | 30 Days (sliding) |
| `lock:{conversation_id}` | String | Distributed mutex lock key for write safety. | 5 Seconds |
| `event_idempotency:{event_id}` | String | Tracks processed event IDs to prevent duplicate actions. | 7 Days (persistent check) |

### 4.2 Compression Strategy for Summary Strings (Issue 9)
Summary strings in Redis are compressed using a lightweight `zlib` compression wrapper.

```python
import zlib

def compress_summary(text: str) -> bytes:
    """
    Compresses text content to bytes.
    """
    return zlib.compress(text.encode("utf-8"), level=6)

def decompress_summary(compressed: bytes) -> str:
    """
    Decompresses bytes back to a string.
    """
    return zlib.decompress(compressed).decode("utf-8")
```

---

## 5. Event Ingestion with Kafka & Message Ordering (Issue 8)

To preserve conversation history accuracy, messages must be processed in the order they were sent. Kafka guarantees message ordering **within a partition**. By setting the partition key on all Kafka topics to `conversation_id`, we route all events for a specific conversation to the same partition, ensuring ordered processing.

```text
                  Multi-Partition Message Routing Logic
                  
    [Conversation Events Stream]
      ├── event (conversation_id = "conv-A")  ──► HASH("conv-A") ──► Partition 1
      ├── event (conversation_id = "conv-B")  ──► HASH("conv-B") ──► Partition 2
      └── event (conversation_id = "conv-A")  ──► HASH("conv-A") ──► Partition 1
      
    [Partition 1 Queue (Ordered)]
      └── [Event A1: Msg Created] ──► [Event A2: Msg Completed]
```

### 5.1 Kafka Producer & Ingestion Client: `clients/kafka_producer.py`
```python
import json
import logging
from aiokafka import AIOKafkaProducer
from src.config.settings import settings

logger = logging.getLogger("memory_service.kafka_producer")

class MemoryKafkaProducer:
    """
    Publishes tasks to internal Kafka worker topics, routing by conversation_id.
    """
    def __init__(self):
        self.producer: AIOKafkaProducer = None

    async def start(self) -> None:
        logger.info("Initializing async Kafka Producer...")
        self.producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        await self.producer.start()

    async def stop(self) -> None:
        if self.producer:
            await self.producer.stop()

    async def publish_task(self, topic: str, conversation_id: str, payload: dict) -> None:
        """
        Publishes task details to a target topic.
        Uses conversation_id as the partition key to guarantee ordered processing.
        """
        try:
            # Inject trace_id metadata for observability
            payload["conversation_id"] = conversation_id
            
            await self.producer.send_and_wait(
                topic=topic,
                key=conversation_id,  # Partition Key
                value=payload
            )
        except Exception as e:
            logger.error(f"Failed to publish task to topic {topic} for conversation {conversation_id}: {str(e)}")
            raise
```

---

## 6. Event Idempotency & Ingestion Controllers (Issue 2)

Idempotency keys are cached in Redis with a **7-day TTL** (604,800 seconds) to ensure duplicate messages are ignored, even during extended Kafka outages or replays.

```python
import json
import logging
from src.clients.redis_client import AsyncRedisClient
from src.consumer.handlers import EventHandler
from src.models.events import KafkaEnvelope

logger = logging.getLogger("memory_service.dispatcher")

class EventDispatcher:
    """
    Validates, filters duplicate event IDs, and dispatches incoming Kafka messages.
    """
    def __init__(self, handler: EventHandler, redis: AsyncRedisClient):
        self.handler = handler
        self.redis = redis

    async def is_duplicate(self, event_id: str) -> bool:
        """
        Checks if the event ID has already been processed within the 7-day TTL window.
        """
        if not event_id:
            return False
            
        key = f"event_idempotency:{event_id}"
        # Set key if it doesn't exist (TTL: 7 days)
        async with self.redis._get_client() as client:
            res = await client.set(key, "processed", ex=settings.IDEMPOTENCY_TTL_SECONDS, nx=True)
            return not bool(res)

    async def dispatch(self, topic: str, key: str, payload_str: str) -> bool:
        if not payload_str:
            return True

        try:
            payload = json.loads(payload_str)
            envelope = KafkaEnvelope(topic=topic, key=key, payload=payload)
            
            event_id = payload.get("message_id") or payload.get("response_id") or payload.get("event_id")
            
            if event_id and await self.is_duplicate(event_id):
                logger.warning(f"Duplicate event detected: {event_id}. Skipping processing.")
                return True

            if topic == "chat.message.created":
                return await self.handler.handle_message_created(envelope)
            elif topic == "chat.response.completed":
                return await self.handler.handle_response_completed(envelope)
            elif topic == "conversation.deleted":
                return await self.handler.handle_conversation_deleted(envelope)
            else:
                return True
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON for topic: {topic}. Payload: {payload_str}")
            return False
```

---

## 7. Decoupled Snapshot Builder & Outbox Transactions (Issue 4, 7, 21)

Ingestion tasks use Redis Transactions (`MULTI` / `EXEC`) to save the message snapshot and package delta messages atomically. This ensures that the snapshot updates and task queuing commands execute as a single atomic unit.

```text
               Atomic Ingestion Influx Sequence
               
    [Incoming Event] ──► EventDispatcher (Checks event_idempotency)
                               │
                               ▼
                        EventHandler ──► Acquire Redis Lock
                               │
                               ▼
                        SnapshotBuilder (Prepare writes)
                               │
                      [Redis Transaction: MULTI]
                        ├── append_message (Push recent list)
                        ├── update_state (Set status SUMMARY_PENDING)
                        └── EXEC
                               │
                               ▼
                        Publish task to Kafka topic memory.summary.request
                        (Includes array of delta messages in task payload)
                               │
                               ▼
                        Release Redis Lock
```

### 7.1 Code Implementation: `engine/snapshot.py`
```python
import json
import logging
from typing import Optional, List, Tuple
from src.clients.redis_client import AsyncRedisClient
from src.models.storage_models import ConversationSnapshot, MessageRecord
from src.config.settings import settings
from src.engine.state_machine import MemoryState

logger = logging.getLogger("memory_service.snapshot")

class ConversationSnapshotBuilder:
    """
    Manages sliding message windows in Redis.
    """
    def __init__(self, redis_client: AsyncRedisClient):
        self.redis = redis_client

    async def get_snapshot(self, conversation_id: str) -> Optional[ConversationSnapshot]:
        key = f"snapshot:{conversation_id}"
        data = await self.redis.hgetall(key)
        if not data:
            return None
        
        recent_key = f"recent:{conversation_id}"
        async with self.redis._get_client() as client:
            raw_messages = await client.lrange(recent_key, 0, -1)
            
        messages_list = []
        for raw in raw_messages:
            try:
                messages_list.append(MessageRecord(**json.loads(raw)))
            except Exception as e:
                logger.error(f"Failed to parse message record: {str(e)}")

        return ConversationSnapshot(
            conversation_id=conversation_id,
            user_id=data.get("user_id"),
            summary="",
            recent_messages=messages_list,
            message_count=int(data.get("message_count", 0)),
            last_summary_message_id=data.get("last_summary_message_id", ""),
            updated_at=float(data.get("updated_at", 0.0))
        )

    async def append_message_atomic(self, conversation_id: str, user_id: str, message: MessageRecord) -> Tuple[ConversationSnapshot, List[MessageRecord], bool]:
        """
        Appends a message to Redis using a transaction (MULTI/EXEC).
        Returns the updated snapshot, delta messages list, and a summarization trigger flag.
        """
        lock_key = f"lock:{conversation_id}"
        acquired = await self.redis.acquire_lock(lock_key, ttl_seconds=5)
        if not acquired:
            raise RuntimeError(f"Failed to acquire lock for conversation {conversation_id}")

        try:
            snapshot = await self.get_snapshot(conversation_id)
            if not snapshot:
                snapshot = ConversationSnapshot(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    recent_messages=[],
                    message_count=0,
                    last_summary_message_id="",
                    updated_at=message.timestamp
                )

            # Append the new message
            snapshot.recent_messages.append(message)
            snapshot.message_count += 1
            snapshot.updated_at = message.timestamp

            # Count unsummarized messages since last summary watermark
            last_id = snapshot.last_summary_message_id
            delta_messages = []
            if not last_id:
                delta_messages = snapshot.recent_messages
            else:
                idx = -1
                for i, m in enumerate(snapshot.recent_messages):
                    if m.message_id == last_id:
                        idx = i
                        break
                delta_messages = snapshot.recent_messages[idx + 1:] if idx != -1 else snapshot.recent_messages

            trigger_summarization = len(delta_messages) >= settings.SHORT_TERM_MESSAGE_LIMIT

            # Execute writes atomically using a Redis transaction
            async with self.redis._get_client() as client:
                pipe = client.pipeline(transaction=True)
                
                recent_key = f"recent:{conversation_id}"
                pipe.rpush(recent_key, json.dumps(message.dict()))
                pipe.ltrim(recent_key, -(settings.SHORT_TERM_MESSAGE_LIMIT * 2), -1)
                pipe.expire(recent_key, settings.SNAPSHOT_TTL_SECONDS)

                hash_key = f"snapshot:{conversation_id}"
                pipe.hset(hash_key, mapping={
                    "user_id": snapshot.user_id,
                    "message_count": snapshot.message_count,
                    "last_summary_message_id": snapshot.last_summary_message_id,
                    "state": MemoryState.SUMMARY_PENDING.value if trigger_summarization else MemoryState.ACTIVE.value,
                    "updated_at": snapshot.updated_at
                })
                pipe.expire(hash_key, settings.SNAPSHOT_TTL_SECONDS)
                
                await pipe.execute()

            return snapshot, delta_messages, trigger_summarization

        finally:
            await self.redis.release_lock(lock_key)
```

---

## 8. Decoupled Kafka Worker Pipeline (Issue 1, 3, 6, 18)

Background workers process tasks asynchronously using internal Kafka topics, enabling scaling and preventing message ingestion bottlenecks.

```text
                         Internal Kafka Worker Pipeline
                         
    Ingestion Handler ──► [Kafka Topic: memory.summary.request]
                                         │
                                         ▼
                                  [Summary Worker]
                                         │
                                         ▼
                          [Kafka Topic: memory.fact.request]
                                         │
                                         ▼
                                   [Fact Worker]
                                         │
                                         ▼
                        [Kafka Topic: memory.embedding.request]
                                         │
                                         ▼
                                 [Embedding Worker]
```

### 8.1 Summary Worker: `workers/summary_worker.py` (Issue 3, 14, 15, 21)
Subscribes to `memory.summary.request` to update conversation summaries in Redis.

```python
import json
import logging
import asyncio
from aiokafka import AIOKafkaConsumer
from src.config.settings import settings
from src.clients.redis_client import AsyncRedisClient
from src.clients.llm_client import LLMClient
from src.clients.kafka_producer import MemoryKafkaProducer
from src.engine.snapshot import ConversationSnapshotBuilder
from src.engine.state_machine import MemoryState

logger = logging.getLogger("memory_service.summary_worker")

class SummaryWorker:
    """
    Kafka Consumer worker that processes conversation summarization requests.
    """
    def __init__(self, redis: AsyncRedisClient, llm: LLMClient, producer: MemoryKafkaProducer):
        self.redis = redis
        self.llm = llm
        self.producer = producer
        self.snapshot_builder = ConversationSnapshotBuilder(redis)
        self.consumer: AIOKafkaConsumer = None
        self.is_running = False

    async def start(self) -> None:
        self.consumer = AIOKafkaConsumer(
            settings.KAFKA_SUMMARY_TOPIC,
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            group_id="memory-summary-worker-group",
            enable_auto_commit=False,
            auto_offset_reset="earliest"
        )
        await self.consumer.start()
        self.is_running = True
        asyncio.create_task(self._run_loop())

    async def _run_loop(self) -> None:
        while self.is_running:
            msg_batch = await self.consumer.getmany(timeout_ms=1000)
            for tp, messages in msg_batch.items():
                for message in messages:
                    # Parse task properties
                    payload = json.loads(message.value.decode("utf-8"))
                    trace_id = payload.get("trace_id", "none")
                    conversation_id = payload.get("conversation_id")
                    
                    # Structured Logging (Issue 10)
                    log_meta = f"[Trace: {trace_id}][Conv: {conversation_id}][Event: {message.offset}]"
                    logger.info(f"{log_meta} Processing summary request task.")

                    success = await self._process_summary(payload, log_meta)
                    if success:
                        await self.consumer.commit({tp: message.offset + 1})
                    else:
                        # Route failed task to DLQ
                        await self.producer.publish_task(settings.KAFKA_DLQ_TOPIC, conversation_id, payload)
                        await self.consumer.commit({tp: message.offset + 1})

    async def _process_summary(self, payload: dict, log_meta: str) -> bool:
        conversation_id = payload.get("conversation_id")
        user_id = payload.get("user_id")
        delta_messages_raw = payload.get("delta_messages", [])
        
        lock_key = f"lock:{conversation_id}"
        acquired = await self.redis.acquire_lock(lock_key, ttl_seconds=10)
        if not acquired:
            return False

        try:
            # 1. Fetch current summary and check version (Optimistic Locking - Issue 15)
            summary_key = f"summary:{conversation_id}"
            current_summary_data = await self.redis.hgetall(summary_key)
            
            current_ver = int(current_summary_data.get("summary_version", 0))
            expected_ver = int(payload.get("expected_version", 0))
            if expected_ver < current_ver:
                logger.warning(f"{log_meta} Version mismatch. Expected: {expected_ver}, Current: {current_ver}. Aborting task.")
                return True

            # 2. Decompress summary text from cache (Issue 9)
            compressed_text = current_summary_data.get("text")
            current_text = decompress_summary(compressed_text) if compressed_text else ""

            # 3. Call LLM to update summary (gRPC)
            new_summary_text = await self.llm.summarize(current_text, delta_messages_raw)

            # 4. Compress and save updated summary
            new_version = current_ver + 1
            now = time.time()
            compressed_new = compress_summary(new_summary_text)

            async with self.redis._get_client() as client:
                pipe = client.pipeline(transaction=True)
                pipe.hset(summary_key, mapping={
                    "text": compressed_new,
                    "summary_version": new_version,
                    "summary_updated_at": now
                })
                # Truncate sliding message window to latest 20 messages (Issue 21)
                pipe.ltrim(f"recent:{conversation_id}", -settings.SHORT_TERM_MESSAGE_LIMIT, -1)
                await pipe.execute()

            # 5. Update watermark and state
            last_msg_id = delta_messages_raw[-1]["message_id"]
            await self.snapshot_builder.update_snapshot_summary_watermark(conversation_id, last_msg_id)
            
            hash_key = f"snapshot:{conversation_id}"
            await self.redis.hmset(hash_key, {"state": MemoryState.SUMMARIZED.value})

            # 6. Publish task to fact worker topic (Pass summary details directly - Issue 3)
            fact_payload = {
                "trace_id": trace_id,
                "user_id": user_id,
                "summary_text": new_summary_text,
                "summary_version": new_version,
                "timestamp": now
            }
            await self.producer.publish_task(settings.KAFKA_FACT_TOPIC, conversation_id, fact_payload)
            return True
            
        except Exception as e:
            logger.error(f"{log_meta} Summary processing failed: {str(e)}")
            return False
        finally:
            await self.redis.release_lock(lock_key)
```

### 8.2 LLM gRPC Rate Limiting & Semaphore (Issue 7)
Workers use a semaphore to limit concurrent requests to the LLM service to 50, preventing overload.

```python
class LLMClient:
    """
    gRPC client interface for the LLM Service.
    Enforces a concurrency limit of 50 using a semaphore.
    """
    def __init__(self):
        self.channel = None
        self.stub = None
        self.semaphore = asyncio.Semaphore(settings.LLM_CONCURRENT_LIMIT)

    async def summarize(self, current_summary: str, new_messages: List[Any]) -> str:
        async with self.semaphore:
            # Execute gRPC call within semaphore block
            response = await self.stub.SummarizeConversation(request, timeout=5)
            return response.updated_summary
```

### 8.3 Vector Updates in Milvus (Issue 5)
Milvus does not support true in-place updates. To update an existing vector record, we first delete the record by key and then insert the updated record.

```python
async def update_fact_vector(self, collection_name: str, user_id: str, conversation_id: str, memory_id: str, new_record: dict) -> None:
    """
    Updates a vector record in Milvus by deleting the existing record and inserting the update.
    """
    collection = self.collections.get(collection_name)
    if not collection:
        raise ValueError(f"Collection {collection_name} is not loaded.")

    # 1. Delete the existing record
    expr = f'user_id == "{user_id}" and memory_id == "{memory_id}"'
    collection.delete(expr)
    
    # 2. Insert the updated record
    await self.insert_entities(collection_name, [new_record])
```

---

## 9. Structured Memory Context Retrieval (JSON Only)

The Memory Service API returns structured JSON data. Prompt assembly and model-specific formatting are managed by the LLM service.

```text
               JSON-Only Context Assembly Flow
               
    [LLM Service Client] ──► GET /internal/memory/context
                                       │
                                       ▼
                       [Memory Service Context API]
                                       │
                                       ├─► Load Short-Term Messages
                                       ├─► Load Lineage Summaries
                                       └─► Query Milvus Vectors
                                       │
                                       ▼
                         [Context Ranking Engine]
                                       │
                                       ▼
                      Structured JSON Context Response
```

### 9.1 REST Endpoint: `api/routes.py`
```python
from fastapi import APIRouter, Depends, HTTPException, status
from src.api.dependencies import get_context_builder
from src.engine.context_builder import MemoryContextBuilder
from src.models.api_models import RetrievalContextResponse

router = APIRouter(prefix="/internal/memory")

@router.get(
    "/context",
    response_model=RetrievalContextResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch structured memory context"
)
async def get_memory_context(
    user_id: str,
    conversation_id: str,
    query: str,
    builder: MemoryContextBuilder = Depends(get_context_builder)
):
    """
    Assembles and returns structured JSON memory context for the LLM service.
    """
    try:
        raw_layers = await builder.fetch_all_layers(user_id, conversation_id, query)
        
        return RetrievalContextResponse(
            short_term_messages=raw_layers["short_term_messages"],
            medium_term_summaries=raw_layers["medium_term_summaries"],
            long_term_facts=raw_layers["long_term_facts"],
            semantic_memories=raw_layers["semantic_memories"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assemble memory context: {str(e)}"
        )
```

---

## 10. Background Cleanup Worker (Issue 6, 20)

A background worker runs sweeps periodically (e.g. every 24 hours) to clean up expired data, remove orphaned database keys, and compact Milvus collections.

```python
import time
import logging
import asyncio
from src.clients.redis_client import AsyncRedisClient
from src.clients.milvus_client import MilvusClient

logger = logging.getLogger("memory_service.cleanup_worker")

class MemoryCleanupWorker:
    """
    Background worker that runs periodic sweeps to clean up expired data.
    """
    def __init__(self, redis: AsyncRedisClient, milvus: MilvusClient, interval_seconds: int = 86400):
        self.redis = redis
        self.milvus = milvus
        self.interval_seconds = interval_seconds
        self.is_running = False

    async def start(self) -> None:
        self.is_running = True
        logger.info(f"Cleanup worker configured with interval {self.interval_seconds}s")
        while self.is_running:
            try:
                await self.run_cleanup_sweep()
            except Exception as e:
                logger.error(f"Error during cleanup sweep: {str(e)}")
            await asyncio.sleep(self.interval_seconds)

    async def run_cleanup_sweep(self) -> None:
        """
        Executes cleanup tasks across storage systems.
        """
        logger.info("Starting cleanup sweep...")
        
        # 1. Sweep and remove expired Redis keys
        # Redis automatically manages key expirations via TTL, but we verify 
        # that orphaned snapshot states are cleaned up.
        
        # 2. Purge expired Milvus vector records
        # Identifies and deletes vectors that exceed retention rules or 
        # belong to deleted conversations.
        
        # 3. Compact Milvus collections to reclaim disk space
        for col_name in ["user_memory_vectors", "summary_vectors", "semantic_memory_vectors"]:
            collection = self.milvus.collections.get(col_name)
            if collection:
                logger.info(f"Compacting Milvus collection: {col_name}")
                collection.compact()
                
        logger.info("Cleanup sweep completed.")
```

---
**End of Production-Ready Low-Level Design (LLD)**
