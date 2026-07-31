Below is the **updated HLD** based on **all the architectural decisions we've finalized**:

* ✅ Conversation → Memory = Kafka
* ✅ Memory → Graph = Internal HTTP/gRPC
* ✅ Memory → LLM = gRPC
* ✅ Vector DB = **Milvus**
* ✅ No Conversation API calls
* ✅ Event-driven snapshot builder
* ✅ Incremental summarization
* ✅ Graph lineage for medium memory

---

# GraphGPT Memory Service

## High Level Design (HLD)

**Version:** 2.0

**Target Scale**

* 100M Users
* 1M Writes/sec (Conversation Layer)
* 99.99% Availability

---

# Chapter 1. Overview

The Memory Service is the AI Memory Engine of GraphGPT.

Its responsibility is to continuously build intelligent memories from user conversations while keeping LLM context small, relevant, and scalable.

Instead of storing raw conversations, it stores:

* Conversation Snapshots
* Conversation Summaries
* User Memories
* Semantic Memories
* Memory Embeddings

The service continuously consumes Kafka events, updates memory, generates summaries through the LLM Service, stores embeddings in Milvus, and retrieves optimized context during inference.

The Memory Service is **not the source of truth** for conversation messages.

---

# Chapter 2. Responsibilities

The Memory Service is responsible for:

* Maintaining Short-Term Memory
* Maintaining Medium-Term Memory
* Maintaining Long-Term Memory
* Maintaining Semantic Memory
* Building Conversation Snapshots
* Incremental Summarization
* User Fact Extraction
* Embedding Generation
* Memory Retrieval
* Context Assembly
* Memory Expiration
* Memory Cleanup

It is NOT responsible for:

* Conversation persistence
* Graph traversal
* AI response generation
* File Retrieval
* Search

---

# Chapter 3. High Level Architecture

```text
                        Kafka
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
 chat.message.created            chat.response.completed
        │                                   │
        └─────────────────┬─────────────────┘
                          ▼
                  Memory Event Consumer
                          │
                          ▼
            Conversation Snapshot Builder
                          │
      ┌───────────────────┼─────────────────────┐
      │                   │                     │
      ▼                   ▼                     ▼
 Short Memory      Summary Manager      Long Memory Manager
      │                   │                     │
      │                   ▼                     ▼
      │            gRPC → LLM Service   gRPC → LLM Service
      │                   │                     │
      └──────────────┬────┴──────────────┬──────┘
                     ▼                   ▼
               Embedding Manager
                     │
          ┌──────────┴───────────┐
          │                      │
        Redis                 Milvus
          │                      │
          └──────────┬───────────┘
                     ▼
             Memory Retrieval API
```

---

# Chapter 4. Memory Layers

---

## 4.1 Short-Term Memory

### Scope

Current Conversation

### Storage

Redis

### Stores

* Latest 20 Messages
* Active Conversation State
* Temporary Context
* Current Entities

### Purpose

Maintains conversational continuity.

---

## 4.2 Medium-Term Memory

### Scope

Conversation Lineage

Uses Graph Service.

```
Current Conversation

↓

Parent

↓

Parent

↓

Root
```

Memory requests

```
GET /internal/graph/conversations/{id}/memory-context
```

Graph returns

```
Root

↓

Ancestors

↓

Current
```

Memory loads summaries for all conversations.

Storage

Redis

---

## 4.3 Long-Term Memory

### Scope

Entire User

Stores

* Preferences
* Facts
* Technologies
* Decisions
* Projects
* Habits

Storage

Milvus

---

## 4.4 Semantic Memory

Stores vectorized memories enabling semantic retrieval.

Storage

Milvus

---

# Chapter 5. Event Processing

Memory Service consumes Kafka topics.

```
conversation.created

conversation.updated

conversation.deleted

chat.message.created

chat.response.completed
```

Conversation Service is never called synchronously.

---

# Chapter 6. Conversation Snapshot Builder

This is the heart of the Memory Service.

Every conversation maintains:

```
Conversation Snapshot

conversation_id

summary

recent_messages

message_count

last_summary_message

updated_at
```

Snapshots are event-driven.

Every Kafka message updates the snapshot.

---

# Chapter 7. Incremental Summarization

Initially

```
Messages

1

↓

20
```

Memory sends

```
Summary = Empty

+

Messages 1-20
```

via gRPC to LLM Service.

LLM returns

```
Summary V1
```

Memory stores

```
Summary V1

+

Latest 20 Messages
```

---

Next cycle

```
Summary V1

+

Messages 21-40
```

↓

LLM

↓

Summary V2

No old messages are reprocessed.

---

# Chapter 8. Long-Term Memory Pipeline

Whenever summary changes

Memory calls

```
Extract Facts
```

Example

```
User likes Python

Uses Neo4j

Building GraphGPT

Prefers FastAPI
```

Memory then calls

```
Generate Embeddings
```

Vectors are stored in Milvus.

---

# Chapter 9. Context Retrieval Pipeline

Whenever LLM receives a user prompt

```
LLM Service

↓

Memory Service

↓

Load Short Memory

↓

Call Graph Service

↓

Load Medium Memory

↓

Search Long Memory

↓

Semantic Search

↓

Rank Memories

↓

Assemble Context

↓

Return Context
```

---

# Chapter 10. External Dependencies

## Conversation Service

Communication

Kafka

Purpose

Receive conversation events.

---

## Graph Service

Communication

Internal HTTP/gRPC

Purpose

Conversation lineage.

---

## LLM Service

Communication

gRPC

Purpose

* Summaries
* Fact Extraction
* Embeddings

---

## Redis

Stores

* Short Memory
* Conversation Snapshots
* Active Summaries

---

## Milvus

Stores

* Long-Term Memory
* Semantic Memory
* Summary Embeddings

---

# Chapter 11. Internal APIs

```
GET /internal/memory/context
```

Returns

* Short Memory
* Medium Memory
* Long Memory
* Semantic Memory

---

```
DELETE /internal/memory/conversations/{id}
```

Deletes memory.

---

```
GET /health
```

---

# Chapter 12. Redis

Redis stores

```
conversation:{id}

summary:{id}

recent:{id}

lock:{id}
```

Redis TTL

* Conversation snapshots
* Active summaries
* Locks

---

# Chapter 13. Milvus Collections

## user_memory_vectors

Stores durable user memories.

---

## summary_vectors

Stores embeddings for conversation summaries.

---

## semantic_memory_vectors

Stores semantic memories.

---

Each record stores

```
memory_id

user_id

conversation_id

embedding

memory_type

importance

created_at

updated_at

metadata
```

---

# Chapter 14. Memory Retrieval

Memory Retrieval performs

```
Load Short Memory

↓

Load Medium Memory

↓

Search Milvus

↓

Rank Results

↓

Assemble Final Context
```

Final response

```
Current Messages

Conversation Summary

Parent Summaries

Relevant User Memories

Relevant Semantic Memories
```

---

# Chapter 15. Scaling

* Stateless Pods
* Kafka Consumer Groups
* Redis Cluster
* Milvus Cluster
* Horizontal Scaling
* At-Least-Once Processing
* Idempotent Consumers

---

# Chapter 16. Failure Recovery

If Memory crashes

```
Kafka Replay

↓

Rebuild Snapshot

↓

Continue
```

If LLM unavailable

```
Retry

↓

Circuit Breaker

↓

DLQ
```

If Milvus unavailable

```
Retry

↓

Queue

↓

Replay
```

---

# Chapter 17. Sequence Diagram

## Memory Update

```
Conversation Service

↓

Kafka

↓

Memory Consumer

↓

Conversation Snapshot

↓

Threshold Reached?

↓

No
    ↓
Update Snapshot

Yes
    ↓
gRPC → LLM

↓

Summary

↓

Extract Facts

↓

Generate Embeddings

↓

Redis

↓

Milvus
```

---

## Context Retrieval

```
LLM Service

↓

Memory Service

↓

Redis

↓

Graph Service

↓

Milvus

↓

Memory Ranking

↓

Return Context
```

---

# Chapter 18. Technology Stack

| Component           | Technology                           |
| ------------------- | ------------------------------------ |
| Framework           | FastAPI                              |
| Broker              | Kafka                                |
| Cache               | Redis                                |
| Vector Database     | Milvus                               |
| AI Communication    | gRPC                                 |
| Graph Communication | Internal HTTP/gRPC                   |
| Observability       | OpenTelemetry + Prometheus + Grafana |
| Deployment          | Docker + Kubernetes                  |

---

# Chapter 19. Service Communication

```
                    Conversation Service
                            │
                         Kafka
                            │
                            ▼
                     Memory Service
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
     Redis       Graph Service    LLM Service
        │        (Lineage API)       │
        │                            │
        └──────────────┬─────────────┘
                       ▼
                    Milvus
```

---

# Chapter 20. Service Ownership

| Responsibility         | Owner                |
| ---------------------- | -------------------- |
| Raw Messages           | Conversation Service |
| Conversation Metadata  | Conversation Service |
| Conversation Lineage   | Graph Service        |
| Short Memory           | Memory Service       |
| Medium Memory          | Memory Service       |
| Long Memory            | Memory Service       |
| Semantic Memory        | Memory Service       |
| Conversation Snapshot  | Memory Service       |
| Summarization          | LLM Service          |
| Fact Extraction        | LLM Service          |
| Embeddings             | LLM Service          |
| AI Response Generation | LLM Service          |

---

# Final Memory Lifecycle

```
User Message
      │
      ▼
Conversation Service
      │
      ▼
Kafka
      │
      ▼
Memory Service
      │
      ├── Update Snapshot
      ├── Update Last 20 Messages
      ├── Threshold Check
      │
      ├── gRPC → LLM (Summarize)
      ├── gRPC → LLM (Extract Facts)
      ├── gRPC → LLM (Embeddings)
      │
      ├── Store Short Memory (Redis)
      ├── Store Summaries (Redis)
      └── Store Long/Semantic Memory (Milvus)
```

## One architectural refinement

I'd make one improvement over the previous version:

Instead of storing **summary text in both Redis and Milvus**, make **Redis the hot cache** (latest snapshots, summaries, recent messages) and **Milvus store only embeddings plus metadata**. If you need the actual summary text during retrieval, keep it in a lightweight metadata store (for example PostgreSQL owned by Memory Service) or as metadata alongside the Milvus vector if your expected summary sizes are modest. This keeps Milvus focused on vector search while Redis serves low-latency active memory.
