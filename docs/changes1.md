Since you're moving **summary generation inside Memory Service**, the HLD only needs to change in the **Summary Generation pipeline**. Everything else (Kafka, Cassandra, Redis, Milvus, Embedding Service) remains the same.

---

# HLD Change: Internal LLM Summarization Engine

## Overview

The Memory Service now owns **all memory summarization** responsibilities.

Instead of delegating summarization to an external LLM Service, it directly invokes the configured LLM provider (currently **Groq**) through an internal LLM module.

This reduces network hops, simplifies the architecture, and keeps all memory-related intelligence inside a single bounded context.

The LLM implementation is abstracted behind a provider interface, allowing future migration to OpenAI, Gemini, Claude, Qwen, or local models without changing business logic.

---

# Three Types of Summaries

The Memory Service generates **three different summaries**, each serving a different purpose.

## 1. Short-Term Conversation Summary

**Purpose**

Maintains the latest state of the active conversation.

**Input**

* Previous conversation summary
* Latest message window (20 messages)

**Output**

Updated conversation summary.

Stored in:

* Cassandra
* Redis

Used for:

* Current conversation context

---

## 2. Medium-Term Lineage Summary

**Purpose**

Represents summaries of parent conversations in the graph.

The Memory Service retrieves conversation lineage from the Graph Service and loads the stored summaries.

No new summarization occurs unless a parent conversation itself changes.

Stored in:

* Cassandra
* Redis

Used for:

* Cross-conversation context
* Branched conversations

---

## 3. Long-Term Memory Summary (Fact Extraction)

**Purpose**

Extract durable user knowledge from conversations.

The LLM analyzes the latest conversation summary and extracts:

* Preferences
* Habits
* Decisions
* Projects
* Goals

These become structured facts.

Stored in:

* Cassandra

Embedded into:

* Milvus

Used for:

* Long-term personalization

---

# Updated High-Level Architecture

```text
                Kafka

chat.message.created
chat.response.completed

          │
          ▼

 Memory Event Consumer

          │

 Snapshot Builder

          │

 Cassandra Logged Batch

          │

 Publish

memory.summary.request

          ▼

 Summary Worker

          │

          ▼

 Internal LLM Service

          │

          ▼

 Groq API

          │

 Updated Summary

          │

 Save Summary

          │

 Publish

memory.fact.request

          ▼

 Fact Worker

          │

          ▼

 Internal LLM Service

          │

          ▼

 Groq API

          │

 Extract Facts

          │

 Save Facts

          │

 Publish

memory.embedding.request

          ▼

 Embedding Worker

          │

          ▼

 Embedding Service

          │

          ▼

 Milvus
```

---

# Internal LLM Module

The Memory Service contains an internal LLM module responsible for:

* Summary generation
* Fact extraction
* Prompt construction
* Retry handling
* Timeout management
* Provider abstraction

It does **not** store memory.

It only performs inference.

---

# Supported Interfaces

The internal LLM module exposes:

### Internal Service Call

Used by

* Summary Worker
* Fact Worker

---

### HTTP API

Used for

* Manual testing
* Swagger
* Development

Example

```http
POST /api/v1/llm/summarize
```

---

### gRPC API

Used for

Future internal services.

Example

```protobuf
rpc Summarize(...)
```

---

# Provider Abstraction

Current Provider

```text
Groq
```

Future providers

* OpenAI
* Gemini
* Claude
* Qwen
* Local Llama

Provider selection is configuration driven.

Business logic never depends on a specific vendor.

---

# Responsibilities

## Summary Worker

* Receive Kafka job
* Build summary request
* Call Internal LLM
* Persist summary
* Publish Fact Job

---

## Fact Worker

* Receive Kafka job
* Build fact extraction request
* Call Internal LLM
* Persist facts
* Publish Embedding Job

---

## Internal LLM Service

* Validate request
* Build prompts
* Invoke provider
* Handle retries
* Handle timeouts
* Return structured response

---

## Groq Provider

Responsible only for:

* API communication
* Authentication
* Completion requests

No business logic.

---

# Benefits

* Memory intelligence remains inside one bounded context.
* One less microservice to maintain.
* Lower latency (no extra service hop).
* Provider-independent architecture.
* Easy future migration to different LLM vendors.
* HTTP endpoint for testing and gRPC endpoint for future service integration.
* Existing Kafka pipeline remains unchanged.

This change fits naturally into your current Memory Service design while preserving the event-driven pipeline for snapshots, summaries, facts, and embeddings.



# Low Level Design (LLD)

## Internal LLM Engine (Memory Service)

---

# Chapter 1. Overview

The Memory Service now owns the complete lifecycle of memory intelligence.

Instead of invoking an external LLM Service, it performs inference internally through an embedded LLM module while keeping the rest of the event-driven pipeline unchanged.

The LLM Engine is responsible for:

* Incremental Short-Term Summary generation
* Long-Term Fact extraction
* Prompt construction
* Provider abstraction
* Retry & timeout handling
* Response parsing

The implementation follows:

* Dependency Injection
* Singleton
* Factory Method
* Strategy Pattern (LLM Providers)
* Adapter Pattern (Groq SDK Wrapper)

---

# Chapter 2. Architecture

```text
                    Summary Worker
                           │
                           ▼
                  Summary Service
                           │
                           ▼
                     LLM Service
                           │
                    Prompt Builder
                           │
                           ▼
                     LLM Manager
                           │
                 LLM Provider Strategy
                           │
                    Groq Provider
                           │
                     Groq Client
                           │
                        Groq API
                           │
                    Generated Summary
                           │
                           ▼
                  Summary Service
                           │
                    Cassandra Store
                           │
                           ▼
                 memory.fact.request
```

Fact Extraction

```text
Fact Worker

      │

      ▼

LongMemoryService

      │

      ▼

LLM Service

      │

Prompt Builder

      │

      ▼

LLM Manager

      │

Groq Provider

      │

Groq API

      │

Facts

      │

Cassandra

      │

memory.embedding.request
```

---

# Chapter 3. Folder Additions

```text
app/

├── api/
│   └── internal/
│       └── llm.py
│
├── grpc/
│   ├── llm.proto
│   ├── handlers.py
│   ├── server.py
│   └── generated/
│
├── clients/
│   └── groq_client.py
│
├── factories/
│   └── llm_factory.py
│
├── managers/
│   └── llm_manager.py
│
├── providers/
│   ├── base.py
│   └── groq_provider.py
│
├── prompts/
│   ├── summary_prompt.py
│   ├── fact_prompt.py
│   └── system_prompt.py
│
├── schemas/
│   └── llm.py
│
└── services/
    └── llm_service.py
```

---

# Chapter 4. Component Responsibilities

## 4.1 Groq Client

Location

```text
clients/groq_client.py
```

Responsibilities

* Initialize SDK
* Authenticate
* Send Completion Request
* Receive Response

Never

* Builds prompts
* Parses business output

Pattern

Adapter

---

## 4.2 Provider Layer

Location

```text
providers/
```

Interface

```python
BaseLLMProvider
```

Methods

```python
summarize()

extract_facts()
```

Current

```text
GroqProvider
```

Future

* OpenAI
* Gemini
* Claude
* Qwen
* Local Llama

Pattern

Strategy Pattern

---

## 4.3 Factory

Location

```text
factories/llm_factory.py
```

Responsibilities

Read configuration

```text
LLM_PROVIDER=groq
```

Instantiate

```text
GroqProvider
```

Future

```text
OpenAIProvider

GeminiProvider

ClaudeProvider
```

No business logic changes required.

Pattern

Factory Method

---

## 4.4 LLM Manager

Location

```text
managers/llm_manager.py
```

Responsibilities

* Hold singleton provider
* Retry failed requests
* Timeout handling
* Circuit breaker
* Metrics
* Provider lifecycle

Never

* Build prompts

Pattern

Singleton

---

## 4.5 Prompt Builder

Location

```text
prompts/
```

Files

```text
summary_prompt.py

fact_prompt.py

system_prompt.py
```

Responsibilities

Generate prompt templates

Short Summary Prompt

```text
Previous Summary

+

Recent Messages

↓

Updated Summary
```

Fact Prompt

```text
Conversation Summary

↓

Extract Facts
```

---

## 4.6 LLM Service

Location

```text
services/llm_service.py
```

Responsibilities

* Validate request
* Select prompt
* Call LLM Manager
* Parse response
* Return DTO

Workers never call Groq directly.

---

# Chapter 5. HTTP API

## Summary

```http
POST /internal/llm/summarize
```

Request

```json
{
  "previous_summary":"...",
  "new_messages":[]
}
```

Response

```json
{
  "summary":"..."
}
```

---

## Fact Extraction

```http
POST /internal/llm/facts
```

Request

```json
{
    "summary":"..."
}
```

Response

```json
{
    "facts":[]
}
```

Purpose

Development

Swagger

Testing

---

# Chapter 6. gRPC API

```protobuf
service LLMService {

    rpc Summarize(
        SummaryRequest
    ) returns (
        SummaryResponse
    );

    rpc ExtractFacts(
        FactRequest
    ) returns (
        FactResponse
    );

}
```

Purpose

Future service reuse.

---

# Chapter 7. Worker Integration

## Summary Worker

Current

```text
Summary Worker

↓

External LLM
```

Updated

```text
Summary Worker

↓

Summary Service

↓

LLM Service

↓

Groq
```

---

## Fact Worker

Current

```text
Fact Worker

↓

External LLM
```

Updated

```text
Fact Worker

↓

LongMemoryService

↓

LLM Service

↓

Groq
```

No worker directly accesses Groq.

---

# Chapter 8. Dependency Injection

Container registers

```text
Groq Client

↓

Groq Provider

↓

LLM Manager

↓

LLM Service

↓

Summary Service

↓

LongMemoryService
```

Every component receives dependencies through constructor injection.

---

# Chapter 9. Configuration

```env
LLM_PROVIDER=groq

GROQ_API_KEY=

LLM_MODEL=llama-3.3-70b-versatile

LLM_TIMEOUT_SECONDS=60

LLM_MAX_RETRIES=3

LLM_TEMPERATURE=0.2

LLM_MAX_TOKENS=1024
```

---

# Chapter 10. Error Handling

Failures handled

* Invalid request
* Timeout
* Rate limit
* Provider unavailable
* Retry exhaustion
* Malformed response

Recovery

* Exponential backoff
* Circuit breaker
* Structured logging
* Error propagation to worker
* Worker retry through existing Kafka retry pipeline

---

# Chapter 11. Sequence Flow

### Short-Term Summary

```text
Summary Worker
      │
      ▼
Summary Service
      │
      ▼
LLM Service
      │
      ▼
Prompt Builder
      │
      ▼
LLM Manager
      │
      ▼
Groq Provider
      │
      ▼
Groq Client
      │
      ▼
Groq API
      │
      ▼
Summary
      │
      ▼
Summary Service
      │
      ▼
Cassandra
      │
      ▼
memory.fact.request
```

---

### Long-Term Fact Extraction

```text
Fact Worker
      │
      ▼
LongMemoryService
      │
      ▼
LLM Service
      │
      ▼
Prompt Builder
      │
      ▼
LLM Manager
      │
      ▼
Groq Provider
      │
      ▼
Groq Client
      │
      ▼
Groq API
      │
      ▼
Facts
      │
      ▼
Cassandra
      │
      ▼
memory.embedding.request
```

---

# Chapter 12. Design Patterns

| Component        | Pattern                    |
| ---------------- | -------------------------- |
| `container.py`   | Dependency Injection       |
| `llm_factory.py` | Factory Method             |
| `llm_manager.py` | Singleton                  |
| `providers/`     | Strategy Pattern           |
| `groq_client.py` | Adapter Pattern            |
| `llm_service.py` | Facade/Application Service |

This design keeps the Memory Service as the single owner of memory intelligence while remaining extensible to additional LLM providers without impacting the worker pipeline or business logic.
