"""Cost metering of every paid LLM call memory-service makes (billing integration plan 4.4.5, 4.9).

How it works
  * A worker wraps ONE job (one summary, one fact extraction, one branch summary, one compaction) in `metered_job(...)`.
  * Every call that reaches the provider (GroqClient.chat_completion, and a timed-out attempt in LLMManager) calls
    `record_call(...)`, which appends one entry to the job's collector (a ContextVar, so call sites deep in the services
    need no plumbing).
  * When the job ends (success OR failure, in a `finally`), ONE `usage.metered` 1.0 event is written through the existing
    Cassandra outbox (topic usage.metered, key = user_id). Nothing is written when the job made no paid call.
  * event_id = uuid5(NAMESPACE, job_key): the same job (a redelivered Kafka message) always produces the same id, so billing,
    which dedupes usage on event_id, can never count a job twice.

Provider-reported tokens are used as they are (usage.prompt_tokens, usage.completion_tokens, cached tokens when reported).
When a provider reports none, they are estimated with the repo's TokenCounter, `usage_estimated` is true and
memory_usage_estimated_total is incremented. The mock provider never calls `record_call`, so it is never metered.
A paid call made outside any job (an internal API / gRPC caller with no job) is counted in memory_unmetered_calls_total and
logged: it must be given a job context.
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional

from app.core.config import settings
from app.core.metrics import MEMORY_SYSTEM_CALLS, MEMORY_UNMETERED_CALLS, MEMORY_USAGE_ESTIMATED, MEMORY_USAGE_EVENTS

logger = logging.getLogger("memory_service.services.usage_metering")

NAMESPACE = uuid.UUID("6a8b2c1e-3f47-5d0a-9b21-7c4e5f6a8d90")      # fixed: event ids must be the same across restarts
USAGE_TOPIC = "usage.metered"
FEATURES = ("memory.summary", "memory.facts", "memory.branch_summary", "memory.compaction", "memory.update")


@dataclass
class JobUsage:
    feature: str
    job_key: str
    user_id: str
    conversation_id: Optional[str] = None
    request_message_id: Optional[str] = None
    calls: List[dict] = field(default_factory=list)


# The request_message_id of the memory.update.requested event being ingested; the summary / fact jobs it triggers carry it.
REQUEST_MESSAGE_ID: ContextVar[Optional[str]] = ContextVar("memory_request_message_id", default=None)
_current: ContextVar[Optional[JobUsage]] = ContextVar("memory_usage_job", default=None)
_sink: Optional[Any] = None                       # the Cassandra repository, set once by the container
_counter = None


def configure(cassandra_repo) -> None:
    global _sink
    _sink = cassandra_repo


def event_id_for(job_key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, job_key))


def _estimate(messages: Optional[list], text: Optional[str]) -> tuple[int, int]:
    global _counter
    if _counter is None:
        from app.utils.token_counter import TokenCounter
        _counter = TokenCounter()
    prompt = "\n".join(str(m.get("content") or "") for m in (messages or []))
    return _counter.count(prompt) if prompt else 0, _counter.count(text) if text else 0


def record_call(*, provider: str, model: str, usage: Any = None, messages: Optional[list] = None,
                output_text: Optional[str] = None, timed_out: bool = False, system: bool = False) -> None:
    """One call that reached the provider. `usage` is the provider's usage object (or None): its prompt/completion token
    counts are used as reported, otherwise they are estimated. A timed-out attempt reports an estimated input and no output."""
    if system:                                    # the service's own health probe: a few tokens, no user to bill
        MEMORY_SYSTEM_CALLS.inc()
        return
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None
    estimated = False
    if prompt is None or completion is None:
        est_in, est_out = _estimate(messages, output_text)
        prompt = est_in if prompt is None else prompt
        completion = est_out if completion is None else completion
        if timed_out:
            completion = 0
        estimated = True
        MEMORY_USAGE_ESTIMATED.labels(provider=provider, model=model).inc()
    job = _current.get()
    entry = {"provider": provider, "model": model, "purpose": job.feature if job else "memory.unattributed",
             "input_tokens": int(prompt), "cached_input_tokens": int(cached or 0), "output_tokens": int(completion),
             "usage_estimated": estimated}
    if job is None:
        MEMORY_UNMETERED_CALLS.inc()
        logger.error("memory_unmetered_llm_call: a paid call ran outside a metered job (it needs a job context)",
                     extra={"model": model})
        return
    job.calls.append(entry)


def build_event(job: JobUsage) -> dict:
    return {
        "event_type": USAGE_TOPIC,
        "schema_version": "1.0",
        "event_id": event_id_for(job.job_key),
        "occurred_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "producer": "memory-service",
        "user_id": job.user_id,
        "conversation_id": job.conversation_id,
        "request_message_id": job.request_message_id,
        "feature": job.feature,
        "calls": job.calls,
        "fixed_cost_micro_inr": 0,
    }


def _enqueue(job: JobUsage) -> None:
    if _sink is None:
        logger.error("usage_metering_not_configured: usage of job %s was dropped", job.job_key)
        return
    event = build_event(job)
    _sink.insert_outbox_job({
        "status": "PENDING",
        "created_at": datetime.now(timezone.utc),
        "job_id": uuid.UUID(event["event_id"]),          # the row id is the event id too
        "topic": USAGE_TOPIC,
        "conversation_id": job.user_id,                  # the outbox's partition-key field: billing wants key = user_id
        "payload": json.dumps(event),
        "attempt_count": 0,
    })
    MEMORY_USAGE_EVENTS.labels(feature=job.feature).inc()
    logger.info("usage_metered_enqueued", extra={"feature": job.feature, "calls": len(job.calls), "event_id": event["event_id"]})


@asynccontextmanager
async def metered_job(feature: str, job_key: str, user_id: str, conversation_id: Optional[str] = None,
                      request_message_id: Optional[str] = None):
    """Wrap ONE job. The usage event is written in `finally`: the cost of a job that failed after calling the model is still
    reported. Metering must never break the job: an enqueue failure is logged."""
    job = JobUsage(feature, job_key, user_id or "unknown_user", conversation_id, request_message_id)
    token = _current.set(job)
    try:
        yield job
    finally:
        _current.reset(token)
        if job.calls and job.user_id != "unknown_user":
            try:
                _enqueue(job)
            except Exception as e:  # noqa: BLE001
                logger.error(f"usage_metered_enqueue_failed for job {job_key}: {e}")
        elif job.calls:
            MEMORY_UNMETERED_CALLS.inc(len(job.calls))
            logger.error(f"usage of job {job_key} has no user_id and cannot be billed ({len(job.calls)} call(s))")


async def check_model_in_billing_catalog() -> Optional[bool]:
    """Startup warning when LLM_MODEL is not a model billing knows (its calls would go to billing's DLQ as 'unknown model').
    Never blocks startup: billing may be down. Returns True / False / None (not checked)."""
    if settings.LLM_PROVIDER == "mock":
        return None
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            data = (await client.get(f"{settings.BILLING_SERVICE_URL.rstrip('/')}/billing/v1/models")).json()
        raw = data.get("models", data) if isinstance(data, dict) else data
        ids = set()
        for entry in raw if isinstance(raw, list) else [m for group in (raw or {}).values() for m in group]:
            ids.add(entry.get("id") if isinstance(entry, dict) else str(entry))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"billing model catalog could not be read ({type(e).__name__}); LLM_MODEL not checked")
        return None
    if settings.LLM_MODEL not in ids:
        logger.warning(f"llm_model_not_in_billing_catalog: LLM_MODEL={settings.LLM_MODEL!r} is not a model billing can price; "
                       "usage.metered events for it will be rejected by billing")
        return False
    return True
