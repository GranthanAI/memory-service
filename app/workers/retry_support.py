"""Shared retry -> FAILED (dead-letter) recording for consumers whose events cannot be dropped.

A failed event becomes a row in `retry_jobs`: PENDING with an exponential backoff (RetryScheduler republishes the ORIGINAL payload
to the job type's topic), or FAILED - the dead-letter record - once MAX_ATTEMPTS is reached or when the payload can never succeed
(undecodable / missing fields). Used by the ingestion worker, the branch-summary worker and the conversation-delete worker.
"""
import json
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

MAX_ATTEMPTS = 5

logger = logging.getLogger("memory_service.workers.retry_support")


def defer_job(cassandra_repo, job_type: str, payload: dict, wait_s: float, conversation_id: str = "unknown") -> str:
    """Groq is busy: run the job again after Groq's reset (+0-3 s jitter). Returns "deferred" or "dropped".

    A deferral is not a failure and uses no attempt: the scheduler re-dispatches with attempt_count = retry_count + 1, so the
    stored retry_count is one LESS than the attempt that just ran. The job keeps `deferred_since` in its payload and is dropped
    only after MEMORY_JOB_MAX_AGE_MIN of waiting. Uses the retry_jobs table (polled by RetryScheduler), nothing sleeps."""
    from app.clients.groq_budget import MEMORY_JOBS_DEFERRED, MEMORY_JOBS_DROPPED
    from app.core.config import settings

    now = datetime.now(timezone.utc)
    try:
        since = datetime.fromisoformat(str(payload.get("deferred_since")))
    except (TypeError, ValueError):
        since = now
    attempt = int(payload.get("attempt_count", 0) or 0)
    if now - since > timedelta(minutes=settings.MEMORY_JOB_MAX_AGE_MIN):
        MEMORY_JOBS_DROPPED.labels(type=job_type).inc()
        logger.warning("memory_job_dropped type=%s conversation_id=%s waited_min=%.0f (MEMORY_JOB_MAX_AGE_MIN=%s)",
                       job_type, conversation_id, (now - since).total_seconds() / 60, settings.MEMORY_JOB_MAX_AGE_MIN)
        return "dropped"
    delay = max(0.0, float(wait_s)) + random.uniform(0.0, 3.0)
    cassandra_repo.insert_retry_job({
        "status": "PENDING", "next_retry": now + timedelta(seconds=delay), "job_id": uuid.uuid4(), "job_type": job_type,
        "payload": json.dumps({**payload, "deferred_since": since.isoformat()}),
        "retry_count": max(attempt - 1, 0), "max_retry": MAX_ATTEMPTS, "last_error": "deferred: groq busy", "created_at": now,
    })
    MEMORY_JOBS_DEFERRED.labels(type=job_type).inc()
    logger.info("memory_job_deferred type=%s conversation_id=%s delay_s=%.1f attempt=%d", job_type, conversation_id, delay, attempt)
    return "deferred"


def schedule_retry_or_park(cassandra_repo, job_type: str, payload: Optional[dict], raw: bytes, exc: Exception,
                           required_key: str = "conversation_id") -> str:
    """Records the failure and returns "retry_scheduled" or "dead_lettered". Raises when the record itself cannot be written, so
    the caller can refuse to commit the offset (a lost event must never be silent)."""
    if cassandra_repo is None:
        raise RuntimeError("no retry store configured")
    now = datetime.now(timezone.utc)
    if getattr(exc, "wait_s", None) is not None and isinstance(payload, dict):
        body0 = payload.get("payload", payload)
        return defer_job(cassandra_repo, job_type, payload, float(exc.wait_s), str(body0.get(required_key, "unknown")) if isinstance(body0, dict) else "unknown")
    error = f"{type(exc).__name__}: {exc}"[:500]
    body = payload.get("payload", payload) if isinstance(payload, dict) else None       # envelopes wrap the event payload
    decodable = isinstance(payload, dict) and bool(isinstance(body, dict) and body.get(required_key))
    attempt = int(payload.get("attempt_count", 0)) if decodable else MAX_ATTEMPTS
    permanent = isinstance(exc, (KeyError, ValueError))                                 # cannot succeed on a second try
    if decodable and attempt + 1 < MAX_ATTEMPTS and not permanent:
        status, next_retry, outcome = "PENDING", now + timedelta(seconds=2 ** (attempt + 1)), "retry_scheduled"
        stored = json.dumps({**payload, "attempt_count": attempt + 1})
    else:
        status, next_retry, outcome = "FAILED", now, "dead_lettered"
        stored = json.dumps(payload) if decodable else raw.decode("utf-8", errors="replace")[:20000]
    cassandra_repo.insert_retry_job({
        "status": status,
        "next_retry": next_retry,
        "job_id": uuid.uuid4(),
        "job_type": job_type,
        "payload": stored,
        "retry_count": attempt + 1 if status == "PENDING" else attempt,
        "max_retry": MAX_ATTEMPTS,
        "last_error": error if status == "PENDING" else f"Dead-lettered: {error}",
        "created_at": now,
    })
    return outcome
