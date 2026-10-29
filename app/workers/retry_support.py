"""Shared retry -> FAILED (dead-letter) recording for consumers whose events cannot be dropped.

A failed event becomes a row in `retry_jobs`: PENDING with an exponential backoff (RetryScheduler republishes the ORIGINAL payload
to the job type's topic), or FAILED - the dead-letter record - once MAX_ATTEMPTS is reached or when the payload can never succeed
(undecodable / missing fields). Used by the ingestion worker, the branch-summary worker and the conversation-delete worker.
"""
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

MAX_ATTEMPTS = 5


def schedule_retry_or_park(cassandra_repo, job_type: str, payload: Optional[dict], raw: bytes, exc: Exception,
                           required_key: str = "conversation_id") -> str:
    """Records the failure and returns "retry_scheduled" or "dead_lettered". Raises when the record itself cannot be written, so
    the caller can refuse to commit the offset (a lost event must never be silent)."""
    if cassandra_repo is None:
        raise RuntimeError("no retry store configured")
    now = datetime.now(timezone.utc)
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
