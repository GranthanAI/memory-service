"""Helpers for the two kinds of message id memory-service meets.

* UUIDv7 - minted by conversation-service; the first 48 bits are a unix-millisecond timestamp (time-ordered across
  milliseconds, random inside one millisecond).
* uuid5  - synthesized by the legacy ingestion path for events that carried no message ids; carries no time at all.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional


def _as_uuid(value) -> Optional[uuid.UUID]:
    try:
        return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


def is_uuid7(value) -> bool:
    parsed = _as_uuid(value)
    return parsed is not None and parsed.version == 7


def uuid7_unix_ms(value) -> Optional[int]:
    """Unix-millisecond timestamp encoded in a UUIDv7, or None for anything else."""
    parsed = _as_uuid(value)
    if parsed is None or parsed.version != 7:
        return None
    return int(parsed.hex[:12], 16)


def uuid7_datetime(value) -> Optional[datetime]:
    ms = uuid7_unix_ms(value)
    return None if ms is None else datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
