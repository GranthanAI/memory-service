"""Is a summary / a message provably at or before a branch point?

Kept in its own module (no service imports) so both the summary job (bounding a summary at a branch) and the context builder
(choosing which ancestor summary version a child may see) use the same rule.
"""
from typing import Any, Optional

from app.utils.message_ids import is_uuid7, uuid7_unix_ms


def _as_utc(value: Any):
    from app.services.summary_service import _as_utc as as_utc          # lazy: summary_service imports this module
    return as_utc(value)


def summary_valid_at_branch(
    through_message_id: Optional[str],
    upto_message_id: Optional[str],
    through_created_at: Any = None,
) -> bool:
    """True when a summary incorporates no message after the branch point (`upto_message_id`, inclusive).

    Two kinds of watermark exist:
      * UUIDv7 (events that carried the real conversation-service ids): both ids are time-ordered, so the
        millisecond timestamps are compared. Inside one millisecond the order of two v7 ids is random, so equality
        of the whole id is the only thing that proves the summary ends at the branch point.
      * uuid5 (legacy events): the id carries no order at all, so it must not be compared with a v7 id. Instead the
        watermark message's `through_created_at` is compared with the time encoded in `upto_message_id`. That
        timestamp is when the turn was PUBLISHED, i.e. at or after the messages were created, so it is accepted only
        when strictly earlier than the branch message (a summary through the very latest legacy turn is therefore
        rejected: not provable).
    Anything unknown or unparsable is NOT valid: a summary whose coverage cannot be proven to end at the branch
    point must not be shown to a child, because it could contain what the parent said afterwards."""
    upto_ms = uuid7_unix_ms(upto_message_id)
    if upto_ms is None:
        return False
    if is_uuid7(through_message_id):
        through_ms = uuid7_unix_ms(through_message_id)
        if through_ms != upto_ms:
            return through_ms < upto_ms
        return str(through_message_id).lower() == str(upto_message_id).lower()
    created = _as_utc(through_created_at)
    if created is None:
        return False
    return int(created.timestamp() * 1000) < upto_ms
