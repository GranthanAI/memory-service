"""
User memory controls, enforced inside memory-service (the memory owner):

* memory on/off ("pause"): when off, saved memories are not read into answers
  (RetrievalService) and no new ones are learned (FactWorker skips extraction,
  LongMemoryService refuses writes). Conversation history inside a conversation is
  not "saved memory" and keeps working.
* tombstones: a memory the user deleted or corrected is never stored again, even if a
  later background extraction (e.g. from an older conversation summary) produces the
  same statement.

Fail closed: if the setting cannot be read, memory is treated as OFF for that request
(no personal data is used or stored) rather than silently ignoring a pause.
"""

import hashlib
import logging
import re

from app.repositories.cassandra_repository import CassandraRepository

logger = logging.getLogger("memory_service.services.memory_controls")

_SPACE = re.compile(r"\s+")


def statement_hash(statement: str) -> str:
    """Stable identity of a fact statement: case-, whitespace- and trailing-punctuation-
    insensitive, so trivial rewordings of the same sentence match."""
    normalised = _SPACE.sub(" ", statement.strip().lower()).rstrip(".!? ")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


class MemoryControls:
    def __init__(self, cassandra_repo: CassandraRepository):
        self.repo = cassandra_repo

    def is_enabled(self, user_id: str) -> bool:
        if not user_id or user_id == "unknown_user":
            return False  # never learn or use memories without a real owner
        try:
            value = self.repo.get_memory_enabled(user_id)
        except Exception as e:
            logger.error(f"Could not read memory setting for {user_id}; treating memory as OFF: {e}")
            return False
        return True if value is None else bool(value)

    def set_enabled(self, user_id: str, enabled: bool) -> None:
        self.repo.set_memory_enabled(user_id, enabled)

    def tombstones(self, user_id: str) -> set:
        try:
            return self.repo.get_fact_tombstones(user_id)
        except Exception as e:
            logger.error(f"Could not read fact tombstones for {user_id}: {e}")
            raise

    def forget(self, user_id: str, statement: str) -> None:
        self.repo.add_fact_tombstone(user_id, statement_hash(statement))
