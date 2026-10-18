"""
User-facing management of saved memories (long-term facts). Every method takes the
authenticated user's id (from the verified token) and only ever touches that user's
partition in Cassandra and that user's rows in Milvus.

Deletion order matters: answers read facts from Milvus, so Milvus is deleted first; the
tombstone is written before either, so a background extraction running at the same time
cannot store the statement again.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.core.config import settings
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.milvus_repository import MilvusRepository
from app.services.memory_controls import MemoryControls

logger = logging.getLogger("memory_service.services.user_memory")

STATEMENT_MAX = 1024  # Milvus `statement` column length


class FactNotFoundError(Exception):
    pass


def _public(fact: Dict[str, Any]) -> Dict[str, Any]:
    created = fact.get("created_at")
    if isinstance(created, (int, float)):
        created = datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
    elif isinstance(created, datetime):
        created = created.isoformat()
    return {
        "id": str(fact["fact_id"]),
        "category": fact.get("category") or "general",
        "statement": fact.get("statement") or "",
        "importance": float(fact.get("importance") or 0.0),
        "created_at": created,
        "conversation_id": fact.get("conversation_id") or None,
    }


class UserMemoryService:
    def __init__(
        self,
        cassandra_repo: CassandraRepository,
        milvus_repo: MilvusRepository,
        embedding_client: Any,
        controls: MemoryControls,
    ):
        self.cassandra = cassandra_repo
        self.milvus = milvus_repo
        self.embeddings = embedding_client
        self.controls = controls

    # ─── settings ─────────────────────────────────────────────────────────

    def get_settings(self, user_id: str) -> Dict[str, Any]:
        return {"memory_enabled": self.controls.is_enabled(user_id)}

    def set_settings(self, user_id: str, memory_enabled: bool) -> Dict[str, Any]:
        self.controls.set_enabled(user_id, memory_enabled)
        return {"memory_enabled": memory_enabled}

    # ─── facts ────────────────────────────────────────────────────────────

    def list_facts(self, user_id: str) -> List[Dict[str, Any]]:
        return [_public(f) for f in self.milvus.list_user_facts(user_id)]

    def _require(self, user_id: str, fact_id: uuid.UUID, category: str) -> Dict[str, Any]:
        fact = self.cassandra.get_fact(user_id, category, fact_id)
        if fact is None:
            raise FactNotFoundError(str(fact_id))
        return fact

    async def update_fact(self, user_id: str, fact_id: uuid.UUID, category: str, statement: str) -> Dict[str, Any]:
        """Corrects a memory: re-embeds it, rewrites both stores under the same id and
        tombstones the old wording so it is not re-learned."""
        old = self._require(user_id, fact_id, category)
        statement = statement.strip()
        vector = await self.embeddings.generate_embedding(statement)
        now = datetime.now(timezone.utc)
        version = int(old.get("fact_version") or 1) + 1
        if old["statement"].strip() != statement:
            self.controls.forget(user_id, old["statement"])
        self.milvus.delete_fact(user_id, str(fact_id))
        self.milvus.insert_facts([{
            "fact_id": str(fact_id), "user_id": user_id, "conversation_id": old.get("conversation_id") or "",
            "category": category, "statement": statement, "importance": float(old.get("importance") or 0.5),
            "fact_version": version, "embedding_ver": settings.EMBEDDING_MODEL_VERSION,
            "created_at": (old.get("created_at") or now).timestamp(), "vector": vector,
        }])
        self.cassandra.upsert_fact({
            **old, "statement": statement, "fact_version": version,
            "embedding_version": settings.EMBEDDING_MODEL_VERSION, "updated_at": now,
        })
        return _public({**old, "statement": statement})

    def delete_fact(self, user_id: str, fact_id: uuid.UUID, category: str) -> None:
        fact = self._require(user_id, fact_id, category)
        self.controls.forget(user_id, fact["statement"])
        self.milvus.delete_fact(user_id, str(fact_id))
        self.cassandra.delete_fact(user_id, category, fact_id)

    def clear_all(self, user_id: str) -> int:
        """Deletes every saved memory of the user. Conversation history and per-conversation
        summaries are not saved memories and are not touched (they go with the conversation)."""
        facts = self.milvus.list_user_facts(user_id, limit=16000)
        for fact in facts:
            self.controls.forget(user_id, fact.get("statement") or "")
        self.milvus.delete_user_facts(user_id)
        for fact in facts:
            try:
                self.cassandra.delete_fact(user_id, fact.get("category") or "general", uuid.UUID(str(fact["fact_id"])))
            except Exception as e:  # Milvus (what answers read) is already clean; log and continue
                logger.error(f"Could not delete fact {fact.get('fact_id')} from Cassandra: {e}")
        return len(facts)
