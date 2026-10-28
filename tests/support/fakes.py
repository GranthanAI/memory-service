"""In-memory doubles for the REAL memory stack (RedisRepository -> MemoryRepository -> RetrievalService -> ContextBuilder,
SummaryService, EventDispatcher). They reproduce the storage semantics that matter for ordering and watermarks:

  * Redis recent window: LPUSH + LTRIM, so index 0 is the NEWEST message.
  * Cassandra conversation_recent_messages: CLUSTERING ORDER BY (created_at DESC, message_id ASC), never pruned.
  * Cassandra conversation_summaries: compare-and-set on summary_version.

No network, no LLM provider: the summarizer is a scripted fake.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def at(n: int) -> datetime:
    return T0 + timedelta(seconds=n)


def msg(n: int, role: str = "user", content: Optional[str] = None) -> Dict[str, Any]:
    """Message number n: ids and timestamps grow with n, so n is its chronological position."""
    return {"message_id": f"m{n:04d}", "role": role, "content": content if content is not None else f"message {n}",
            "created_at": at(n)}


class FakeRedis:
    def __init__(self) -> None:
        self.kv: Dict[str, Any] = {}
        self.lists: Dict[str, List[str]] = {}
        self.hashes: Dict[str, Dict[str, str]] = {}
        self.expiry: Dict[str, int] = {}

    # --- plain ops
    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        if ex:
            self.expiry[key] = ex
        return True

    async def delete(self, *keys):
        for k in keys:
            self.kv.pop(k, None)
            self.lists.pop(k, None)
            self.hashes.pop(k, None)

    async def incrby(self, key, amount=1):
        self.kv[key] = int(self.kv.get(key, 0)) + amount
        return self.kv[key]

    async def expire(self, key, seconds):
        self.expiry[key] = seconds

    async def lrange(self, key, start, end):
        data = self.lists.get(key, [])
        return list(data[start:] if end == -1 else data[start:end + 1])

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def pipeline(self, transaction=True):
        return _Pipe(self)


class _Pipe:
    def __init__(self, redis: FakeRedis) -> None:
        self.r = redis
        self.ops: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def lpush(self, key, value):
        self.ops.append(("lpush", key, value))

    def rpush(self, key, *values):
        self.ops.append(("rpush", key, values))

    def ltrim(self, key, start, end):
        self.ops.append(("ltrim", key, start, end))

    def expire(self, key, seconds):
        self.ops.append(("expire", key, seconds))

    def delete(self, key):
        self.ops.append(("delete", key))

    def hset(self, key, mapping=None, **kwargs):
        self.ops.append(("hset", key, mapping or kwargs))

    async def execute(self):
        for op in self.ops:
            kind = op[0]
            if kind == "lpush":
                self.r.lists.setdefault(op[1], []).insert(0, op[2])
            elif kind == "rpush":
                self.r.lists.setdefault(op[1], []).extend(op[2])
            elif kind == "ltrim":
                lst = self.r.lists.get(op[1], [])
                self.r.lists[op[1]] = lst[op[2]:op[3] + 1]
            elif kind == "expire":
                self.r.expiry[op[1]] = op[2]
            elif kind == "delete":
                self.r.lists.pop(op[1], None)
            elif kind == "hset":
                self.r.hashes.setdefault(op[1], {}).update({k: str(v) for k, v in op[2].items()})
        self.ops = []


class FakeCassandra:
    """The subset of CassandraRepository the memory stack uses."""

    def __init__(self) -> None:
        self.messages: Dict[str, List[Dict[str, Any]]] = {}
        self.summaries: Dict[str, Dict[str, Any]] = {}
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.outbox: List[Dict[str, Any]] = []
        self.summary_reads = 0

    def append_recent_message(self, conversation_id, message):
        rows = self.messages.setdefault(conversation_id, [])
        rows.append(dict(message))
        rows.sort(key=lambda m: (m["created_at"], m["message_id"]), reverse=True)    # created_at DESC

    def get_recent_messages(self, conversation_id, limit=20):
        return [dict(m) for m in self.messages.get(conversation_id, [])[:limit]]

    def get_summary(self, conversation_id):
        self.summary_reads += 1
        row = self.summaries.get(conversation_id)
        return dict(row) if row else None

    def write_summary_if_current(self, summary, expected_version):
        current = self.summaries.get(summary["conversation_id"])
        if (current["summary_version"] if current else None) != expected_version:
            return False
        self.summaries[summary["conversation_id"]] = dict(summary)
        return True

    def upsert_summary(self, summary):
        self.summaries[summary["conversation_id"]] = dict(summary)

    def get_snapshot(self, conversation_id):
        s = self.snapshots.get(conversation_id)
        return dict(s) if s else None

    def upsert_snapshot(self, snapshot):
        self.snapshots[snapshot["conversation_id"]] = dict(snapshot)

    def archive_summary_version(self, row):
        archive = self.__dict__.setdefault("archive", {}).setdefault(row["conversation_id"], {})
        archive.setdefault(int(row["summary_version"]), dict(row))              # IF NOT EXISTS

    def list_summary_versions(self, conversation_id, limit=50):
        rows = self.__dict__.get("archive", {}).get(conversation_id, {})
        return [dict(rows[v]) for v in sorted(rows, reverse=True)][:limit]

    def update_snapshot_summary_fields(self, conversation_id, summary_version, last_summary_msg_id):
        snap = self.snapshots.setdefault(conversation_id, {"conversation_id": conversation_id})
        snap["summary_version"], snap["last_summary_msg_id"] = summary_version, last_summary_msg_id

    def update_snapshot_state_if_current(self, conversation_id, state, new_version, expected_version):
        snap = self.snapshots.get(conversation_id)
        if not snap or snap.get("snapshot_version") != expected_version:
            return False
        snap["state"], snap["snapshot_version"] = state, new_version
        return True

    def insert_outbox_job(self, job):
        self.outbox.append(dict(job))


class ScriptedSummarizer:
    """Stands in for LLMService.summarize/compress: deterministic, offline, records every call."""

    def __init__(self, reply=None) -> None:
        self.calls: list = []
        self.reply = reply

    async def summarize(self, request):
        from app.schemas.llm import SummarizeResponse
        self.calls.append(request)
        if callable(self.reply):
            text = self.reply(request)
        elif self.reply is not None:
            text = self.reply
        else:
            text = (request.previous_summary or "").strip() + " | " + "; ".join(m.content for m in request.new_messages)
        if isinstance(text, Exception):
            raise text
        return SummarizeResponse(summary=text.strip())
