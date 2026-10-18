"""Ancestor summaries are returned only when provably valid at the child's branch point, and only for the
caller's own conversations; the gRPC response carries them and the other memory layers."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.proto import memory_pb2
from app.proto.memory_handlers import MemoryContextHandler
from app.services.context_builder import ContextBuilder, summary_valid_at_branch

pytestmark = pytest.mark.asyncio(loop_scope="function")

USER, OTHER = "user-1", "user-2"


def mid(n: int) -> str:
    """Time-ordered ids like UUIDv7: larger n = later."""
    return str(uuid.UUID(int=(0x018F << 80) + n))


class Repo:
    """Just what ContextBuilder reads: snapshots (ownership) and summary rows."""

    def __init__(self):
        self.snapshots, self.summaries = {}, {}
        self.cassandra_repo = SimpleNamespace(get_summary=lambda cid: self.summaries.get(cid))

    async def get_snapshot(self, cid):
        return self.snapshots.get(cid)

    def add(self, cid, summary, through, owner=USER, version=1):
        self.snapshots[cid] = {"conversation_id": cid, "user_id": owner}
        self.summaries[cid] = {"summary_text": summary, "through_message_id": through, "summary_version": version}


def builder(repo):
    return ContextBuilder(SimpleNamespace(memory_repo=repo), graph_client=AsyncMock())


def test_validity_is_a_strict_ordering_check_and_fails_closed():
    assert summary_valid_at_branch(mid(5), mid(5))              # covers exactly up to the branch point
    assert summary_valid_at_branch(mid(4), mid(5))
    assert not summary_valid_at_branch(mid(6), mid(5))          # includes a message after the branch point
    for bad in (None, "", "not-a-uuid"):
        assert not summary_valid_at_branch(bad, mid(5)), bad    # unknown coverage is never trusted
    assert not summary_valid_at_branch(mid(1), "garbage")


async def test_a_summary_that_went_past_the_branch_point_is_not_returned():
    repo = Repo()
    repo.add("A", "A summary through message 9", through=mid(9))
    out = await builder(repo).ancestor_summaries(USER, [{"conversation_id": "A", "upto_message_id": mid(5)}])
    assert out == []                                            # the parent kept talking after the fork


async def test_summaries_valid_at_each_branch_point_are_returned_root_first():
    repo = Repo()
    repo.add("A", "A: KV cache", through=mid(3), version=2)
    repo.add("B", "B: batching", through=mid(14), version=1)
    out = await builder(repo).ancestor_summaries(USER, [
        {"conversation_id": "A", "upto_message_id": mid(5)}, {"conversation_id": "B", "upto_message_id": mid(20)}])
    assert [(a["conversation_id"], a["summary"], a["summary_version"]) for a in out] == [("A", "A: KV cache", 2), ("B", "B: batching", 1)]


async def test_other_users_missing_and_legacy_ancestors_are_skipped():
    repo = Repo()
    repo.add("theirs", "private", through=mid(1), owner=OTHER)
    repo.add("legacy", "no watermark", through=None)
    repo.add("empty", "   ", through=mid(1))
    refs = [{"conversation_id": c, "upto_message_id": mid(9)} for c in ("theirs", "missing", "legacy", "empty")]
    assert await builder(repo).ancestor_summaries(USER, refs) == []
    assert await builder(repo).ancestor_summaries("", refs) == []          # unknown caller: nothing
    assert await builder(repo).ancestor_summaries(USER, None) == []
    assert await builder(repo).ancestor_summaries(USER, [{"conversation_id": "A"}]) == []   # no boundary given


async def test_grpc_response_carries_every_layer_and_the_request_lineage_reaches_the_builder():
    context_builder = AsyncMock()
    context_builder.build_context.return_value = {
        "short_term_messages": [], "current_summary": "older part of this chat", "relevant_facts": [{"statement": "prefers concise answers", "importance": 0.8, "category": "preferences", "fact_id": "f-1"}],
        "ancestor_summaries": [{"conversation_id": "A", "summary": "A: KV cache", "covers_through_message_id": mid(3), "summary_version": 2}],
        "summary_meta": {"summary_version": 4, "through_message_id": mid(7)},
    }
    handler = MemoryContextHandler(context_builder, AsyncMock(generate_embedding=AsyncMock(return_value=[0.1])))
    request = memory_pb2.GetMemoryContextRequest(
        conversation_id="C", user_id=USER, query="q", ancestors=[memory_pb2.AncestorRef(conversation_id="A", upto_message_id=mid(5))])
    context = SimpleNamespace(invocation_metadata=lambda: (), set_code=lambda *_: None, set_details=lambda *_: None)

    response = await handler.GetMemoryContext(request, context)
    wire = memory_pb2.GetMemoryContextResponse.FromString(response.SerializeToString())

    kwargs = context_builder.build_context.call_args.kwargs
    assert kwargs["ancestors"] == [{"conversation_id": "A", "upto_message_id": mid(5)}] and kwargs["use_graph_parents"] is False
    assert wire.medium_summary == "older part of this chat" and (wire.summary_version, wire.summary_through_message_id) == (4, mid(7))
    assert [(a.conversation_id, a.summary, a.covers_through_message_id) for a in wire.ancestor_summaries] == [("A", "A: KV cache", mid(3))]
    assert (wire.long_term_facts[0].content, wire.long_term_facts[0].fact_id) == ("prefers concise answers", "f-1")
