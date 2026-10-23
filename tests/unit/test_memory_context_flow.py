"""Context assembly over the REAL repository stack (Redis window + durable Cassandra rows + summary watermark), using
in-memory doubles for storage and no LLM. These pin down how the conversation-memory layers are built for a request:
chronological order, no re-sending of what the summary already covers, no silent hole between summary and window, and a
bounded, branch-safe search of older messages."""

from unittest.mock import MagicMock

import pytest

from app.repositories.memory_repository import MemoryRepository
from app.repositories.redis_repository import RedisRepository
from app.services.context_builder import ContextBuilder
from app.services.retrieval_service import RetrievalService
from tests.support.fakes import FakeCassandra, FakeRedis, at, msg

USER, CONV = "user-1", "conv-1"


async def make_stack(count: int, summary_through: int = 0, summary_text: str = "Earlier: the user prefers Postgres."):
    cass, redis = FakeCassandra(), FakeRedis()
    repo = MemoryRepository(cass, RedisRepository(redis))
    cass.snapshots[CONV] = {"conversation_id": CONV, "user_id": USER, "message_count": count, "state": "ACTIVE",
                            "summary_version": 1 if summary_through else 0, "fact_version": 0, "snapshot_version": count,
                            "last_summary_msg_id": None}
    for n in range(1, count + 1):
        await repo.append_recent_message(CONV, msg(n, "user" if n % 2 else "assistant"))
    if summary_through:
        cass.summaries[CONV] = {"conversation_id": CONV, "summary_text": summary_text, "summary_version": 1,
                                "through_message_id": f"m{summary_through:04d}", "through_created_at": at(summary_through)}
    retrieval = RetrievalService(repo, MagicMock(), None)
    return ContextBuilder(retrieval, MagicMock()), cass, redis


def numbers(context) -> list[int]:
    return [int(m["message_id"][1:]) for m in context["short_term_messages"]]


async def build(builder, **kw):
    return await builder.build_context(user_id=USER, conversation_id=CONV, use_graph_parents=False, **kw)


@pytest.mark.asyncio
async def test_recent_turns_are_returned_oldest_first():
    builder, *_ = await make_stack(8)
    context = await build(builder)
    assert numbers(context) == sorted(numbers(context)) == list(range(1, 9))


@pytest.mark.asyncio
async def test_turns_the_summary_already_covers_are_not_sent_again():
    builder, *_ = await make_stack(30, summary_through=20)
    context = await build(builder)
    assert context["current_summary"]
    # everything after the watermark (21..30) is shown verbatim; the older window rows are represented by the summary
    assert numbers(context) == list(range(21, 31))


@pytest.mark.asyncio
async def test_a_short_verbatim_tail_is_kept_even_when_the_summary_is_current():
    builder, *_ = await make_stack(30, summary_through=30)
    context = await build(builder)
    assert numbers(context) == list(range(25, 31))          # MIN verbatim tail (6): continuity of the last exchange


@pytest.mark.asyncio
async def test_messages_between_a_stale_summary_and_the_window_are_not_silently_lost():
    # The summary job is behind (covers 1..10) and 45 messages exist: the 20-message cache window starts at 26, so
    # messages 11..25 are in neither the summary nor the window unless the gap is read back from durable storage.
    builder, *_ = await make_stack(45, summary_through=10)
    context = await build(builder)
    assert numbers(context) == list(range(11, 46))
