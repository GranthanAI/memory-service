"""Summary compaction: fitting the result to the token limit (refine -> deterministic fallback), periodic consolidation,
idempotency and the signal back to the compaction scheduler. Storage and the model are in-memory fakes (no network, no LLM)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.services.summary_service import SummaryOutputError, SummaryService
from app.utils.token_counter import TokenCounter
from tests.support.fakes import FakeCassandra, msg

pytestmark = pytest.mark.asyncio(loop_scope="function")
COUNTER = TokenCounter()
CAP = settings.SUMMARY_MAX_TOKENS


def words(n: int, word: str = "the") -> str:
    """About n tokens ("the", "and", ... are one token each; the sizes below are asserted, not assumed)."""
    return " ".join([word] * n)


def test_the_helper_builds_texts_of_the_intended_size():
    assert abs(COUNTER.count(words(1000)) - 1000) <= 2 and abs(COUNTER.count(words(500, "and")) - 500) <= 2


class FakeMemoryRepo:
    def __init__(self):
        self.snapshot = {"conversation_id": "conv-1", "summary_version": 0, "last_summary_msg_id": None}
        self.invalidated = 0

    async def get_snapshot(self, cid):
        return dict(self.snapshot)

    async def save_snapshot(self, snap):
        self.snapshot = dict(snap)

    async def invalidate_conversation(self, cid):
        self.invalidated += 1


def make(messages, summarize_reply, consolidate=None, scheduler=None):
    cass, repo = FakeCassandra(), FakeMemoryRepo()
    for m in messages:
        cass.append_recent_message("conv-1", m)
    calls = {"summarize": 0, "consolidate": []}

    async def summarize(request):
        calls["summarize"] += 1
        text = summarize_reply(request) if callable(summarize_reply) else summarize_reply
        return SimpleNamespace(summary=text)

    llm = SimpleNamespace(summarize=summarize)
    if consolidate is not None:
        async def consolidate_summary(request):
            calls["consolidate"].append(request.target_tokens)
            return SimpleNamespace(summary=consolidate(request))
        llm.consolidate_summary = consolidate_summary
    return SummaryService(repo, cass, llm, scheduler=scheduler), cass, repo, calls


def stored(cass) -> dict:
    return cass.summaries["conv-1"]


async def test_a_normal_summary_is_committed_unchanged_and_never_consolidated():
    service, cass, _, calls = make([msg(1), msg(2)], "short and sweet", consolidate=lambda r: "SHOULD NOT RUN")
    await service.process_incremental_summary("conv-1")
    assert stored(cass)["summary_text"] == "short and sweet" and calls["consolidate"] == []


async def test_an_oversized_summary_is_refined_and_the_refined_text_is_validated_and_stored():
    service, cass, _, calls = make([msg(1)], words(CAP + 400), consolidate=lambda r: words(CAP // 2, "and"))
    await service.process_incremental_summary("conv-1")
    text = stored(cass)["summary_text"]
    assert calls["consolidate"] == [int(CAP * 0.8)] and text.startswith("and")
    assert COUNTER.count(text) <= CAP


async def test_a_summary_still_over_the_limit_after_refinement_is_cut_deterministically_never_stored_oversized():
    service, cass, _, calls = make([msg(1)], words(CAP + 400), consolidate=lambda r: words(CAP + 300))
    await service.process_incremental_summary("conv-1")
    text = stored(cass)["summary_text"]
    assert len(calls["consolidate"]) == settings.SUMMARY_MAX_REFINEMENTS            # bounded: no endless refinement loop
    assert COUNTER.count(text) <= CAP and len(text) <= 6000 and text.endswith("…")


async def test_oversized_output_without_a_refinement_capability_is_trimmed_not_rejected_and_the_watermark_advances():
    service, cass, repo, _ = make([msg(1), msg(2)], "x" * 7000)
    await service.process_incremental_summary("conv-1")
    row = stored(cass)
    assert len(row["summary_text"]) <= 6000 and row["through_message_id"] == "m0002"     # the new messages are NOT lost
    assert repo.snapshot["summary_version"] == 1


async def test_a_failing_refinement_falls_back_instead_of_failing_the_update():
    async def boom(request):
        raise RuntimeError("provider down")

    service, cass, _, _ = make([msg(1)], words(CAP + 400))
    service.llm_service.consolidate_summary = boom
    await service.process_incremental_summary("conv-1")
    assert COUNTER.count(stored(cass)["summary_text"]) <= CAP


async def test_a_summary_close_to_the_limit_is_consolidated_once_so_it_cannot_only_grow():
    near = words(int(CAP * 0.9))
    service, cass, _, calls = make([msg(1)], near, consolidate=lambda r: words(int(CAP * 0.5), "and"))
    await service.process_incremental_summary("conv-1")
    assert calls["consolidate"] == [int(CAP * settings.SUMMARY_CONSOLIDATE_TARGET_RATIO)]
    assert stored(cass)["summary_text"].startswith("and")


async def test_a_consolidation_that_is_not_shorter_is_ignored():
    near = words(int(CAP * 0.9))
    service, cass, _, _ = make([msg(1)], near, consolidate=lambda r: words(int(CAP * 0.95), "and"))
    await service.process_incremental_summary("conv-1")
    assert stored(cass)["summary_text"] == near


@pytest.mark.parametrize("empty", ["", "   "])
async def test_an_empty_result_is_still_rejected_and_the_previous_summary_is_kept(empty):
    service, cass, _, _ = make([msg(1)], empty)
    cass.summaries["conv-1"] = {"conversation_id": "conv-1", "summary_text": "previous", "summary_version": 4,
                                "through_message_id": "m0000", "through_created_at": msg(0)["created_at"]}
    with pytest.raises(SummaryOutputError):
        await service.process_incremental_summary("conv-1")
    assert stored(cass)["summary_text"] == "previous" and stored(cass)["summary_version"] == 4


async def test_running_the_same_update_twice_creates_one_version():
    service, cass, _, calls = make([msg(1), msg(2)], "the summary")
    await service.process_incremental_summary("conv-1")
    await service.process_incremental_summary("conv-1")            # a duplicate request / redelivered event
    assert stored(cass)["summary_version"] == 1 and calls["summarize"] == 1


async def test_the_scheduler_is_reset_after_a_commit_and_after_a_no_op_so_pressure_restarts():
    scheduler = SimpleNamespace(reset=AsyncMock())
    service, _, _, _ = make([msg(1)], "s", scheduler=scheduler)
    await service.process_incremental_summary("conv-1")
    await service.process_incremental_summary("conv-1")
    assert scheduler.reset.await_count == 2


async def test_a_summary_far_behind_reads_deeper_once_so_old_messages_are_not_lost():
    # 250 messages exist but the summary only covers the first 10: the first read (newest 200) starts after the watermark.
    messages = [msg(i) for i in range(1, 251)]
    service, cass, _, calls = make(messages, lambda r: f"covers {len(r.new_messages)}")
    cass.summaries["conv-1"] = {"conversation_id": "conv-1", "summary_text": "old", "summary_version": 1,
                                "through_message_id": "m0010", "through_created_at": msg(10)["created_at"]}
    reads = []
    original = cass.get_recent_messages
    cass.get_recent_messages = lambda cid, limit=20: (reads.append(limit), original(cid, limit))[1]
    await service.process_incremental_summary("conv-1")
    assert reads[:2] == [200, 1000]                                  # the catch-up read
    assert calls["summarize"] >= 1 and stored(cass)["summary_version"] >= 2
