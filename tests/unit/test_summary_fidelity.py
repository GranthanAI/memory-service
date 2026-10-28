"""Phase 6: summary fidelity - anchor ledger, validation of every new summary (retry once, then keep the old summary and the
watermark), per-section trimming, rendering. Offline: a scripted summarizer stands in for the model."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from app.core.metrics import COMPACTION_RESULTS
from app.proto import memory_pb2
from app.proto.memory_handlers import fit_context_to_budget
from app.services import anchors as ledger
from app.services.anchors import Anchor
from app.services.summary_service import SummaryService, SummaryValidationError
from tests.support.fakes import FakeCassandra, ScriptedSummarizer, msg

pytestmark = pytest.mark.asyncio


def counter_value(outcome: str) -> float:
    return COMPACTION_RESULTS.labels(outcome=outcome)._value.get()


# ─── deterministic extraction ────────────────────────────────────────────────────────────────────────────────────────────


def test_identifiers_are_pinned_exactly_with_their_source_message():
    messages = [
        {"message_id": "m1", "content": "Deploy to https://staging.example.com:8443/api using config/settings.yaml, v2.3.1"},
        {"message_id": "m2", "content": "The budget is $4,500 and the ticket is JIRA-1234; commit 9fceb02d0ae598e95dc970b74767f19372d61af8"},
        {"message_id": "m3", "content": "we should maybe talk about 3 things, and that is all"},
    ]
    found = {a.text: a.source_message_id for a in ledger.extract_identifiers(messages)}
    for expected in ("https://staging.example.com:8443/api", "settings.yaml", "2.3.1", "$4,500", "JIRA-1234"):
        assert any(expected in text for text in found), expected
    assert found[next(t for t in found if "staging" in t)] == "m1"
    assert not any(t.lower() in ("we", "maybe", "that", "3") for t in found)          # prose and tiny numbers are not anchors
    assert all(a.kind == "identifier" for a in ledger.extract_identifiers(messages))


def test_identifier_extraction_is_deduplicated_and_capped():
    messages = [{"message_id": f"m{i}", "content": f"item id_{i}x7 and again id_{i}x7"} for i in range(100)]
    anchors = ledger.extract_identifiers(messages, limit=25)
    assert len(anchors) == 25 and len({a.text.lower() for a in anchors}) == 25


def test_llm_anchor_answers_are_parsed_defensively():
    messages = [{"message_id": "m0"}, {"message_id": "m1"}]
    raw = 'Here you go: {"anchors": [{"kind": "decision", "text": "Use Postgres because of JSONB", "message": 1}, ' \
          '{"kind": "open_item", "text": "Who owns the migration?", "message": 0}, {"kind": "banana", "text": "x"}, ' \
          '{"kind": "constraint", "text": ""}], "done": ["Who owns the old migration?"]}'
    anchors, done = ledger.parse_llm_anchors(raw, messages)
    assert [(a.kind, a.source_message_id, a.status) for a in anchors] == [("decision", "m1", "active"), ("open_item", "m0", "open")]
    assert done == ["Who owns the old migration?"]
    with pytest.raises(ValueError):
        ledger.parse_llm_anchors("no json", messages)


def test_merge_keeps_every_existing_anchor_and_closes_only_open_items_marked_done():
    existing = [{"kind": "open_item", "text": "Decide who owns the migration", "status": "open", "source_message_id": "m1"},
                {"kind": "decision", "text": "Use Postgres", "status": "active", "source_message_id": "m2"}]
    merged = ledger.merge_ledger(existing, [Anchor("identifier", "8443", "m3")], ["owns the migration"])
    by_text = {a.text: a for a in merged}
    assert set(by_text) == {"Decide who owns the migration", "Use Postgres", "8443"}
    assert by_text["Decide who owns the migration"].status == "done" and by_text["Use Postgres"].status == "active"


def test_render_order_puts_open_items_first_hides_closed_ones_and_never_drops_judged_kinds_for_size():
    rows = [{"kind": "identifier", "text": f"id{i}x", "status": "active"} for i in range(50)] + [
        {"kind": "decision", "text": "Use Postgres", "status": "active"},
        {"kind": "open_item", "text": "Pick a region", "status": "open"},
        {"kind": "open_item", "text": "Old question", "status": "done"},
        {"kind": "constraint", "text": "Answer in German", "status": "active"}]
    shown = ledger.render_order(rows, max_chars=200)
    kinds = [a["kind"] for a in shown]
    assert kinds[:3] == ["open_item", "constraint", "decision"] and "Old question" not in {a["text"] for a in shown}
    assert kinds.count("identifier") < 50                                            # identifiers give way to the size cap


# ─── validation ──────────────────────────────────────────────────────────────────────────────────────────────────────────


def check(new_summary, previous="", messages="", prior=(), merged=()):
    return ledger.validate_summary(new_summary=new_summary, previous_summary=previous, new_messages_text=messages,
                                   prior_ledger=list(prior), merged_ledger=list(merged))


def test_an_entity_in_neither_the_old_summary_nor_the_new_messages_is_flagged():
    report = check("Deploys to port 9000 with v9.9.9", previous="Deploys to port 8443", messages="we moved to v2.0.0")
    assert "9000" in report.unsupported_entities and "v9.9.9" in report.unsupported_entities and not report.ok
    assert "9000" in report.feedback()
    assert check("Deploys to port 8443 with v2.0.0", previous="Deploys to port 8443", messages="we moved to v2.0.0").ok


def test_an_open_item_cannot_leave_the_summary_unless_a_message_marked_it_done():
    open_item = Anchor("open_item", "Decide who owns the database migration", "m1", "open")
    previous = "OPEN ITEMS: decide who owns the database migration."
    assert check("CURRENT STATE: all quiet.", previous=previous, merged=[open_item]).dropped_open_items
    closed = ledger.merge_ledger([{"kind": "open_item", "text": open_item.text, "status": "open"}], [], ["who owns the migration"])
    assert check("CURRENT STATE: all quiet.", previous=previous, merged=closed).ok
    assert check("OPEN ITEMS: someone still has to decide the database migration ownership.", previous=previous,
                 merged=[open_item]).ok                                                # a paraphrase still counts as kept


def test_a_pinned_fact_missing_from_both_the_summary_and_the_ledger_is_reported_as_lost():
    prior = [{"kind": "decision", "text": "Use Postgres for JSONB support"}]
    assert check("Nothing relevant.", prior=prior, merged=[]).lost_anchors == ["Use Postgres for JSONB support"]
    assert check("Nothing relevant.", prior=prior, merged=[Anchor("decision", "Use Postgres for JSONB support")]).ok


# ─── the summary job ─────────────────────────────────────────────────────────────────────────────────────────────────────


def world(reply, messages=None, extractor=None):
    cass = FakeCassandra()
    cass.messages["c1"] = list(reversed(messages or [msg(1, content="Our API runs on port 8443 behind nginx"),
                                                     msg(2, "assistant", "Noted: port 8443")]))
    cass.snapshots["c1"] = {"conversation_id": "c1", "user_id": "u", "summary_version": 0}
    summarizer = ScriptedSummarizer(reply)
    if extractor is not None:
        summarizer.extract_anchors = extractor
    repo = SimpleNamespace(update_snapshot_summary_fields=AsyncMock(), invalidate_conversation=AsyncMock(),
                           get_snapshot=AsyncMock(return_value=dict(cass.snapshots["c1"])))
    return cass, summarizer, SummaryService(repo, cass, summarizer)


async def test_a_summary_that_forgets_an_identifier_still_commits_because_the_ledger_pins_it():
    cass, summarizer, service = world("CURRENT STATE: the API is deployed behind nginx.")          # no "8443" in the text
    await service.process_incremental_summary("c1")
    assert cass.summaries["c1"]["summary_version"] == 1
    assert any(a["text"] == "8443" and a["kind"] == "identifier" for a in cass.list_anchors("c1"))
    shown = ledger.render_order(cass.list_anchors("c1"), 3000)
    assert any(a["text"] == "8443" for a in shown)                                  # and it is rendered next to the summary


async def test_an_invented_number_triggers_exactly_one_corrective_retry_with_the_offending_items_listed():
    replies = iter(["Runs on port 9000 behind nginx.", "Runs on port 8443 behind nginx."])
    cass, summarizer, service = world(lambda request: next(replies))
    await service.process_incremental_summary("c1")
    assert len(summarizer.calls) == 2 and summarizer.calls[0].feedback is None
    assert "9000" in summarizer.calls[1].feedback
    assert cass.summaries["c1"]["summary_text"] == "Runs on port 8443 behind nginx."


async def test_a_summary_that_keeps_failing_validation_leaves_the_old_summary_and_the_watermark_untouched():
    cass, summarizer, service = world("Runs on port 9000.")
    before = counter_value("unsupported_entity")
    with pytest.raises(SummaryValidationError):
        await service.process_incremental_summary("c1")
    assert len(summarizer.calls) == 2                                               # the first attempt + ONE retry
    assert "c1" not in cass.summaries and cass.list_anchors("c1") == []             # nothing committed, no ledger written
    assert counter_value("unsupported_entity") == before + 1
    # the batch is still unsummarized: the next job (with a model that behaves) summarizes the same messages
    summarizer.reply = "Runs on port 8443."
    await service.process_incremental_summary("c1")
    assert cass.summaries["c1"]["summary_version"] == 1


async def test_dropping_an_open_item_is_rejected_and_counted_as_anchor_lost():
    cass = FakeCassandra()
    cass.messages["c1"] = [msg(3, content="Any update on the vendor contract?"), msg(2, content="Still pending."), msg(1)]
    cass.snapshots["c1"] = {"conversation_id": "c1", "user_id": "u", "summary_version": 1}
    cass.summaries["c1"] = {"conversation_id": "c1", "summary_text": "OPEN ITEMS: sign the vendor contract with Acme.",
                            "summary_version": 1, "through_message_id": "m0001", "through_created_at": msg(1)["created_at"]}
    cass.upsert_anchors("c1", [Anchor("open_item", "Sign the vendor contract with Acme", "m0001", "open")])
    summarizer = ScriptedSummarizer("CURRENT STATE: quiet week.")
    repo = SimpleNamespace(update_snapshot_summary_fields=AsyncMock(), invalidate_conversation=AsyncMock(),
                           get_snapshot=AsyncMock(return_value=dict(cass.snapshots["c1"])))
    before = counter_value("anchor_lost")
    with pytest.raises(SummaryValidationError):
        await SummaryService(repo, cass, summarizer).process_incremental_summary("c1")
    assert counter_value("anchor_lost") == before + 1 and cass.summaries["c1"]["summary_version"] == 1


async def test_a_message_that_finishes_an_open_item_lets_it_leave_the_summary():
    cass = FakeCassandra()
    cass.messages["c1"] = [msg(3, content="The vendor contract is signed."), msg(1)]
    cass.snapshots["c1"] = {"conversation_id": "c1", "user_id": "u", "summary_version": 1}
    cass.summaries["c1"] = {"conversation_id": "c1", "summary_text": "OPEN ITEMS: sign the vendor contract with Acme.",
                            "summary_version": 1, "through_message_id": "m0001", "through_created_at": msg(1)["created_at"]}
    cass.upsert_anchors("c1", [Anchor("open_item", "Sign the vendor contract with Acme", "m0001", "open")])
    summarizer = ScriptedSummarizer("CURRENT STATE: contract done.")
    summarizer.extract_anchors = AsyncMock(return_value=([], ["sign the vendor contract with Acme"]))
    repo = SimpleNamespace(update_snapshot_summary_fields=AsyncMock(), invalidate_conversation=AsyncMock(),
                           get_snapshot=AsyncMock(return_value=dict(cass.snapshots["c1"])))
    await SummaryService(repo, cass, summarizer).process_incremental_summary("c1")
    assert cass.summaries["c1"]["summary_version"] == 2
    assert next(a for a in cass.list_anchors("c1") if a["kind"] == "open_item")["status"] == "done"


async def test_model_extracted_anchors_are_stored_and_a_failing_extractor_does_not_block_the_summary():
    good = AsyncMock(return_value=([Anchor("decision", "Use nginx because it is already deployed", "m0001")], []))
    cass, _, service = world("Runs behind nginx on port 8443.", extractor=good)
    await service.process_incremental_summary("c1")
    assert {a["kind"] for a in cass.list_anchors("c1")} == {"identifier", "decision"}

    broken = AsyncMock(side_effect=RuntimeError("llm down"))
    cass, _, service = world("Runs behind nginx on port 8443.", extractor=broken)
    await service.process_incremental_summary("c1")
    assert cass.summaries["c1"]["summary_version"] == 1 and [a["kind"] for a in cass.list_anchors("c1")] == ["identifier"]


async def test_the_ledger_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "ANCHOR_LEDGER_ENABLED", False)
    cass, summarizer, service = world("Runs on port 9000.")                          # would fail validation if it were on
    await service.process_incremental_summary("c1")
    assert cass.summaries["c1"]["summary_version"] == 1 and cass.list_anchors("c1") == [] and len(summarizer.calls) == 1


async def test_a_summary_written_before_the_ledger_existed_is_bootstrapped_not_rejected():
    cass = FakeCassandra()
    cass.messages["c1"] = [msg(2, content="ok"), msg(1)]
    cass.snapshots["c1"] = {"conversation_id": "c1", "user_id": "u", "summary_version": 1}
    cass.summaries["c1"] = {"conversation_id": "c1", "summary_text": "KEY FACTS: build 4417 on port 8443.", "summary_version": 1,
                            "through_message_id": "m0001", "through_created_at": msg(1)["created_at"]}
    repo = SimpleNamespace(update_snapshot_summary_fields=AsyncMock(), invalidate_conversation=AsyncMock(),
                           get_snapshot=AsyncMock(return_value=dict(cass.snapshots["c1"])))
    service = SummaryService(repo, cass, ScriptedSummarizer("CURRENT STATE: fine."))      # the new text lost 4417 and 8443
    await service.process_incremental_summary("c1")
    assert cass.summaries["c1"]["summary_version"] == 2
    assert {"4417", "8443"} <= {a["text"] for a in cass.list_anchors("c1")}                # moved into the ledger


# ─── per-section trimming ────────────────────────────────────────────────────────────────────────────────────────────────


def test_fallback_trim_cuts_every_section_to_its_own_budget_instead_of_the_tail(monkeypatch):
    monkeypatch.setattr(settings, "SUMMARY_MAX_TOKENS", 300)
    service = SummaryService(SimpleNamespace(), FakeCassandra(), ScriptedSummarizer())
    def bullets(tag, n):
        return "\n".join(f"- {tag} point number {i} with a few extra words to take up room" for i in range(n))
    text = "\n".join(f"{h}\n{bullets(h[:3], 25)}" for h in (
        "CONSTRAINTS & PREFERENCES", "DECISIONS", "PROPOSALS", "CORRECTIONS", "OPEN ITEMS", "CURRENT STATE", "KEY FACTS & OUTCOMES",
        "TIMELINE"))
    trimmed = service._trim(text)
    assert service.counter.count(trimmed) <= 300
    for heading in ("CONSTRAINTS & PREFERENCES", "DECISIONS", "OPEN ITEMS", "KEY FACTS & OUTCOMES"):
        assert heading in trimmed, heading                         # a tail cut would have removed KEY FACTS and TIMELINE entirely
    lines = trimmed.split("\n")
    assert all(l.startswith("- ") and l.rstrip().endswith("room") or not l.startswith("- ") for l in lines)   # whole bullets only
    assert trimmed.index("CONSTRAINTS") < trimmed.index("KEY FACTS")


def test_text_without_the_expected_headings_still_falls_back_to_the_token_cut(monkeypatch):
    monkeypatch.setattr(settings, "SUMMARY_MAX_TOKENS", 200)
    service = SummaryService(SimpleNamespace(), FakeCassandra(), ScriptedSummarizer())
    assert service.counter.count(service._trim("word " * 2000)) <= 200


# ─── rendering / transport ───────────────────────────────────────────────────────────────────────────────────────────────


def test_anchors_travel_in_the_grpc_response_and_are_never_dropped_by_the_token_budget():
    result = {"current_summary": "s " * 50, "ancestor_summaries": [], "relevant_facts": [], "retrieved_history": [],
              "short_term_messages": [{"content": "m " * 200} for _ in range(5)], "metadata": {},
              "anchors": [{"kind": "decision", "text": "Use Postgres", "source_message_id": "m1"}]}
    fit_context_to_budget(result, 120)
    assert result["anchors"] and len(result["short_term_messages"]) == 1
    message = memory_pb2.GetMemoryContextResponse(anchors=[memory_pb2.Anchor(kind="decision", text="Use Postgres", source_message_id="m1")])
    round_trip = memory_pb2.GetMemoryContextResponse.FromString(message.SerializeToString())
    assert (round_trip.anchors[0].kind, round_trip.anchors[0].text) == ("decision", "Use Postgres")
