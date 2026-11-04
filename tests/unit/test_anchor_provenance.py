"""Phase 5: only the user's own text and document-sourced text is pinned as an exact, never-dropped fact. What the assistant
said is stored as 'unconfirmed' and never rendered in the pinned ledger (the "Phase 6 conversation pattern": the assistant
guessed HMAC-SHA256 for a phase it had not read, and the guess became a Pinned Fact)."""
import pytest

from app.core.config import settings
from app.services import anchors as ledger
from app.services.anchors import Anchor
from app.services.summary_service import SummaryService
from tests.support.fakes import msg
from tests.unit.test_summary_fidelity import world

pytestmark = pytest.mark.asyncio

PHASE6 = [
    msg(1, "user", "what is phase 6 about? I set MAX_RETRY_ATTEMPTS to 7 in the runbook"),
    msg(2, "assistant", "Phase 6 is about request signing with HMAC-SHA256 and key rotation to v3.2"),
]


def texts(anchors):
    return {a.text for a in anchors}


def test_assistant_guess_is_not_an_active_anchor_but_the_users_identifier_is():
    anchors = ledger.extract_identifiers(PHASE6)
    active = {a.text for a in anchors if a.status == "active"}
    unconfirmed = {a.text for a in anchors if a.status == "unconfirmed"}
    assert "MAX_RETRY_ATTEMPTS" in active
    assert "HMAC-SHA256" not in active and "HMAC-SHA256" in unconfirmed


def test_unconfirmed_anchors_are_never_rendered_as_pinned():
    rows = [{"kind": a.kind, "text": a.text, "source_message_id": a.source_message_id, "status": a.status}
            for a in ledger.extract_identifiers(PHASE6)]
    shown = {a["text"] for a in ledger.render_order(rows, 3000)}
    assert "MAX_RETRY_ATTEMPTS" in shown and "HMAC-SHA256" not in shown


def test_document_sourced_text_is_pinned():
    tool = {"message_id": "m9", "role": "tool", "content": "The runbook says TXN-102394 was refunded"}
    assert [a.status for a in ledger.extract_identifiers([tool]) if a.text == "TXN-102394"] == ["active"]


def test_messages_without_a_role_keep_the_old_behaviour():
    assert all(a.status == "active" for a in ledger.extract_identifiers([{"message_id": "m1", "content": "port 8443"}]))


def test_when_the_user_later_types_it_the_unconfirmed_anchor_is_promoted():
    first = ledger.extract_identifiers([PHASE6[1]])
    prior = [{"kind": a.kind, "text": a.text, "source_message_id": a.source_message_id, "status": a.status} for a in first]
    user_turn = ledger.extract_identifiers([msg(3, "user", "yes, use HMAC-SHA256 as you said")])
    merged = {a.text: a.status for a in ledger.merge_ledger(prior, user_turn, [])}
    assert merged["HMAC-SHA256"] == "active"


def test_assistant_judged_anchors_are_unconfirmed_user_ones_are_active():
    raw = ('{"anchors": [{"kind": "decision", "text": "Use HMAC-SHA256 for phase 6", "message": 1}, '
           '{"kind": "constraint", "text": "Never log secrets", "message": 0}], "done": []}')
    anchors, _ = ledger.parse_llm_anchors(raw, PHASE6)
    status = {a.text: a.status for a in anchors}
    assert status == {"Use HMAC-SHA256 for phase 6": "unconfirmed", "Never log secrets": "active"}


async def test_summary_job_does_not_pin_the_assistants_guess(monkeypatch):
    cass, _s, service = world("CURRENT STATE: phase 6 discussed; MAX_RETRY_ATTEMPTS is 7; signing via HMAC-SHA256.", messages=PHASE6)
    await service.process_incremental_summary("c1")
    by_text = {a["text"]: a["status"] for a in cass.list_anchors("c1")}
    assert by_text["MAX_RETRY_ATTEMPTS"] == "active" and by_text.get("HMAC-SHA256") == "unconfirmed"
    shown = {a["text"] for a in ledger.render_order(cass.list_anchors("c1"), 3000)}
    assert "HMAC-SHA256" not in shown and "MAX_RETRY_ATTEMPTS" in shown


async def test_a_summary_may_drop_an_unconfirmed_item_without_failing_validation():
    cass, _s, service = world("CURRENT STATE: phase 6 discussed; MAX_RETRY_ATTEMPTS is 7.", messages=PHASE6)
    await service.process_incremental_summary("c1")
    assert cass.summaries["c1"]["summary_version"] == 1


def test_flag_off_restores_pinning_of_assistant_text(monkeypatch):
    monkeypatch.setattr(settings, "ANCHOR_USER_ONLY_PINNING", False)
    assert {a.text for a in ledger.extract_identifiers(PHASE6) if a.status == "active"} >= {"HMAC-SHA256", "MAX_RETRY_ATTEMPTS"}
