"""GetMemoryContext honours max_tokens and reports the real total."""
from app.proto.memory_handlers import fit_context_to_budget


def ctx(n_recent=10, n_retrieved=3, words=60):
    return {
        "current_summary": "summary " * 20,
        "ancestor_summaries": [{"summary": "ancestor " * 10}],
        "relevant_facts": [{"statement": "likes tea"}],
        "retrieved_history": [{"content": f"older {i} " + "w " * words} for i in range(n_retrieved)],
        "short_term_messages": [{"content": f"recent {i} " + "w " * words} for i in range(n_recent)],
        "metadata": {},
    }


def test_no_budget_drops_nothing_and_reports_the_real_total():
    result = ctx()
    total = fit_context_to_budget(result, 0)
    assert total > 500 and len(result["short_term_messages"]) == 10 and len(result["retrieved_history"]) == 3


def test_budget_drops_retrieved_history_first_then_the_oldest_recent_messages():
    result = ctx()
    full = fit_context_to_budget(ctx(), 0)
    total = fit_context_to_budget(result, full - 150)
    assert total <= full - 150
    assert len(result["retrieved_history"]) < 3                            # verbatim extras give way first
    assert [m["content"].split()[1] for m in result["short_term_messages"]][-1] == "9"   # newest message kept
    assert result["metadata"]["budget_trimmed"] is True


def test_tight_budget_keeps_summary_facts_and_the_newest_message():
    result = ctx()
    total = fit_context_to_budget(result, 100)
    assert result["current_summary"] and result["relevant_facts"] and result["ancestor_summaries"]
    assert len(result["short_term_messages"]) == 1 and result["retrieved_history"] == []
    assert total > 100                                                     # compact layers are never cut: report honestly


def test_oldest_recent_messages_go_before_newer_ones():
    result = ctx(n_retrieved=0)
    fit_context_to_budget(result, 400)
    kept = [int(m["content"].split()[1]) for m in result["short_term_messages"]]
    assert kept == sorted(kept) and kept[-1] == 9 and kept[0] > 0
