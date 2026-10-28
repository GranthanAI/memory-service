"""The evaluation harness is deterministic, offline, and shows what the ledger and the validation buy."""
import pytest

from evaluation.run import format_report, main
from evaluation.runner import run_all, run_supersede
from evaluation.scenarios import build_conversation

pytestmark = pytest.mark.asyncio


def test_scenarios_plant_every_kind_and_are_reproducible():
    a, b = build_conversation(3, 120), build_conversation(3, 120)
    assert [(p.kind, p.token, p.index) for p in a.planted] == [(p.kind, p.token, p.index) for p in b.planted]
    assert {p.kind for p in a.planted} == {"identifier", "constraint", "decision", "open_item"}
    texts = " ".join(m["content"] for m in a.messages)
    assert all(p.token in texts or p.stale_token in texts for p in a.planted)
    assert [m["role"] for m in a.messages[:4]] == ["user", "assistant", "user", "assistant"]


async def test_ledger_run_beats_the_baseline_and_never_leaks_across_branches():
    report = await run_all(seed=5, cycles=6, length=80, conversations=2)
    with_ledger, without = report["with_ledger"], report["without_ledger"]
    assert with_ledger["mean_recall"] >= 0.95 and with_ledger["mean_recall"] > without["mean_recall"] + 0.3
    assert with_ledger["recall"]["identifier"] == 1.0 and without["recall"]["identifier"] < 0.5
    assert with_ledger["branch_leaked_facts"] == 0 and without["branch_leaked_facts"] == 0
    assert with_ledger["validation_retries"] > 0                                  # the invented numbers were caught and retried
    assert with_ledger["rejected_batches"] == 0


async def test_runs_are_deterministic():
    assert await run_all(seed=9, cycles=4, length=60, conversations=1) == await run_all(seed=9, cycles=4, length=60, conversations=1)


async def test_supersede_precision_with_a_perfect_and_a_noisy_judge():
    perfect = await run_supersede(1, cases=30, judge_error_rate=0.0)
    assert perfect["precision"] == 1.0 and perfect["recall"] == 1.0
    noisy = await run_supersede(1, cases=60, judge_error_rate=0.5)
    assert noisy["recall"] < 1.0 or noisy["precision"] < 1.0                          # mistakes are visible in the metrics


def test_cli_prints_a_report_and_refuses_the_paid_model_mode(capsys, monkeypatch):
    monkeypatch.delenv("EVAL_ALLOW_REAL_LLM", raising=False)
    assert main(["--seed", "2", "--cycles", "3", "--length", "40", "--conversations", "1"]) == 0
    out = capsys.readouterr().out
    assert "with_ledger" in out and "without_ledger" in out and "supersede_perfect_judge" in out
    assert main(["--real"]) == 2                                                      # no accidental paid calls
    assert "EVAL_ALLOW_REAL_LLM" in capsys.readouterr().err
