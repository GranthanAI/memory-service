"""Runs the pipeline over synthetic conversations and measures: recall of planted facts, leakage across branches, and
fact-supersede precision. Deterministic for a given seed."""
import random
import uuid
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import AsyncMock

from app.core.config import settings
from app.services import anchors as ledger
from app.services.context_builder import ContextBuilder
from app.services.long_memory_service import LongMemoryService
from app.services.summary_service import SummaryService, SummaryValidationError
from evaluation.fakes import LossySummarizer, RuleExtractor, SummarizerWithExtractor
from evaluation.scenarios import Conversation, build_conversation
from tests.support.fakes import FakeCassandra

USER = "eval-user"


class _Redis:
    async def get_snapshot(self, cid): return None
    async def set_snapshot(self, s): pass
    async def invalidate_conversation(self, cid): pass


def _final_context(cass: FakeCassandra, cid: str, with_ledger: bool) -> str:
    """What the model would be shown for this conversation: the summary text plus the rendered anchor ledger."""
    summary = (cass.get_summary(cid) or {}).get("summary_text", "")
    if not with_ledger:
        return summary
    shown = ledger.render_order(cass.list_anchors(cid), settings.ANCHOR_RENDER_MAX_CHARS)
    return summary + "\n" + "\n".join(f"[{a['kind']}] {a['text']}" for a in shown)


async def run_conversation(seed: int, cycles: int, length: int, with_ledger: bool) -> Dict:
    conv: Conversation = build_conversation(seed, length)
    cass = FakeCassandra()
    cass.messages["A"] = []
    cass.snapshots["A"] = {"conversation_id": "A", "user_id": USER, "summary_version": 0}
    model = LossySummarizer(seed)
    llm = SummarizerWithExtractor(model, RuleExtractor() if with_ledger else None)
    repo = SimpleNamespace(update_snapshot_summary_fields=AsyncMock(), invalidate_conversation=AsyncMock(),
                           get_snapshot=AsyncMock(side_effect=lambda cid: cass.get_snapshot(cid)))
    service = SummaryService(repo, cass, llm)
    saved = settings.ANCHOR_LEDGER_ENABLED
    settings.ANCHOR_LEDGER_ENABLED = with_ledger
    rejected = 0
    try:
        step = max(1, length // cycles)
        for start in range(0, length, step):
            for m in conv.messages[start:start + step]:
                cass.append_recent_message("A", m)
            try:
                await service.process_incremental_summary("A")
            except SummaryValidationError:
                rejected += 1                                              # the batch stays unsummarized; retried next cycle
        try:
            await service.process_incremental_summary("A")                 # catch up whatever a rejection left behind
        except SummaryValidationError:
            rejected += 1
        context = _final_context(cass, "A", with_ledger)

        kinds: Dict[str, List[bool]] = {"identifier": [], "constraint": [], "decision": [], "open_item": []}
        for p in conv.planted:
            if p.kind == "open_item" and p.resolved:
                continue                                                   # finished items are not expected to be shown
            kinds[p.kind].append(p.token in context)
        recall = {k: (sum(v) / len(v) if v else 1.0) for k, v in kinds.items()}
        stale_open = sum(1 for p in conv.planted if p.kind == "open_item" and p.resolved
                         and any(p.token in a["text"] and a["status"] == "open" for a in cass.list_anchors("A")))

        # branch: a child made halfway through; it may only see what existed at the branch point
        branch_index = (length // 2) // 2 * 2 + 1
        upto = conv.messages[branch_index]["message_id"]
        retrieval = SimpleNamespace(memory_repo=SimpleNamespace(
            cassandra_repo=cass, get_snapshot=AsyncMock(side_effect=lambda cid: cass.get_snapshot(cid))))
        inherited = await ContextBuilder(retrieval, graph_client=None).ancestor_summaries(
            USER, [{"conversation_id": "A", "upto_message_id": upto}])
        inherited_text = "\n".join(a["summary"] for a in inherited)
        later = [p for p in conv.planted if p.index > branch_index]
        leaked = sum(1 for p in later if p.token in inherited_text or (p.stale_token and p.stale_token in inherited_text))
        before = [p for p in conv.planted if p.index <= branch_index and p.kind in ("identifier", "constraint", "open_item")]
        branch_recall = sum(1 for p in before if p.token in inherited_text) / len(before) if before else 1.0
        return {"seed": seed, "recall": recall, "rejected_batches": rejected, "validation_retries": model.retries,
                "stale_open_items": stale_open, "branch": {"leaked": leaked, "later_facts": len(later),
                                                           "recall_before_branch": round(branch_recall, 3),
                                                           "inherited_summary_found": bool(inherited)}}
    finally:
        settings.ANCHOR_LEDGER_ENABLED = saved


# ─── fact supersede ─────────────────────────────────────────────────────────────────────────────────────────────────────


class _Facts:
    def __init__(self):
        self.rows, self.vectors = {}, {}

    def get_facts(self, u, c, include_inactive=False):
        return [dict(r) for (uu, cc, _), r in self.rows.items()
                if uu == u and cc == c and (include_inactive or (r.get("status") or "ACTIVE") == "ACTIVE")]

    def upsert_fact(self, f): self.rows[(f["user_id"], f["category"], f["fact_id"])] = {**f, "status": "ACTIVE"}
    def retire_fact(self, u, c, f, by): self.rows[(u, c, f)].update(status="SUPERSEDED", superseded_by=by)
    def touch_fact_confirmed(self, u, c, f): pass
    def get_memory_enabled(self, u): return None
    def get_fact_tombstones(self, u): return set()
    def insert_facts(self, records):
        for r in records:
            self.vectors[r["fact_id"]] = dict(r)

    def search_facts(self, user_id, query_vector, limit=5, **_):
        hits = [dict(r, distance=sum(a * b for a, b in zip(query_vector, r["vector"]))) for r in self.vectors.values()]
        return sorted(hits, key=lambda h: h["distance"], reverse=True)[:limit]

    def list_user_facts(self, user_id, limit=500): return [{k: v for k, v in r.items() if k != "vector"} for r in self.vectors.values()]
    def delete_fact(self, *a):
        if len(a) == 3:
            self.rows.pop(a, None)
        else:
            self.vectors.pop(str(a[1]), None)


def _unit(angle: float):
    import math
    return [math.cos(angle), math.sin(angle)]


async def run_supersede(seed: int, cases: int = 30, judge_error_rate: float = 0.0) -> Dict:
    """Corrections must retire the old fact; restatements and merely related facts must not. The 'judge' is an oracle that is
    wrong with probability `judge_error_rate` (0 = perfect), to show what a mistaken judgement costs in precision."""
    rng = random.Random(seed)
    expected: Dict[str, str] = {}

    async def judge(new, existing):
        truth = expected[new]
        if rng.random() < judge_error_rate:
            truth = rng.choice(["same", "correction", "different"])
        return {"relation": truth, "index": 0 if truth != "different" else None}

    retired = correct_retired = corrections = corrections_retired = 0
    for i in range(cases):
        facts = _Facts()
        service = LongMemoryService(facts, facts, judge=judge)
        kind = ("correction", "same", "different")[i % 3]
        old, new = f"User lives in City{i}", {"correction": f"User now lives in Town{i}", "same": f"User lives in City{i}.",
                                              "different": f"User works at Firm{i}"}[kind]
        expected[new] = kind
        angle = 0.3 if kind != "different" else 0.5                       # similar enough to be shown to the judge
        await service.merge_user_facts(USER, "c1", [{"statement": old, "category": "personal", "importance": 0.9, "vector": _unit(0.0)}])
        if kind == "same":
            new = f"User is based in City{i}"
            expected[new] = "same"
        await service.merge_user_facts(USER, "c2", [{"statement": new, "category": "personal", "importance": 0.9, "vector": _unit(angle)}])
        was_retired = any(r.get("status") == "SUPERSEDED" for r in facts.rows.values())
        retired += was_retired
        correct_retired += was_retired and kind == "correction"
        corrections += kind == "correction"
        corrections_retired += was_retired and kind == "correction"
    return {"cases": cases, "judge_error_rate": judge_error_rate, "retired": retired,
            "precision": round(correct_retired / retired, 3) if retired else 1.0,
            "recall": round(corrections_retired / corrections, 3) if corrections else 1.0}


async def run_all(seed: int = 7, cycles: int = 6, length: int = 120, conversations: int = 5) -> Dict:
    out: Dict = {"seed": seed, "cycles": cycles, "length": length, "conversations": conversations}
    for label, with_ledger in (("with_ledger", True), ("without_ledger", False)):
        runs = [await run_conversation(seed + i, cycles, length, with_ledger) for i in range(conversations)]
        recall = {k: round(sum(r["recall"][k] for r in runs) / len(runs), 3) for k in runs[0]["recall"]}
        out[label] = {
            "recall": recall,
            "mean_recall": round(sum(recall.values()) / len(recall), 3),
            "rejected_batches": sum(r["rejected_batches"] for r in runs),
            "validation_retries": sum(r["validation_retries"] for r in runs),
            "stale_open_items": sum(r["stale_open_items"] for r in runs),
            "branch_leaked_facts": sum(r["branch"]["leaked"] for r in runs),
            "branch_recall_before_branch": round(sum(r["branch"]["recall_before_branch"] for r in runs) / len(runs), 3),
        }
    out["supersede_perfect_judge"] = await run_supersede(seed, judge_error_rate=0.0)
    out["supersede_noisy_judge"] = await run_supersede(seed, judge_error_rate=0.3)
    return out
