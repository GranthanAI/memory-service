"""CLI: `python -m evaluation.run [--seed N] [--cycles N] [--length N] [--conversations N] [--json]`.

Runs fully offline with deterministic fakes (what CI uses). `--real` is a MANUAL mode that would call the configured paid model;
it refuses to run unless EVAL_ALLOW_REAL_LLM=1 is set, and is not wired into any test or make target."""
import argparse
import asyncio
import json
import os
import sys

from evaluation.runner import run_all


def format_report(r: dict) -> str:
    lines = [f"Memory evaluation (seed={r['seed']}, cycles={r['cycles']}, messages/conversation={r['length']}, "
             f"conversations={r['conversations']})", ""]
    for label in ("without_ledger", "with_ledger"):
        s = r[label]
        lines += [f"[{label}] mean planted-fact recall {s['mean_recall']:.3f}  " +
                  "  ".join(f"{k}={v:.2f}" for k, v in s["recall"].items()),
                  f"    branch: leaked facts={s['branch_leaked_facts']}  recall before branch={s['branch_recall_before_branch']:.2f}"
                  f"   validation retries={s['validation_retries']}  rejected batches={s['rejected_batches']}"
                  f"   resolved-but-open={s['stale_open_items']}"]
    for label in ("supersede_perfect_judge", "supersede_noisy_judge"):
        s = r[label]
        lines.append(f"[{label}] precision={s['precision']:.2f} recall={s['recall']:.2f} (retired {s['retired']}/{s['cases']}, "
                     f"judge error rate {s['judge_error_rate']})")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--length", type=int, default=120)
    ap.add_argument("--conversations", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--real", action="store_true", help="MANUAL: use the real model (paid). Needs EVAL_ALLOW_REAL_LLM=1.")
    args = ap.parse_args(argv)
    if args.real:
        if os.environ.get("EVAL_ALLOW_REAL_LLM") != "1":
            print("Refusing: --real calls the configured paid model. Set EVAL_ALLOW_REAL_LLM=1 to confirm.", file=sys.stderr)
            return 2
        print("--real is not implemented in this build; use the offline mode.", file=sys.stderr)
        return 2
    report = asyncio.run(run_all(args.seed, args.cycles, args.length, args.conversations))
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
