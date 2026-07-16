"""
Compare two retrieval-eval runs.
=================================
The experiment loop this enables:

    1. run eval  ->  baseline saved to eval/runs/
    2. change ONE knob (chunk size, top_k, model, ...)
    3. re-ingest if the change affects chunking/embeddings
    4. run eval again
    5. python eval/compare.py       <- shows exactly what the change bought

Usage (from the repo root):
    python eval/compare.py                       # two most recent runs
    python eval/compare.py old.json new.json     # explicit files

Generation-eval files (gen_*.json) are ignored when auto-picking.
"""
import os
import sys
import json
import glob

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(EVAL_DIR, "runs")

STAGES = ["dense", "bm25", "fused", "rerank"]


def load_run(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_latest_two() -> tuple[str, str]:
    """Two most recent retrieval runs (timestamped names sort chronologically)."""
    runs = sorted(
        p for p in glob.glob(os.path.join(RUNS_DIR, "*.json"))
        if not os.path.basename(p).startswith("gen_")
    )
    if len(runs) < 2:
        print(f"[ERROR] Need at least 2 retrieval runs in {RUNS_DIR}, found {len(runs)}.")
        sys.exit(1)
    return runs[-2], runs[-1]


def print_config_diff(old: dict, new: dict):
    old_cfg, new_cfg = old["config"], new["config"]
    keys = sorted(set(old_cfg) | set(new_cfg))
    changed = [k for k in keys if old_cfg.get(k) != new_cfg.get(k)]
    print("CONFIG")
    if not changed:
        print("  identical between the two runs")
        return
    for k in changed:
        print(f"  {k}: {old_cfg.get(k)}  ->  {new_cfg.get(k)}")


def print_aggregate_diff(old: dict, new: dict):
    for label in ["overall", "keyword", "paraphrase"]:
        o, n = old["aggregate"][label], new["aggregate"][label]
        print(f"\n{label.upper()}")
        print(f"  {'stage':<10}{'hit-rate':>18}{'MRR':>18}")
        for stage in STAGES:
            oh, nh = o[stage]["hit_rate"], n[stage]["hit_rate"]
            om, nm = o[stage]["mrr"], n[stage]["mrr"]
            print(f"  {stage:<10}"
                  f"{oh:>7.2f} -> {nh:<5.2f}{_delta(nh - oh):>4}"
                  f"{om:>9.2f} -> {nm:<5.2f}{_delta(nm - om):>4}")


def _delta(d: float) -> str:
    if abs(d) < 0.005:
        return "  ="
    return f"{d:+.2f}"


def print_question_diff(old: dict, new: dict):
    """Per-question changes at the rerank stage — the chunks the LLM actually sees."""
    old_q = {r["id"]: r for r in old["per_question"] if r["type"] != "negative"}
    new_q = {r["id"]: r for r in new["per_question"] if r["type"] != "negative"}

    regressions, improvements, rr_changes = [], [], []
    for qid in sorted(set(old_q) & set(new_q)):
        o_hit = old_q[qid]["scores"]["rerank"]["hit"]
        n_hit = new_q[qid]["scores"]["rerank"]["hit"]
        o_rr  = old_q[qid]["scores"]["rerank"]["rr"]
        n_rr  = new_q[qid]["scores"]["rerank"]["rr"]
        if o_hit and not n_hit:
            regressions.append(qid)
        elif n_hit and not o_hit:
            improvements.append(qid)
        elif abs(n_rr - o_rr) > 0.005:
            rr_changes.append((qid, o_rr, n_rr))

    print("\nPER-QUESTION (rerank stage)")
    if not (regressions or improvements or rr_changes):
        print("  no changes")
    for qid in regressions:
        print(f"  REGRESSED  {qid}: {new_q[qid]['question']}")
        print(f"             top chunk now from: {new_q[qid]['top_rerank_source']}")
    for qid in improvements:
        print(f"  IMPROVED   {qid}: {new_q[qid]['question']}")
    for qid, o_rr, n_rr in rr_changes:
        print(f"  RANK MOVED {qid}: rr {o_rr:.2f} -> {n_rr:.2f}  ({new_q[qid]['question'][:60]})")

    added   = set(new_q) - set(old_q)
    removed = set(old_q) - set(new_q)
    if added:
        print(f"  questions only in new run: {', '.join(sorted(added))}")
    if removed:
        print(f"  questions only in old run: {', '.join(sorted(removed))}")


def main():
    if len(sys.argv) == 3:
        old_path, new_path = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        old_path, new_path = pick_latest_two()
    else:
        print(__doc__)
        sys.exit(1)

    old, new = load_run(old_path), load_run(new_path)
    print("=" * 62)
    print(f"OLD: {os.path.basename(old_path)}")
    print(f"NEW: {os.path.basename(new_path)}")
    print("=" * 62)
    print_config_diff(old, new)
    print_aggregate_diff(old, new)
    print_question_diff(old, new)


if __name__ == "__main__":
    main()
