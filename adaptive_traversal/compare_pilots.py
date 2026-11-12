"""Side-by-side comparison of two pilot runs with identical metric definitions (same 24 questions).

duplicate-text exploration = an explored node whose normalized text (lowercased, whitespace-collapsed) equals
the text of a passage already in the evidence when it was selected.

Usage: python -P -s compare_pilots.py <run_dir_before> <run_dir_after> [out.json]
"""
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "traversal_adapter"))
from graph_seed_adapter import GraphSeedAdapter  # noqa: E402


def load(d):
    d = Path(d)
    R = [json.loads(l) for l in open(d / "results.jsonl")]
    S = [json.loads(l) for l in open(d / "steps.jsonl")]
    return R, S, json.loads((d / "summary.json").read_text())


def metrics(R, S, summ, A):
    norm = lambda c: " ".join(A.node(c).content.lower().split())
    calls = [c for s in S for c in s.get("calls", [])]
    first = [c for c in calls if c["attempt"] == 1]
    dup = 0
    for r in R:
        seen = set()
        for i, c in enumerate(r["retrieved_nodes"]):
            if i >= len(r["seed_nodes"]) and norm(c) in seen:
                dup += 1
            seen.add(norm(c))
    n = len(R); mean = lambda xs: round(sum(xs) / len(xs), 3)
    term = Counter(r["termination_reason"] for r in R)
    return {
        "questions": n,
        "invalid_first_attempts": f"{sum(not c['valid'] for c in first)}/{len(first)}",
        "invalid_first_attempt_errors": dict(Counter(c["error"] for c in first if not c["valid"])),
        "retries": sum(r["retries"] for r in R),
        "retry_recovered_calls": sum(c["valid"] for c in calls if c["attempt"] == 2),
        "controller_errors": term.get("CONTROLLER_ERROR", 0),
        "termination": dict(term),
        "stop_rate (SLM_STOP)": round(term.get("SLM_STOP", 0) / n, 3),
        "avg_explore_steps": mean([r["explore_steps"] for r in R]),
        "explored_nodes": sum(r["explore_steps"] for r in R),
        "duplicate_text_explorations": dup,
        "avg_slm_calls": mean([r["slm_calls"] for r in R]),
        "avg_input_tokens": mean([r["input_tokens"] for r in R]),
        "avg_output_tokens": mean([r["output_tokens"] for r in R]),
        "avg_slm_latency_s": mean([r["slm_latency_ms"] / 1000 for r in R]),
        "avg_total_latency_s": mean([r["total_latency_ms"] / 1000 for r in R]),
        "recall_seed": mean([r["eval"]["seed_recall"] for r in R]),
        "recall_final": mean([r["eval"]["final_recall"] for r in R]),
        "recall_by_stratum": {s: {"seed": v["seed_only"], "final": v["final"]}
                              for s, v in summ["recall"]["by_stratum"].items()},
        "queries_with_new_gold": sum(bool(r["eval"]["gold_discovered"]) for r in R),
        "new_gold_items": sum(len(r["eval"]["gold_discovered"]) for r in R),
        "explored_gold": sum(sum(r["eval"]["explored_is_gold"]) for r in R),
    }


def main():
    A = GraphSeedAdapter()
    (Ra, Sa, sa), (Rb, Sb, sb) = load(sys.argv[1]), load(sys.argv[2])
    assert [r["query_id"] for r in Ra] == [r["query_id"] for r in Rb], "runs must cover the same questions in order"
    out = {"before": {"run": sys.argv[1], **{k: sa["config"].get(k) for k in ("prompt_version", "dedup_text")},
                      **metrics(Ra, Sa, sa, A)},
           "after": {"run": sys.argv[2], **{k: sb["config"].get(k) for k in ("prompt_version", "dedup_text")},
                     **metrics(Rb, Sb, sb, A)},
           "per_question": [{"query_id": a["query_id"], "stratum": a["eval"]["stratum"],
                             "before": [a["termination_reason"], a["explore_steps"], a["eval"]["final_recall"]],
                             "after": [b["termination_reason"], b["explore_steps"], b["eval"]["final_recall"]]}
                            for a, b in zip(Ra, Rb)]}
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_question"}, indent=2))
    for p in out["per_question"]:
        print(f"{p['query_id']} {p['stratum']:18s} {p['before']} -> {p['after']}")


if __name__ == "__main__":
    main()
