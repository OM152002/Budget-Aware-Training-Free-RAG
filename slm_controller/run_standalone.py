"""Standalone controller test: ONE decision per question (no iterative traversal).

Inputs per question (k=3, m=20): evidence = top-3 seeds (adapter), candidates = top-20 pre-filtered neighbors taken
from the accepted pre-filter run (candidate_prefilter/runs/eval_cachefree_runA/rankings.jsonl, k=3), in pre-filter
order; history = [] (first decision); remaining budget = 4.

Sample (gold labels used ONLY for stratification and post-hoc analysis, never in the prompt), seed 0, 6 per stratum:
  gold_in_cands_small   gold among presented candidates, < 20 candidates
  gold_in_cands_full    gold among presented candidates, 20 candidates
  no_gold_small         no gold among presented candidates, < 20 candidates
  no_gold_full          no gold among presented candidates, 20 candidates (incl. gold-reachable-but-filtered)
A determinism re-run repeats the first 6 cases.

Usage: python -P -s run_standalone.py <out_dir>
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import psutil

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT / "traversal_adapter"))
from controller import ControllerConfig, Passage, TraversalController  # noqa: E402
from graph_seed_adapter import GraphSeedAdapter  # noqa: E402

K, M, BUDGET, PER_STRATUM = 3, 20, 4, 6
RANKINGS = PROJECT / "candidate_prefilter" / "runs" / "eval_cachefree_runA" / "rankings.jsonl"


def main():
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=False)
    proc = psutil.Process()
    A = GraphSeedAdapter()
    rows = {}
    with open(RANKINGS) as f:
        for line in f:
            o = json.loads(line)
            if o["k"] == K:
                rows[o["question_id"]] = o
    strata = {"gold_in_cands_small": [], "gold_in_cands_full": [], "no_gold_small": [], "no_gold_full": []}
    for q, o in rows.items():
        cands = [c for c, _ in o["ranked"][:M]]
        if not cands:
            continue
        assert o["seeds"] == A.questions[q]["ranking"][:K]
        gold = set(A.questions[q]["doc_gold"])
        g = "gold_in_cands" if gold & set(cands) else "no_gold"
        strata[f"{g}_{'full' if len(cands) == M else 'small'}"].append(q)
    rng = random.Random(0)
    sample = [(s, q) for s, qs in strata.items() for q in rng.sample(sorted(qs), PER_STRATUM)]

    from mlx_backend import MLXBackend
    rss0 = proc.memory_info().rss
    backend = MLXBackend()
    rss_model = proc.memory_info().rss
    cfg = ControllerConfig()
    ctl = TraversalController(backend, cfg)

    def run_case(stratum, q):
        o = rows[q]; Q = A.questions[q]
        ev = [Passage(c, A.node(c).content) for c in o["seeds"]]
        ranked = o["ranked"][:M]
        ca = [Passage(c, A.node(c).content) for c, _ in ranked]
        d = ctl.choose_action(Q["question"], ev, ca, [], BUDGET)
        gold = set(Q["doc_gold"]); cset = [c for c, _ in ranked]
        return {"stratum": stratum, "question_id": q, "n_candidates": len(ca),
                "gold_candidates_ranks": [i + 1 for i, c in enumerate(cset) if c in gold],
                "selected_rank": (cset.index(d.node_id) + 1) if d.node_id else None,
                "selected_is_gold": (d.node_id in gold) if d.node_id else None,
                "selected_in_candidate_set": (d.node_id in cset) if d.node_id else None,
                **d.to_dict()}

    results, peak_rss, t0 = [], rss_model, time.time()
    for s, q in sample:
        r = run_case(s, q); results.append(r)
        peak_rss = max(peak_rss, proc.memory_info().rss)
        print(f"{s:20s} {q} n={r['n_candidates']:2d} -> {r['raw_output']!r:45s} valid={r['valid']} "
              f"in={r['prompt_tokens']} out={r['output_tokens']} {r['latency_ms']:.0f}ms", flush=True)
    rerun = [run_case(s, q) for s, q in sample[:6]]
    det = all(a["raw_output"] == b["raw_output"] for a, b in zip(results[:6], rerun))

    n = len(results)
    ex = [r for r in results if r["effective_action"] == "EXPLORE"]
    with_gold = [r for r in results if r["gold_candidates_ranks"]]
    ex_gold_avail = [r for r in with_gold if r["action"] == "EXPLORE"]
    lat = sorted(r["latency_ms"] for r in results)
    summary = {
        "model": {"repo": backend.repo, "revision": backend.revision, "quantization": "4-bit, group_size 64 (MLX)",
                  "backend": "mlx-lm 0.32.2 / mlx 0.32.4", "load_seconds": round(backend.load_seconds, 1),
                  "decoding": f"greedy (temperature {cfg.temperature}), max_new_tokens {cfg.max_new_tokens}"},
        "setup": {"k": K, "m": M, "remaining_budget": BUDGET, "history": "empty (first decision)",
                  "max_passage_chars": cfg.max_passage_chars, "questions": n,
                  "strata": dict(Counter(r["stratum"] for r in results))},
        "memory": {"process_rss_before_model_gb": round(rss0 / 1e9, 2), "process_rss_after_load_gb": round(rss_model / 1e9, 2),
                   "process_rss_peak_gb": round(peak_rss / 1e9, 2),
                   "mlx_peak_memory_gb": round(max(r["backend_stats"]["mlx_peak_memory_gb"] for r in results), 2)},
        "tokens": {"prompt_mean": round(sum(r["prompt_tokens"] for r in results) / n, 1),
                   "prompt_min": min(r["prompt_tokens"] for r in results), "prompt_max": max(r["prompt_tokens"] for r in results),
                   "output_mean": round(sum(r["output_tokens"] for r in results) / n, 1),
                   "output_max": max(r["output_tokens"] for r in results)},
        "latency_ms": {"mean": round(sum(lat) / n), "median": round(lat[n // 2]), "max": round(lat[-1]),
                       "prompt_tps_mean": round(sum(r["backend_stats"]["prompt_tps"] for r in results) / n, 1),
                       "generation_tps_mean": round(sum(r["backend_stats"]["generation_tps"] for r in results) / n, 1)},
        "validity": {"valid_rate": round(sum(r["valid"] for r in results) / n, 3),
                     "errors": dict(Counter(r["error"] for r in results if not r["valid"])),
                     "format_flags": dict(Counter(f for r in results for f in r["flags"]))},
        "actions": dict(Counter(r["action"] for r in results)),
        "actions_by_stratum": {s: dict(Counter(r["action"] for r in results if r["stratum"] == s)) for s in strata},
        "explore_selected_in_candidate_set": f"{sum(bool(r['selected_in_candidate_set']) for r in ex)}/{len(ex)}",
        "selected_rank_distribution": dict(sorted(Counter(r["selected_rank"] for r in ex).items())),
        "ANALYSIS_ONLY_gold": {
            "questions_with_gold_in_candidates": len(with_gold),
            "of_those_explore": len(ex_gold_avail),
            "explore_selects_gold": sum(r["selected_is_gold"] for r in ex_gold_avail),
            "chance_rate_mean_gold_fraction": round(sum(len(r["gold_candidates_ranks"]) / r["n_candidates"] for r in ex_gold_avail) / max(1, len(ex_gold_avail)), 3),
            "prefilter_rank1_is_gold": sum(1 in r["gold_candidates_ranks"] for r in with_gold),
            "stop_when_gold_available": sum(r["action"] == "STOP" for r in with_gold),
        },
        "determinism_rerun_first6_identical": det,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
