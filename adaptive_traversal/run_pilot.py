"""Pilot of SLM-controlled traversal (k=3, m=20, B=4) on ~24 document-gold questions. NOT the full benchmark.

Stratified sample (gold used ONLY here, for selection, and in the post-hoc evaluation, never inside traverse()),
seed 0, 8 per stratum, defined on the first-step candidates of the accepted k=3 pre-filter run:
  reachable_top20     some document gold is a 1-hop neighbor of the seeds AND in the first-step top-20
  reachable_filtered  some document gold is a 1-hop neighbor but none is in the first-step top-20
  unreachable_1hop    no document gold among the seeds' 1-hop neighbors

Usage: python -P -s run_pilot.py <out_dir>   (stderr: SAGE progress bars)
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import psutil

PROJECT = Path(__file__).resolve().parent.parent
for p in (Path(__file__).resolve().parent, PROJECT / "slm_controller", PROJECT / "traversal_adapter",
          PROJECT / "candidate_prefilter"):
    sys.path.insert(0, str(p))
from controller import ControllerConfig, TraversalController  # noqa: E402
from graph_seed_adapter import GraphSeedAdapter  # noqa: E402
from sage_prefilter import SagePrefilter  # noqa: E402
from traversal import TraversalConfig, traverse  # noqa: E402

PER_STRATUM = 8
RANKINGS = PROJECT / "candidate_prefilter" / "runs" / "eval_cachefree_runA" / "rankings.jsonl"


def select(A, cfg):
    strata = {"reachable_top20": [], "reachable_filtered": [], "unreachable_1hop": []}
    with open(RANKINGS) as f:
        for line in f:
            o = json.loads(line)
            if o["k"] != cfg.k or A.questions[o["question_id"]]["eval_group"] != "doc":
                continue
            gold = set(A.questions[o["question_id"]]["doc_gold"])
            all_c = {c for c, _ in o["ranked"]}
            top = {c for c, _ in o["ranked"][:cfg.m]}
            key = ("reachable_top20" if gold & top else "reachable_filtered") if gold & all_c else "unreachable_1hop"
            strata[key].append(o["question_id"])
    rng = random.Random(0)
    return [(s, q) for s, qs in strata.items() for q in rng.sample(sorted(qs), PER_STRATUM)]


def main():
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=False)
    proc = psutil.Process()
    cfg = TraversalConfig(k=3, m=20, budget=4, max_retries=1)
    A = GraphSeedAdapter()
    P = SagePrefilter(A, use_cache=False)
    from mlx_backend import MLXBackend
    backend = MLXBackend()
    C = TraversalController(backend, ControllerConfig())
    sample = select(A, cfg)

    results, peak, t0 = [], 0, time.time()
    with open(out / "steps.jsonl", "w") as fs, open(out / "results.jsonl", "w") as fr:
        for stratum, q in sample:
            r = traverse(q, A, P, C, cfg)                      # gold-free
            # ---- post-hoc evaluation (gold) ----
            gold = set(A.questions[q]["doc_gold"])
            explored = r["retrieved_nodes"][len(r["seed_nodes"]):]
            r["eval"] = {"stratum": stratum, "n_gold": len(gold),
                         "seed_recall": len(gold & set(r["seed_nodes"])) / len(gold),
                         "final_recall": len(gold & set(r["retrieved_nodes"])) / len(gold),
                         "gold_discovered": sorted(gold & set(explored)),
                         "explored_is_gold": [c in gold for c in explored],
                         "gold_in_candidates_per_step": [sum(c["chunk_id"] in gold for c in s.get("top_m", []))
                                                         for s in r["steps"] if "top_m" in s]}
            for s in r["steps"]:
                fs.write(json.dumps({"query_id": q, **s}) + "\n")
            fr.write(json.dumps({k: v for k, v in r.items() if k != "steps"}) + "\n")
            results.append(r)
            peak = max(peak, proc.memory_info().rss)
            print(f"{stratum:18s} {q} {r['termination_reason']:16s} explore={r['explore_steps']} calls={r['slm_calls']} "
                  f"retries={r['retries']} recall {r['eval']['seed_recall']:.2f}->{r['eval']['final_recall']:.2f} "
                  f"{r['total_latency_ms'] / 1000:.1f}s", flush=True)

    n = len(results)
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    by = lambda key: {s: [r for r in results if r["eval"]["stratum"] == s] for s in dict.fromkeys(x for x, _ in sample)}
    term = Counter(r["termination_reason"] for r in results)
    summary = {
        "config": {"k": cfg.k, "m": cfg.m, "budget": cfg.budget, "max_retries": cfg.max_retries,
                   "prompt_version": C.cfg.prompt_version, "dedup_text": cfg.dedup_text,
                   "model": f"{backend.repo}@{backend.revision[:7]} (4-bit MLX, greedy)",
                   "prefilter": "SAGE _calculate_hybrid_similarities, cache-free", "questions": n,
                   "strata": dict(Counter(s for s, _ in sample))},
        "successful_traversal_rate": round(sum(r["termination_reason"] != "CONTROLLER_ERROR" for r in results) / n, 3),
        "termination_reasons": dict(term),
        "avg_explore_steps": mean([r["explore_steps"] for r in results]),
        "early_stop_rate (SLM_STOP with budget left)": round(sum(r["termination_reason"] == "SLM_STOP" for r in results) / n, 3),
        "controller_error_rate (queries)": round(term.get("CONTROLLER_ERROR", 0) / n, 3),
        "invalid_output_rate (calls)": round(sum(r["invalid_outputs"] for r in results) / sum(r["slm_calls"] for r in results), 3),
        "retries_total": sum(r["retries"] for r in results),
        "retry_recovered": sum(r["retries"] > 0 and r["termination_reason"] != "CONTROLLER_ERROR" for r in results),
        "per_query_avg": {"slm_calls": mean([r["slm_calls"] for r in results]),
                          "input_tokens": mean([r["input_tokens"] for r in results]),
                          "output_tokens": mean([r["output_tokens"] for r in results]),
                          "slm_latency_s": mean([r["slm_latency_ms"] / 1000 for r in results]),
                          "prefilter_latency_s": mean([r["prefilter_latency_ms"] / 1000 for r in results]),
                          "total_latency_s": mean([r["total_latency_ms"] / 1000 for r in results])},
        "per_call_avg": {"input_tokens": round(sum(r["input_tokens"] for r in results) / sum(r["slm_calls"] for r in results), 1),
                         "latency_s": round(sum(r["slm_latency_ms"] for r in results) / sum(r["slm_calls"] for r in results) / 1000, 2)},
        "recall": {"seed_only": mean([r["eval"]["seed_recall"] for r in results]),
                   "final": mean([r["eval"]["final_recall"] for r in results]),
                   "by_stratum": {s: {"n": len(rs), "seed_only": mean([r["eval"]["seed_recall"] for r in rs]),
                                      "final": mean([r["eval"]["final_recall"] for r in rs]),
                                      "termination": dict(Counter(r["termination_reason"] for r in rs))}
                                  for s, rs in by(None).items()}},
        "gold_discovery": {"queries_with_new_gold": sum(bool(r["eval"]["gold_discovered"]) for r in results),
                           "new_gold_items": sum(len(r["eval"]["gold_discovered"]) for r in results),
                           "explored_nodes_total": sum(r["explore_steps"] for r in results),
                           "explored_nodes_that_are_gold": sum(sum(r["eval"]["explored_is_gold"]) for r in results)},
        "selected_prefilter_rank_distribution": dict(sorted(Counter(s["selected_prefilter_rank"] for r in results
                                                                    for s in r["steps"] if s.get("selected_action") == "EXPLORE").items())),
        "peak_rss_gb": round(peak / 1e9, 2), "mlx_peak_memory_gb": round(backend.mx.get_peak_memory() / 1e9, 2),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
