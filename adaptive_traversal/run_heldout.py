"""Held-out pilot: frozen v2 controller + dedup on 50 fresh document-gold questions. Evaluation only; changes nothing.

Sampling (fixed seed 20261028): uniform random over the 1,728 questions that have >=1 document gold, excluding the
24 questions of the earlier pilots. Gold is used for nothing else before retrieval.
Methods evaluated: seeds-only (top-3) and frozen SLM traversal (k=3, m=20, B=4, v2 prompt, dedup on, 1 retry).

Baseline-ready records (gold-free, per question): seeds, the full step-1 deduplicated candidate ranking with SAGE
pre-filter scores and duplicate-member lists, the SLM's budget used, and every traversal step (steps.jsonl).
Post-hoc categories (gold, evaluation only), from the seeds' raw 1-hop neighbors and the step-1 top-m:
  reachable_top20 / reachable_filtered / unreachable_1hop / all_gold_in_seeds.

Usage: python -P -s run_heldout.py <out_dir>   (stderr: SAGE progress bars)
"""
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
for p in (Path(__file__).resolve().parent, PROJECT / "slm_controller", PROJECT / "traversal_adapter",
          PROJECT / "candidate_prefilter"):
    sys.path.insert(0, str(p))
from controller import ControllerConfig, TraversalController  # noqa: E402
from graph_seed_adapter import GraphSeedAdapter  # noqa: E402
from sage_prefilter import SagePrefilter  # noqa: E402
from traversal import TraversalConfig, dedup_candidates, normalize_text, traverse  # noqa: E402

SEED, N = 20261028, 50
PRIOR_PILOT = PROJECT / "adaptive_traversal" / "runs" / "pilot_k3_m20_B4_v2" / "results.jsonl"
FROZEN = ["slm_controller/controller.py", "slm_controller/mlx_backend.py", "adaptive_traversal/traversal.py",
          "candidate_prefilter/sage_prefilter.py", "traversal_adapter/graph_seed_adapter.py"]


def sha(p):
    return hashlib.sha256((PROJECT / p).read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=False)
    tcfg, ccfg = TraversalConfig(), ControllerConfig()
    assert (tcfg.k, tcfg.m, tcfg.budget, tcfg.max_retries, tcfg.dedup_text) == (3, 20, 4, 1, True)
    assert (ccfg.prompt_version, ccfg.temperature, ccfg.max_passage_chars) == ("v2", 0.0, 300)
    A = GraphSeedAdapter()
    prior = {json.loads(l)["query_id"] for l in open(PRIOR_PILOT)}
    pool = sorted(q for q in A.question_ids if A.questions[q]["eval_group"] == "doc" and q not in prior)
    sample = random.Random(SEED).sample(pool, N)
    (out / "questions.json").write_text(json.dumps({"seed": SEED, "pool_size": len(pool), "excluded_prior": sorted(prior),
                                                    "question_ids": sample}, indent=2))
    P = SagePrefilter(A, use_cache=False)
    from mlx_backend import MLXBackend
    backend = MLXBackend()
    C = TraversalController(backend, ccfg)

    results, t0 = [], time.time()
    with open(out / "steps.jsonl", "w") as fs, open(out / "results.jsonl", "w") as fr, \
            open(out / "baseline_inputs.jsonl", "w") as fb:
        for i, q in enumerate(sample):
            r = traverse(q, A, P, C, tcfg)                                   # frozen, gold-free
            # gold-free record for a later equal-budget non-SLM baseline (same seeds, same dedup + pre-filter)
            S = r["seed_nodes"]
            raw = [n.node.chunk_id for n in A.get_neighbors(S)]
            reps, members, dropped = dedup_candidates(raw, S, lambda c: A.node(c).content)
            ranked = P.score(r["query"], reps)
            fb.write(json.dumps({"query_id": q, "query": r["query"], "seeds": S, "slm_budget_used": r["budget_used"],
                                 "budget": tcfg.budget, "step1_raw_neighbor_count": len(raw),
                                 "step1_dropped_same_text_as_evidence": dropped,
                                 "step1_ranked": [{"chunk_id": c, "score": s, "members": members[c]} for c, s in ranked]})
                     + "\n")
            # ---- post-hoc evaluation (gold) ----
            gold = set(A.questions[q]["doc_gold"])
            seeds, final = set(S), set(r["retrieved_nodes"])
            first_step = r["steps"][0]
            step1_consistent = ([c["chunk_id"] for c in first_step.get("top_m", [])] == [c for c, _ in ranked[:tcfg.m]])
            top_members = {m for x in ranked[:tcfg.m] for m in members[x[0]]}
            gold_left = gold - seeds
            if not gold_left:
                cat = "all_gold_in_seeds"
            elif gold_left & top_members:
                cat = "reachable_top20"
            elif gold_left & set(raw):
                cat = "reachable_filtered"
            else:
                cat = "unreachable_1hop"
            explored = r["retrieved_nodes"][len(S):]
            seed_norm = {normalize_text(A.node(c).content) for c in S}
            r["eval"] = {"category": cat, "n_gold": len(gold), "step1_baseline_record_matches_traversal": step1_consistent, "seed_recall": len(gold & seeds) / len(gold),
                         "final_recall": len(gold & final) / len(gold),
                         "gold_discovered": sorted(gold & set(explored)),
                         "explored_is_gold": [c in gold for c in explored],
                         "gold_via_duplicate_member": sorted(g for c in explored for g in
                                                             set(r["explored_text_duplicate_members"].get(c, [])) & gold
                                                             if g != c),
                         "gold_textdup_of_seed": sorted(g for g in gold - seeds
                                                        if normalize_text(A.node(g).content) in seed_norm)}
            for s in r["steps"]:
                fs.write(json.dumps({"query_id": q, **s}) + "\n")
            fr.write(json.dumps({k: v for k, v in r.items() if k != "steps"}) + "\n")
            results.append(r)
            print(f"{i + 1:2d}/{N} {q} {cat:18s} {r['termination_reason']:16s} explore={r['explore_steps']} "
                  f"calls={r['slm_calls']} recall {r['eval']['seed_recall']:.2f}->{r['eval']['final_recall']:.2f} "
                  f"{r['total_latency_ms'] / 1000:.1f}s", flush=True)

    # ---------------- summary ----------------
    n = len(results); mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    calls = [c for r in results for s in r["steps"] for c in s.get("calls", [])]
    first = [c for c in calls if c["attempt"] == 1]
    term = Counter(r["termination_reason"] for r in results)
    dup = 0
    for r in results:
        seen = set()
        for j, c in enumerate(r["retrieved_nodes"]):
            t = normalize_text(A.node(c).content)
            dup += j >= len(r["seed_nodes"]) and t in seen
            seen.add(t)
    gain = [r for r in results if r["eval"]["final_recall"] > r["eval"]["seed_recall"]]
    loss = [r for r in results if r["eval"]["final_recall"] < r["eval"]["seed_recall"]]
    cats = sorted({r["eval"]["category"] for r in results})
    summary = {
        "config": {"questions": n, "sampling_seed": SEED, "pool": len(pool), "k": tcfg.k, "m": tcfg.m,
                   "budget": tcfg.budget, "max_retries": tcfg.max_retries, "dedup_text": tcfg.dedup_text,
                   "prompt_version": ccfg.prompt_version, "temperature": ccfg.temperature,
                   "model": f"{backend.repo}@{backend.revision[:7]} (4-bit MLX)",
                   "frozen_code_sha256": {p: sha(p) for p in FROZEN}},
        "recall": {"seeds_only": mean([r["eval"]["seed_recall"] for r in results]),
                   "slm_traversal": mean([r["eval"]["final_recall"] for r in results]),
                   "by_category": {c: {"n": sum(r["eval"]["category"] == c for r in results),
                                       "seeds_only": mean([r["eval"]["seed_recall"] for r in results if r["eval"]["category"] == c]),
                                       "slm_traversal": mean([r["eval"]["final_recall"] for r in results if r["eval"]["category"] == c]),
                                       "termination": dict(Counter(r["termination_reason"] for r in results if r["eval"]["category"] == c))}
                                   for c in cats}},
        "questions_gaining_gold": len(gain), "questions_losing_gold": len(loss),
        "gain_ids": [r["query_id"] for r in gain],
        "new_gold_items": sum(len(r["eval"]["gold_discovered"]) for r in results),
        "explored_nodes_that_are_gold": sum(sum(r["eval"]["explored_is_gold"]) for r in results),
        "gold_reached_only_as_duplicate_member": sum(len(r["eval"]["gold_via_duplicate_member"]) for r in results),
        "termination_reasons": dict(term),
        "avg_explore_steps": mean([r["explore_steps"] for r in results]),
        "avg_budget_used": mean([r["budget_used"] for r in results]),
        "stop_rate": round(term.get("SLM_STOP", 0) / n, 4),
        "invalid_first_attempts": f"{sum(not c['valid'] for c in first)}/{len(first)}",
        "invalid_errors": dict(Counter(c["error"] for c in calls if not c["valid"])),
        "retries": sum(r["retries"] for r in results),
        "controller_error_rate": round(term.get("CONTROLLER_ERROR", 0) / n, 4),
        "duplicate_text_explorations": dup,
        "per_question_avg": {"slm_calls": mean([r["slm_calls"] for r in results]),
                             "input_tokens": mean([r["input_tokens"] for r in results]),
                             "output_tokens": mean([r["output_tokens"] for r in results]),
                             "slm_latency_s": mean([r["slm_latency_ms"] / 1000 for r in results]),
                             "prefilter_latency_s": mean([r["prefilter_latency_ms"] / 1000 for r in results]),
                             "total_latency_s": mean([r["total_latency_ms"] / 1000 for r in results])},
        "baseline_inputs_consistent_with_traversal_step1": sum(r["eval"]["step1_baseline_record_matches_traversal"]
                                                               for r in results),
        "mlx_peak_memory_gb": round(backend.mx.get_peak_memory() / 1e9, 2),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
