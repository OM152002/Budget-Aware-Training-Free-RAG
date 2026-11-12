"""Equal-budget comparison on the held-out 50 (frozen v2 SLM results are read, never re-run).

Methods: seeds-only | random (seed 0) | sage_score | frozen SLM (adaptive_traversal/runs/heldout50_k3_m20_B4_v2).
Comparisons:
  max_budget        baselines get B=4 (they never stop early; only NO_CANDIDATES ends them sooner)
  matched_expansion baselines get exactly the SLM's number of EXPLORE actions on that question
Metrics (gold used only here, after retrieval):
  primary   ID recall      |G ∩ R| / |G|                     (same as all earlier runs)
  diagnostic text recall   gold g credited if normalized text(g) equals normalized text of any retrieved node
Checks: matched runs must be prefixes of the B=4 runs (determinism); sage_score step-1 pick must equal the
recorded baseline_inputs.jsonl step-1 rank-1 candidate.

Usage: python -P -s run_baselines_heldout.py <out_dir>   (stderr: SAGE progress bars)
"""
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
for p in (Path(__file__).resolve().parent, PROJECT / "traversal_adapter", PROJECT / "candidate_prefilter",
          PROJECT / "adaptive_traversal", PROJECT / "slm_controller"):
    sys.path.insert(0, str(p))
from fixed_traversal import traverse_fixed  # noqa: E402
from graph_seed_adapter import GraphSeedAdapter  # noqa: E402
from sage_prefilter import SagePrefilter  # noqa: E402
from traversal import normalize_text  # noqa: E402

HELDOUT = PROJECT / "adaptive_traversal" / "runs" / "heldout50_k3_m20_B4_v2"
B, RANDOM_SEED = 4, 0


def evaluate(A, qid, seeds, retrieved):
    gold = set(A.questions[qid]["doc_gold"])
    norm = lambda c: normalize_text(A.node(c).content)
    R, Sd = set(retrieved), set(seeds)
    rt, st = {norm(c) for c in retrieved}, {norm(c) for c in seeds}
    explored = [c for c in retrieved if c not in Sd]
    return {"id_recall": len(gold & R) / len(gold), "id_seed_recall": len(gold & Sd) / len(gold),
            "text_recall": sum(norm(g) in rt for g in gold) / len(gold),
            "text_seed_recall": sum(norm(g) in st for g in gold) / len(gold),
            "new_gold_ids": sorted(gold & set(explored))}


def aggregate(rows):
    n = len(rows); mean = lambda xs: round(sum(xs) / n, 4)
    return {"id_recall": mean([r["id_recall"] for r in rows]),
            "text_recall_diag": mean([r["text_recall"] for r in rows]),
            "questions_gaining_gold_id": sum(r["id_recall"] > r["id_seed_recall"] for r in rows),
            "questions_gaining_gold_text_diag": sum(r["text_recall"] > r["text_seed_recall"] for r in rows),
            "additional_gold_found_id": sum(len(r["new_gold_ids"]) for r in rows),
            "avg_expansions": round(sum(r["explore_steps"] for r in rows) / n, 3),
            "avg_runtime_s": round(sum(r["runtime_s"] for r in rows) / n, 3)}


def main():
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=False)
    A = GraphSeedAdapter()
    P = SagePrefilter(A, use_cache=False)
    qids = json.loads((HELDOUT / "questions.json").read_text())["question_ids"]
    slm = {json.loads(l)["query_id"]: json.loads(l) for l in open(HELDOUT / "results.jsonl")}
    binp = {json.loads(l)["query_id"]: json.loads(l) for l in open(HELDOUT / "baseline_inputs.jsonl")}
    assert set(slm) == set(qids) == set(binp)

    rows = {(meth, cmp): [] for meth in ("seeds_only", "random", "sage_score", "slm")
            for cmp in ("max_budget", "matched_expansion")}
    checks = {"matched_is_prefix_of_max": 0, "sage_step1_matches_recorded": 0, "seeds_match_slm": 0}
    t0 = time.time()
    with open(out / "runs.jsonl", "w") as f:
        for i, q in enumerate(qids):
            s = slm[q]
            S, n_slm = s["seed_nodes"], s["explore_steps"]
            runs = {}
            for pol in ("random", "sage_score"):
                for cmp, budget in (("max_budget", B), ("matched_expansion", n_slm)):
                    r = traverse_fixed(q, A, P, pol, budget, seed=RANDOM_SEED)
                    runs[(pol, cmp)] = r
                    f.write(json.dumps({k: v for k, v in r.items()} | {"comparison": cmp}) + "\n")
            checks["seeds_match_slm"] += runs[("sage_score", "max_budget")]["seed_nodes"] == S
            checks["matched_is_prefix_of_max"] += all(
                runs[(p, "max_budget")]["retrieved_nodes"][:len(S) + runs[(p, "matched_expansion")]["explore_steps"]]
                == runs[(p, "matched_expansion")]["retrieved_nodes"] for p in ("random", "sage_score"))
            st = runs[("sage_score", "max_budget")]["steps"]
            checks["sage_step1_matches_recorded"] += bool(st) and st[0]["selected"] == binp[q]["step1_ranked"][0]["chunk_id"]
            for cmp in ("max_budget", "matched_expansion"):
                rows[("seeds_only", cmp)].append({**evaluate(A, q, S, S), "explore_steps": 0, "runtime_s": 0.0})
                rows[("slm", cmp)].append({**evaluate(A, q, S, s["retrieved_nodes"]), "explore_steps": n_slm,
                                           "runtime_s": s["total_latency_ms"] / 1000})
                for pol in ("random", "sage_score"):
                    r = runs[(pol, cmp)]
                    rows[(pol, cmp)].append({**evaluate(A, q, S, r["retrieved_nodes"]),
                                             "explore_steps": r["explore_steps"], "runtime_s": r["latency_ms"] / 1000})
            print(f"{i + 1:2d}/{len(qids)} {q} slm_explore={n_slm} "
                  + " ".join(f"{p}:{rows[(p, 'max_budget')][-1]['id_recall']:.2f}" for p in ("seeds_only", "random", "sage_score", "slm")),
                  flush=True)

    summary = {"config": {"questions": len(qids), "B_max": B, "random_seed": RANDOM_SEED, "k": 3, "m": 20,
                          "dedup_text": True, "prefilter": "SAGE _calculate_hybrid_similarities, cache-free",
                          "slm_source": str(HELDOUT.relative_to(PROJECT)),
                          "note": "SLM row is identical in both comparisons (its own expansions)."},
               "checks": {k: f"{v}/{len(qids)}" for k, v in checks.items()},
               "max_budget": {m: aggregate(rows[(m, "max_budget")]) for m in ("seeds_only", "random", "sage_score", "slm")},
               "matched_expansion": {m: aggregate(rows[(m, "matched_expansion")]) for m in ("seeds_only", "random", "sage_score", "slm")},
               "wall_seconds": round(time.time() - t0, 1)}
    per_q = [{"query_id": q, **{f"{m}_{c}": {k: rows[(m, c)][i][k] for k in ("id_recall", "text_recall", "explore_steps")}
                                for m in ("random", "sage_score", "slm") for c in ("max_budget", "matched_expansion")}}
             for i, q in enumerate(qids)]
    (out / "per_question.jsonl").write_text("\n".join(json.dumps(x) for x in per_q) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
