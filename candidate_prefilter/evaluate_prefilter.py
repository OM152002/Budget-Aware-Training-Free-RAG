"""Evaluate the SAGE-scorer top-m pre-filter on the primary document-retrieval set.

Questions: the 1,728 questions with >=1 document gold (BM25-file gold restricted to the 21,742-doc corpus).
For k in {1,3,5}: seeds S = adapter top-k, candidates N = adapter 1-hop neighbors of S (seeds excluded).
Candidates are scored once per (question, k) by SagePrefilter, cache-free (the cache failed verification); top-m for m in {5,10,20} (+50 diag).

Metrics (macro = mean over questions unless noted):
  seed_recall            |G ∩ S| / |G|
  recall_before_filter   |G ∩ (S ∪ N)| / |G|
  recall_after_top_m     |G ∩ (S ∪ top_m)| / |G|
  reachable gold         R = G ∩ N  (gold reachable in one hop and not already a seed)
  retention@m micro      Σ|R ∩ top_m| / Σ|R|
  retention@m macro      mean over questions with R≠∅ of |R ∩ top_m| / |R|
  any_reachable_kept@m   fraction of questions with R≠∅ where top_m keeps ≥1 of R
Reference (not a method): the same retention for the adapter's own order (best seed rank, then chunk_id).

Usage: python -P -s evaluate_prefilter.py <out_dir>
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sage_prefilter import SagePrefilter  # noqa: E402
from graph_seed_adapter import GraphSeedAdapter  # noqa: E402

KS, MS, MS_DIAG = [1, 3, 5], [5, 10, 20], [50]


def summarize(v):
    v = np.asarray(v, float)
    return {"mean": round(float(v.mean()), 2), "median": float(np.median(v)), "p90": float(np.quantile(v, .9)),
            "max": float(v.max()), "zero": int((v == 0).sum())}


def main():
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=False)
    proc, t_all = psutil.Process(), time.time()
    A = GraphSeedAdapter()
    P = SagePrefilter(A, use_cache=False)  # cache failed verification (runs/verify_n60): ranking-level differences
    qids = [q for q in A.question_ids if A.questions[q]["eval_group"] == "doc"]

    per_k = {k: {"cands": [], "secs": [], "seed_rec": [], "rec_before": [],
                 "rec_after": {m: [] for m in MS + MS_DIAG}, "R": [], "kept": {m: [] for m in MS + MS_DIAG},
                 "kept_adapter": {m: [] for m in MS + MS_DIAG}} for k in KS}
    peak = 0
    with open(out / "rankings.jsonl", "w") as f:
        for i, q in enumerate(qids):
            gold = set(A.questions[q]["doc_gold"])
            for k in KS:
                ex = A.expand(q, k)
                S = [s.node.chunk_id for s in ex.seeds]
                N = [n.node.chunk_id for n in ex.neighbors]            # adapter order
                t = time.perf_counter(); ranked = P.score(ex.question, N); dt = time.perf_counter() - t
                f.write(json.dumps({"question_id": q, "k": k, "seeds": S,
                                    "ranked": [[c, s] for c, s in ranked]}) + "\n")
                d = per_k[k]
                d["cands"].append(len(N)); d["secs"].append(dt)
                R = gold & set(N)
                d["seed_rec"].append(len(gold & set(S)) / len(gold))
                d["rec_before"].append(len(gold & (set(S) | set(N))) / len(gold))
                d["R"].append(len(R))
                for m in MS + MS_DIAG:
                    top = {c for c, _ in ranked[:m]}
                    d["rec_after"][m].append(len(gold & (set(S) | top)) / len(gold))
                    d["kept"][m].append(len(R & top))
                    d["kept_adapter"][m].append(len(R & set(N[:m])))
            if i % 100 == 0:
                peak = max(peak, proc.memory_info().rss)
                print(f"{i}/{len(qids)} {time.time() - t_all:.0f}s", flush=True)
    peak = max(peak, proc.memory_info().rss)

    res = {}
    for k in KS:
        d = per_k[k]
        R = np.array(d["R"]); has = R > 0
        def retention(kept):
            kept = np.array(kept)
            return {"micro": round(float(kept.sum() / R.sum()), 4) if R.sum() else None,
                    "macro": round(float((kept[has] / R[has]).mean()), 4) if has.any() else None,
                    "any_reachable_kept": round(float((kept[has] > 0).mean()), 4) if has.any() else None}
        res[f"k={k}"] = {
            "questions": len(qids),
            "candidates": summarize(d["cands"]),
            "score_seconds_per_question": {"mean": round(float(np.mean(d["secs"])), 4),
                                           "median": round(float(np.median(d["secs"])), 4),
                                           "p90": round(float(np.quantile(d["secs"], .9)), 4),
                                           "total": round(float(np.sum(d["secs"])), 1)},
            "seed_recall": round(float(np.mean(d["seed_rec"])), 4),
            "recall_before_filter": round(float(np.mean(d["rec_before"])), 4),
            "reachable_gold": {"questions_with_any": int(has.sum()), "total_gold_items": int(R.sum())},
            "recall_after_top_m": {f"m={m}" + (" (diag)" if m in MS_DIAG else ""): round(float(np.mean(d["rec_after"][m])), 4)
                                   for m in MS + MS_DIAG},
            "retention_of_reachable_gold": {f"m={m}" + (" (diag)" if m in MS_DIAG else ""): retention(d["kept"][m])
                                            for m in MS + MS_DIAG},
            "REFERENCE_adapter_order_retention": {f"m={m}": retention(d["kept_adapter"][m]) for m in MS + MS_DIAG},
        }
    h = hashlib.sha256((out / "rankings.jsonl").read_bytes()).hexdigest()
    rep = {"config": {"scorer": "GraphEnhancedAccuracyAnalyzer._calculate_hybrid_similarities (0.40 emb + 0.60 BM25)",
                      "question_entities_events": "empty (A6 unavailable)", "tie_break": "stable sort (-score, chunk_id)",
                      "embedding_cache": "disabled: cache-free scoring per call (cache failed identical-ranking verification)", "ks": KS,
                      "ms": MS, "ms_diagnostic": MS_DIAG, "device": "cpu"},
           "results": res, "rankings_sha256": h, "total_seconds": round(time.time() - t_all, 1),
           "peak_rss_gb": round(peak / 1e9, 2)}
    (out / "evaluation.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
