"""Pre-evaluation verification of the pre-filter.

V1  cache vs cache-free: with SAGE's embedding cache warmed by OTHER questions (so cached vectors were computed in
    different batch compositions), rankings must be identical to cache-free scoring for every sampled case.
V2  equivalence with SAGE's own end-to-end neighbor selection: for the same seeds, SAGE's
    get_graph_neighbors_for_retrieved_chunks (L591-618, fed a networkx graph built from the reconstructed edges)
    must select the same top-m set, apart from exact ties at the cut-off (SAGE orders ties by set iteration).

Usage: python -P -s verify_prefilter.py <out_dir> [--n 60]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sage_prefilter import SagePrefilter  # noqa: E402
from graph_seed_adapter import GraphSeedAdapter  # noqa: E402

KS, MS = [1, 3, 5], [5, 10, 20, 50]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    A = GraphSeedAdapter()
    doc_q = [q for q in A.question_ids if A.questions[q]["eval_group"] == "doc"]
    rng = random.Random(0)
    sample = rng.sample(doc_q, a.n)
    warm = rng.sample([q for q in doc_q if q not in set(sample)], a.n)
    cases = [(q, k) for q in sample for k in KS]

    P = SagePrefilter(A, use_cache=True)
    t = time.time()
    for q in warm:  # warm the cache with other questions' candidates first
        ex = A.expand(q, 5)
        P.score(ex.question, [n.node.chunk_id for n in ex.neighbors])
    warm_s = time.time() - t
    cached, t_cached = {}, []
    for q, k in cases:
        ex = A.expand(q, k)
        t = time.perf_counter(); cached[(q, k)] = P.score(ex.question, [n.node.chunk_id for n in ex.neighbors])
        t_cached.append(time.perf_counter() - t)

    F = SagePrefilter(A, use_cache=False)
    free, t_free = {}, []
    for q, k in cases:
        ex = A.expand(q, k)
        t = time.perf_counter(); free[(q, k)] = F.score(ex.question, [n.node.chunk_id for n in ex.neighbors])
        t_free.append(time.perf_counter() - t)

    v1 = {"cases": len(cases), "ranking_identical": 0, "max_score_abs_diff": 0.0, "topm_identical": {m: 0 for m in MS}}
    for c in cases:
        r1, r2 = cached[c], free[c]
        v1["ranking_identical"] += [x for x, _ in r1] == [x for x, _ in r2]
        d2 = dict(r2)
        if r1:
            v1["max_score_abs_diff"] = max(v1["max_score_abs_diff"], max(abs(s - d2[x]) for x, s in r1))
        for m in MS:
            v1["topm_identical"][m] += [x for x, _ in r1[:m]] == [x for x, _ in r2[:m]]
    v1["PASS"] = v1["ranking_identical"] == len(cases)

    # V2: SAGE's own end-to-end selector on the same graph and seeds
    G = nx.Graph(); G.add_nodes_from(A.adj)
    G.add_edges_from((s, t) for s, nb in A.adj.items() for t in nb if s < t)
    F.analyzer.graph = G
    v2 = {"cases": 0, "same_set": 0, "differs_only_by_boundary_ties": 0, "real_mismatch": 0, "examples": []}
    for (q, k) in cases:
        ex = A.expand(q, k)
        seeds = [s.node.chunk_id for s in ex.seeds]
        ours = free[(q, k)]
        for m in [5, 20]:
            F.clear_cache()
            theirs = F.analyzer.get_graph_neighbors_for_retrieved_chunks(seeds, ex.question, [], [], m)
            v2["cases"] += 1
            mine = [x for x, _ in ours[:m]]
            if set(mine) == set(theirs):
                v2["same_set"] += 1
                continue
            sc = dict(ours)
            cut = sc[mine[-1]] if mine else None
            diff = set(mine) ^ set(theirs)
            if cut is not None and all(abs(sc[x] - cut) < 1e-12 for x in diff):
                v2["differs_only_by_boundary_ties"] += 1
            else:
                v2["real_mismatch"] += 1
                if len(v2["examples"]) < 3:
                    v2["examples"].append({"q": q, "k": k, "m": m, "ours_only": sorted(set(mine) - set(theirs))[:3],
                                           "sage_only": sorted(set(theirs) - set(mine))[:3]})
    v2["PASS"] = v2["real_mismatch"] == 0

    rep = {"V1_cache_vs_cache_free": v1, "V2_vs_sage_get_graph_neighbors": v2,
           "timing_s": {"warm_questions": a.n, "warm_total": round(warm_s, 1),
                        "per_call_cached_mean": round(sum(t_cached) / len(t_cached), 3),
                        "per_call_cache_free_mean": round(sum(t_free) / len(t_free), 3)},
           "ALL_PASS": v1["PASS"] and v2["PASS"]}
    (out / "verification.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
