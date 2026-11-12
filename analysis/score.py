"""ANALYSIS_PLAN.md scoring phase: the ONLY place gold labels are read and recall is computed.

Every frozen run is scored once with the approved evaluator (`baselines/run_baselines_heldout.py::evaluate`, imported
unchanged) and written to canonical files that every later table/statistics script must read:
  analysis/scores/per_question.jsonl   one row per (condition, question)
  analysis/scores/per_question.csv     same rows, scalar columns only
  analysis/scores/question_info.jsonl  method-independent per-question facts (n_gold, category, seed recall)
  analysis/scores/aggregates.json/.csv raw means per (condition, population); no CIs, no tests
  analysis/scores/invariants.json      post-scoring invariant checks
  analysis/scores/scoring_manifest.json hashes of evaluator, plan, inputs (before and after) and outputs

Conditions are derived from logged records only, by direct reading or exact prefix truncation (ANALYSIS_PLAN §1).
Populations: TEST-1704, NEW-1654, SUB-300 (a condition is aggregated on a population only if it covers all of it).
No experiment output is written; input hashes are compared before and after scoring.

Usage: OMP_NUM_THREADS=1 PYTHONHASHSEED=0 .venv-sage/bin/python -P -s analysis/score.py
"""
import csv
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "experiments" / "runs"
HELDOUT = PROJECT / "adaptive_traversal" / "runs" / "heldout50_k3_m20_B4_v2"
OUT = PROJECT / "analysis" / "scores"
INTEGRITY = PROJECT / "analysis" / "integrity" / "integrity_report.json"
for p in (PROJECT / "baselines", PROJECT / "traversal_adapter", PROJECT / "candidate_prefilter",
          PROJECT / "adaptive_traversal", PROJECT / "slm_controller", PROJECT / "experiments"):
    sys.path.insert(0, str(p))

FROZEN = json.loads((PROJECT / "experiments" / "FROZEN.json").read_text())
SETS = {k: v["ids"] for k, v in json.loads((PROJECT / "experiments" / "question_sets.json").read_text())["sets"].items()}
POPS = ("TEST-1704", "NEW-1654", "SUB-300")
K = 3
GOLD_FIELDS = {"eval"}            # old gold-derived field in PILOT-50 records: dropped unread


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        for k in GOLD_FIELDS & r.keys():
            del r[k]
        rows.append(r)
    return rows


def input_files():
    files = [RUNS / d / f for d in FROZEN["canonical_run_dirs"].values() for f in ("results.jsonl", "manifest.json")]
    files += [HELDOUT / "results.jsonl", HELDOUT / "steps.jsonl", HELDOUT / "summary.json"]
    files += [PROJECT / "seed_retrieval/runs/seeds_fusionA_top100_seed0_runA/seeds_top100.csv",
              PROJECT / "experiments/question_sets.json", PROJECT / "experiments/FROZEN.json"]
    return {str(f.relative_to(PROJECT)): sha(f) for f in files}


# ---------------------------------------------------------------------------------------------------------------
def verify_before_scoring():
    approved = json.loads(INTEGRITY.read_text())
    assert approved["ALL_PASS"], "integrity report is not ALL_PASS"
    import run_baselines_heldout as RB
    now = {"baselines/run_baselines_heldout.py": sha(PROJECT / "baselines/run_baselines_heldout.py"),
           "evaluate()_source_sha256": hashlib.sha256(inspect.getsource(RB.evaluate).encode()).hexdigest(),
           "adaptive_traversal/traversal.py (normalize_text)": sha(PROJECT / "adaptive_traversal/traversal.py"),
           "analysis/integrity_checks.py": sha(PROJECT / "analysis/integrity_checks.py"),
           "ANALYSIS_PLAN.md": sha(PROJECT / "ANALYSIS_PLAN.md")}
    rec = approved["5_evaluator_hashes"]
    mism = {k: (rec[k], v) for k, v in now.items() if rec[k] != v}
    if mism:
        raise SystemExit(f"hash mismatch vs approved integrity report: {mism}")
    if now["adaptive_traversal/traversal.py (normalize_text)"] != FROZEN["code"]["adaptive_traversal/traversal.py"]:
        raise SystemExit("traversal.py differs from FROZEN.json")
    rerun = subprocess.run([sys.executable, "-P", "-s", str(PROJECT / "analysis/integrity_checks.py")],
                           capture_output=True, text=True, cwd=PROJECT)
    if rerun.returncode != 0:
        raise SystemExit("integrity checks no longer pass:\n" + rerun.stdout[-2000:])
    return RB, {**now, "integrity_rerun": rerun.stdout.strip().splitlines()[-1]}


# ---------------------------------------------------------------------------------------------------------------
def slm_cost(r, max_step=None):
    """Calls/tokens/latency summed over logged steps (optionally only steps <= max_step, for A2 truncation)."""
    steps = [s for s in r["steps"] if max_step is None or s["step"] <= max_step]
    calls = [c for s in steps for c in s.get("calls", [])]
    pre = sum(s.get("prefilter_ms", 0.0) for s in steps if "candidate_count" in s)
    return {"slm_calls_logged": len(calls), "input_tokens": sum(c["prompt_tokens"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls), "slm_latency_ms": sum(c["latency_ms"] for c in calls),
            "prefilter_latency_ms": pre, "prefilter_calls": sum(1 for s in steps if s.get("candidate_count", 0) > 0),
            "invalid_first_attempts": sum(1 for s in steps for c in s.get("calls", [])[:1] if not c["valid"]),
            "retries": sum(1 for s in steps if s.get("retried")),
            "selected_ranks": [s["selected_prefilter_rank"] for s in steps if s.get("selected_action") == "EXPLORE"]}


def main():
    RB, verified = verify_before_scoring()
    hashes_before = input_files()
    from graph_seed_adapter import GraphSeedAdapter
    from traversal import dedup_candidates
    A = GraphSeedAdapter()                                          # gold is read here, for the first time
    evaluate = RB.evaluate
    R = {d: load(RUNS / d / "results.jsonl") for d in FROZEN["canonical_run_dirs"].values()}
    pilot = load(HELDOUT / "results.jsonl")
    pst = {}
    for s in load(HELDOUT / "steps.jsonl"):
        pst.setdefault(s["query_id"], []).append(s)
    for r in pilot:
        r["steps"] = sorted(pst[r["query_id"]], key=lambda s: s["step"])
    E2 = {r["query_id"]: r for r in R["E2_baselines_B8_test1704"]}
    E2r = {r["query_id"]: r for r in R["E2r_random_B4_test1704"]}
    E2r8 = {r["query_id"]: r for r in R["E2r8_random_B8_sub300"]}
    M5B4 = {r["query_id"]: dict(r, _src="E1_slm_B4_new1654") for r in R["E1_slm_B4_new1654"]}
    M5B4.update({r["query_id"]: dict(r, _src="heldout50_k3_m20_B4_v2") for r in pilot})
    test, new, sub = (set(SETS[p]) for p in POPS)

    # ---- method-independent question facts --------------------------------------------------------------------
    info = {}
    for q in SETS["TEST-1704"]:
        seeds = E2[q]["M0_seeds"]
        gold = set(A.questions[q]["doc_gold"])
        raw = [n.node.chunk_id for n in A.get_neighbors(seeds)]
        _, members, _ = dedup_candidates(raw, seeds, lambda c: A.node(c).content)
        top = E2[q]["M4"]["steps"][0]["top_m"] if E2[q]["M4"]["steps"] else []
        top_members = {m for rep in top for m in members.get(rep, [rep])}
        left = gold - set(seeds)
        cat = ("all_gold_in_seeds" if not left else "reachable_top20" if left & top_members
               else "reachable_filtered" if left & set(raw) else "unreachable_1hop")
        info[q] = {"query_id": q, "n_gold": len(gold), "category": cat, "seeds": seeds, "hop1": set(raw),
                   "gold": gold, "gold_in_graph": sum(A.in_graph(g) for g in gold),
                   "in_TEST-1704": True, "in_NEW-1654": q in new, "in_SUB-300": q in sub, "in_PILOT-50": q not in new}

    rows = []

    def add(cond, q, retrieved, **extra):
        I = info[q]
        e = evaluate(A, q, I["seeds"], retrieved)
        expl = retrieved[K:]
        gold_pos = [i for i, c in enumerate(expl) if c in I["gold"]]
        hops = {"hop0": len(I["gold"] & set(I["seeds"])),
                "hop1": sum(1 for g in e["new_gold_ids"] if g in I["hop1"]),
                "hop2plus": sum(1 for g in e["new_gold_ids"] if g not in I["hop1"])}
        row = {**cond, "query_id": q, **{f"in_{p}": I[f"in_{p}"] for p in POPS},
               "n_gold": I["n_gold"], "category": I["category"],
               "id_recall": e["id_recall"], "id_seed_recall": e["id_seed_recall"],
               "text_recall_diag": e["text_recall"], "text_seed_recall_diag": e["text_seed_recall"],
               "hit": e["id_recall"] > 0, "gained": e["id_recall"] > e["id_seed_recall"],
               "new_gold_ids": e["new_gold_ids"], "n_new_gold": len(e["new_gold_ids"]),
               "gold_hop0": hops["hop0"], "gold_hop1": hops["hop1"], "gold_hop2plus": hops["hop2plus"],
               "expansions": len(expl), "retrieved_nodes": retrieved,
               "wasted_expansions": len(expl) - (gold_pos[-1] + 1 if gold_pos else 0)}
        row.update({"termination_reason": None, "slm_calls": 0, "slm_calls_actual": 0, "input_tokens": 0,
                    "output_tokens": 0, "total_tokens": 0, "slm_latency_ms": 0.0, "prefilter_latency_ms": None,
                    "total_latency_ms": None, "prefilter_calls": 0, "stopped": False, "stopped_step0": False,
                    "stop_when_sufficient": False, "invalid_first_attempts": 0, "retries": 0,
                    "controller_error": False, "selected_ranks": []})
        row.update(extra)
        row["total_tokens"] = row["input_tokens"] + row["output_tokens"]
        if row["termination_reason"] == "SLM_STOP":
            row["stopped"], row["stopped_step0"] = True, len(expl) == 0
            row["stop_when_sufficient"] = e["id_recall"] == 1.0
        row["controller_error"] = row["termination_reason"] == "CONTROLLER_ERROR"
        rows.append(row)
        return row

    def C(method, B, comparison="max_budget", seed=None, source=None, variant=None):
        return {"method": method, "variant": variant, "B": B, "comparison": comparison, "random_seed": seed,
                "source_run": source}

    def slm_extra(r, max_step=None, actual=None):
        c = slm_cost(r, max_step)
        if max_step is None:   # untruncated: use the record's own totals (checked against step sums in invariants)
            c.update(input_tokens=r["input_tokens"], output_tokens=r["output_tokens"], slm_latency_ms=r["slm_latency_ms"],
                     prefilter_latency_ms=r["prefilter_latency_ms"])
            total = r["total_latency_ms"]
            term, calls = r["termination_reason"], r["slm_calls"]
        else:
            total = c["slm_latency_ms"] + c["prefilter_latency_ms"]
            term = "BUDGET_EXHAUSTED" if r["explore_steps"] >= max_step else r["termination_reason"]
            calls = c["slm_calls_logged"]
        return {"termination_reason": term, "slm_calls": calls, "slm_calls_actual": actual if actual is not None else calls,
                "input_tokens": c["input_tokens"], "output_tokens": c["output_tokens"], "slm_latency_ms": c["slm_latency_ms"],
                "prefilter_latency_ms": c["prefilter_latency_ms"], "total_latency_ms": total,
                "prefilter_calls": c["prefilter_calls"], "invalid_first_attempts": c["invalid_first_attempts"],
                "retries": c["retries"], "selected_ranks": c["selected_ranks"], "_slm_calls_logged": c["slm_calls_logged"]}

    # ---- baselines (TEST-1704), max budget B in {1,2,4,8} --------------------------------------------------------
    for q in SETS["TEST-1704"]:
        r, S = E2[q], E2[q]["M0_seeds"]
        add(C("M0", 0, source="E2"), q, S)
        m2_ms = r["M2_faithful_latency_ms"]
        m4_steps = len(r["M4"]["steps"])
        for B in (1, 2, 4, 8):
            add(C("M1", B, source="E2"), q, r["M1_flat_top"][:K + B])
            add(C("M2", B, source="E2", variant="faithful"), q, S + [x["chunk_id"] for x in r["M2_faithful_top"][:B]],
                prefilter_calls=1, prefilter_latency_ms=m2_ms, total_latency_ms=m2_ms)
            add(C("M2", B, source="E2", variant="dedup_diag"), q, S + r["M2_dedup_top"][:B], prefilter_calls=1)
            exact = B == 8
            add(C("M4", B, source="E2"), q, r["M4"]["retrieved_nodes"][:K + B],
                termination_reason=("BUDGET_EXHAUSTED" if r["M4"]["explore_steps"] >= B else r["M4"]["termination_reason"]),
                prefilter_calls=min(B, m4_steps), total_latency_ms=r["M4"]["latency_ms"] if exact else None,
                prefilter_latency_ms=r["M4"]["latency_ms"] if exact else None)
        for B in (1, 2, 4):
            for s in "012":
                v = E2r[q]["random"][s]
                exact = B == 4
                add(C("M3", B, seed=int(s), source="E2r"), q, v["retrieved_nodes"][:K + B],
                    prefilter_calls=min(B, len(v["steps"])), total_latency_ms=v["latency_ms"] if exact else None,
                    prefilter_latency_ms=v["latency_ms"] if exact else None)
    for q in SETS["SUB-300"]:
        for s in "012":
            v = E2r8[q]["random"][s]
            add(C("M3", 8, seed=int(s), source="E2r8"), q, v["retrieved_nodes"], prefilter_calls=len(v["steps"]),
                total_latency_ms=v["latency_ms"], prefilter_latency_ms=v["latency_ms"])

    # ---- M5 / ablations ---------------------------------------------------------------------------------------
    for q in SETS["TEST-1704"]:
        r = M5B4[q]
        add(C("M5", 4, source=r["_src"]), q, r["retrieved_nodes"], **slm_extra(r))
    for B, d in ((1, "E3_B1_slm_sub300"), (2, "E3_B2_slm_sub300"), (8, "E3_B8_slm_sub300")):
        for r in R[d]:
            add(C("M5", B, source=d), r["query_id"], r["retrieved_nodes"], **slm_extra(r))
    for r in R["E4a_B4_nostop_test1704"]:
        add(C("A3", 4, source="E4a_B4"), r["query_id"], r["retrieved_nodes"], **slm_extra(r, actual=r.get("slm_calls_actual", r["slm_calls"])),
            a3_source=r["a3_source"])
    for r in R["E4a_B8_nostop_sub300"]:
        add(C("A3", 8, source="E4a_B8"), r["query_id"], r["retrieved_nodes"], **slm_extra(r, actual=r.get("slm_calls_actual", r["slm_calls"])),
            a3_source=r["a3_source"])
    for r in R["E4b_nohistory_B4_sub300"]:
        add(C("A1", 4, source="E4b"), r["query_id"], r["retrieved_nodes"], **slm_extra(r))
    for r in R["E4c_nobudget_B8_sub300"]:
        for B in (1, 2, 4, 8):
            ex = slm_extra(r) if B == 8 else slm_extra(r, max_step=B)
            add(C("A2", B, source="E4c" + ("" if B == 8 else f" truncated@{B}")), r["query_id"], r["retrieved_nodes"][:K + B], **ex)

    # ---- matched-expansion baselines at B=4 (TEST-1704): M1-M4 get M5's per-question expansions ------------------
    for q in SETS["TEST-1704"]:
        e, r, S = M5B4[q]["explore_steps"], E2[q], E2[q]["M0_seeds"]
        add(C("M1", 4, "matched_M5_B4", source="E2"), q, r["M1_flat_top"][:K + e])
        add(C("M2", 4, "matched_M5_B4", source="E2", variant="faithful"), q, S + [x["chunk_id"] for x in r["M2_faithful_top"][:e]],
            prefilter_calls=1, total_latency_ms=r["M2_faithful_latency_ms"], prefilter_latency_ms=r["M2_faithful_latency_ms"])
        add(C("M2", 4, "matched_M5_B4", source="E2", variant="dedup_diag"), q, S + r["M2_dedup_top"][:e], prefilter_calls=1)
        add(C("M4", 4, "matched_M5_B4", source="E2"), q, r["M4"]["retrieved_nodes"][:K + e],
            prefilter_calls=min(e, len(r["M4"]["steps"])))
        for s in "012":
            v = E2r[q]["random"][s]
            add(C("M3", 4, "matched_M5_B4", seed=int(s), source="E2r"), q, v["retrieved_nodes"][:K + e],
                prefilter_calls=min(e, len(v["steps"])))

    # ---- M3 per-question mean over the 3 seeds (D2) --------------------------------------------------------------
    by = {}
    for x in rows:
        if x["method"] == "M3":
            by.setdefault((x["B"], x["comparison"], x["query_id"]), []).append(x)
    num = ["id_recall", "text_recall_diag", "n_new_gold", "gold_hop1",
           "gold_hop2plus", "expansions", "wasted_expansions", "prefilter_calls"]
    for (B, comp, q), xs in by.items():
        assert len(xs) == 3
        # seed recall is the same for every random seed (shared seeds); copy it rather than average (float rounding)
        assert all(x["id_seed_recall"] == xs[0]["id_seed_recall"] and x["text_seed_recall_diag"] == xs[0]["text_seed_recall_diag"]
                   for x in xs)
        m = dict(xs[0], method="M3_mean", random_seed=None, retrieved_nodes=None, new_gold_ids=None)
        for f in num:
            m[f] = sum(x[f] for x in xs) / 3
        for f in ("hit", "gained"):
            m[f] = sum(x[f] for x in xs) / 3          # fraction of seeds; aggregates to the seed-mean rate
        lat = [x["total_latency_ms"] for x in xs]
        m["total_latency_ms"] = m["prefilter_latency_ms"] = (sum(lat) / 3 if None not in lat else None)
        rows.append(m)

    # ---- aggregates --------------------------------------------------------------------------------------------
    def key(x):
        return (x["method"], x["variant"], x["B"], x["comparison"], x["random_seed"])

    groups = {}
    for x in rows:
        groups.setdefault(key(x), []).append(x)
    aggs = []
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    for k, xs in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        cover = {x["query_id"] for x in xs}
        for P in POPS:
            if not set(SETS[P]) <= cover:
                continue
            ys = [x for x in xs if x[f"in_{P}"]]
            assert len(ys) == len(SETS[P])
            B = k[2]
            lat = [y["total_latency_ms"] for y in ys]
            slm = k[0] in ("M5", "A1", "A2", "A3")
            a = {"method": k[0], "variant": k[1], "B": B, "comparison": k[3], "random_seed": k[4], "population": P,
                 "n": len(ys),
                 "id_recall": mean([y["id_recall"] for y in ys]), "id_seed_recall": mean([y["id_seed_recall"] for y in ys]),
                 "text_recall_diag": mean([y["text_recall_diag"] for y in ys]),
                 "hit_rate": mean([float(y["hit"]) for y in ys]),
                 "questions_gaining": sum(float(y["gained"]) for y in ys),
                 "new_gold_total": sum(y["n_new_gold"] for y in ys),
                 "gold_hop1_total": sum(y["gold_hop1"] for y in ys), "gold_hop2plus_total": sum(y["gold_hop2plus"] for y in ys),
                 "expansions_mean": mean([y["expansions"] for y in ys]),
                 "expansions_per_B": (mean([y["expansions"] for y in ys]) / B) if B else None,
                 "wasted_expansions_mean": mean([y["wasted_expansions"] for y in ys]),
                 "prefilter_calls_mean": mean([y["prefilter_calls"] for y in ys]),
                 "total_latency_s_mean": (mean(lat) / 1000 if None not in lat else None)}
            if slm:
                stops = [y for y in ys if y["stopped"]]
                a.update({"stop_rate": len(stops) / len(ys), "stop_step0": sum(y["stopped_step0"] for y in ys),
                          "stop_when_sufficient": sum(y["stop_when_sufficient"] for y in ys), "n_stop": len(stops),
                          "slm_calls_mean": mean([y["slm_calls"] for y in ys]),
                          "slm_calls_actual_mean": mean([y["slm_calls_actual"] for y in ys]),
                          "input_tokens_mean": mean([y["input_tokens"] for y in ys]),
                          "output_tokens_mean": mean([y["output_tokens"] for y in ys]),
                          "total_tokens_mean": mean([y["total_tokens"] for y in ys]),
                          "slm_latency_s_mean": mean([y["slm_latency_ms"] for y in ys]) / 1000,
                          "prefilter_latency_s_mean": mean([y["prefilter_latency_ms"] for y in ys]) / 1000,
                          "invalid_first_attempts": sum(y["invalid_first_attempts"] for y in ys),
                          "retries": sum(y["retries"] for y in ys),
                          "controller_errors": sum(y["controller_error"] for y in ys),
                          "terminations": {t: sum(y["termination_reason"] == t for y in ys)
                                           for t in sorted({y["termination_reason"] for y in ys})}})
                gain = a["id_recall"] - a["id_seed_recall"]
                a["recall_gain_per_1k_tokens"] = gain / (a["total_tokens_mean"] / 1000) if a["total_tokens_mean"] else None
                a["recall_gain_per_second"] = gain / a["total_latency_s_mean"] if a["total_latency_s_mean"] else None
            a["category_counts"] = {c: sum(y["category"] == c for y in ys) for c in
                                    ("all_gold_in_seeds", "reachable_top20", "reachable_filtered", "unreachable_1hop")}
            aggs.append(a)
    # M3 seed SD of aggregate recall (headline mean ± SD across seeds)
    for a in aggs:
        if a["method"] == "M3_mean":
            per = [b["id_recall"] for b in aggs if b["method"] == "M3" and b["B"] == a["B"] and b["comparison"] == a["comparison"]
                   and b["population"] == a["population"]]
            mu = sum(per) / 3
            a["id_recall_seed_values"] = per
            a["id_recall_seed_sd"] = (sum((p - mu) ** 2 for p in per) / 2) ** 0.5   # sample SD, n=3

    # ---- invariants --------------------------------------------------------------------------------------------
    inv = {}

    def chk(name, cond, **d):
        inv[name] = {"PASS": bool(cond), **d}

    real = [x for x in rows if x["method"] != "M3_mean"]
    chk("recall_in_[0,1]", all(0 <= x[f] <= 1 for x in rows for f in ("id_recall", "id_seed_recall", "text_recall_diag")))
    chk("evidence_superset_of_seeds (recall >= seed recall)", all(x["id_recall"] >= x["id_seed_recall"] - 1e-12 for x in rows))
    chk("retrieved starts with the 3 seeds", all(x["retrieved_nodes"][:K] == info[x["query_id"]]["seeds"] for x in real))
    chk("text_recall_diag >= id_recall", all(x["text_recall_diag"] >= x["id_recall"] - 1e-12 for x in rows))
    chk("n_gold >= 1 for every question", all(i["n_gold"] >= 1 for i in info.values()),
        gold_ids_not_in_graph=sum(i["n_gold"] - i["gold_in_graph"] for i in info.values()))
    chk("seed recall identical across all conditions per question",
        len({(x["query_id"], x["id_seed_recall"]) for x in rows}) == len(info))
    chk("expansions <= B (max budget)", all(x["expansions"] <= x["B"] for x in real if x["comparison"] == "max_budget"))
    chk("hop counts sum to discovered gold", all(x["gold_hop1"] + x["gold_hop2plus"] == x["n_new_gold"] for x in real))
    counts = {}
    for k, xs in groups.items():
        counts["|".join(map(str, k))] = len(xs)
    chk("each condition has one row per question of its set",
        all(len({x["query_id"] for x in xs}) == len(xs) and len(xs) in (1704, 300) for xs in groups.values()), counts=counts)
    R4 = {(x["method"], x["variant"], x["B"], x["comparison"], x["random_seed"], x["query_id"]): x for x in real}

    def mono(method, variant, Bs, seed=None):
        bad = 0
        for q in info:
            vals = [R4.get((method, variant, b, "max_budget", seed, q)) for b in Bs]
            vals = [v for v in vals if v]
            bad += any(vals[i + 1]["id_recall"] < vals[i]["id_recall"] - 1e-12 or
                       vals[i + 1]["retrieved_nodes"][:len(vals[i]["retrieved_nodes"])] != vals[i]["retrieved_nodes"]
                       for i in range(len(vals) - 1))
        return bad
    chk("cumulative: truncated methods nested + recall non-decreasing in B",
        all(mono(*a) == 0 for a in [("M1", None, (1, 2, 4, 8)), ("M2", "faithful", (1, 2, 4, 8)), ("M2", "dedup_diag", (1, 2, 4, 8)),
                                     ("M4", None, (1, 2, 4, 8)), ("A2", None, (1, 2, 4, 8))] +
            [("M3", None, (1, 2, 4, 8), s) for s in (0, 1, 2)]))
    sup = lambda a, b: all(set(R4[(*b, q)]["retrieved_nodes"]) <= set(R4[(*a, q)]["retrieved_nodes"])
                           and R4[(*a, q)]["id_recall"] >= R4[(*b, q)]["id_recall"] - 1e-12
                           for q in (SETS["TEST-1704"] if a[2] == 4 else SETS["SUB-300"]))
    chk("A3 evidence superset of M5 (B=4 TEST, B=8 SUB)",
        sup(("A3", None, 4, "max_budget", None), ("M5", None, 4, "max_budget", None))
        and sup(("A3", None, 8, "max_budget", None), ("M5", None, 8, "max_budget", None)))
    chk("A3 equals M5 on every question where M5 did not stop",
        all(R4[("A3", None, b, "max_budget", None, q)]["retrieved_nodes"] == R4[("M5", None, b, "max_budget", None, q)]["retrieved_nodes"]
            for b, P in ((4, "TEST-1704"), (8, "SUB-300")) for q in SETS[P]
            if R4[("M5", None, b, "max_budget", None, q)]["termination_reason"] != "SLM_STOP"))
    chk("M2 dedup B=1 == M4 B=1 (same top-1 pre-filter pick)",
        all(R4[("M2", "dedup_diag", 1, "max_budget", None, q)]["retrieved_nodes"] == R4[("M4", None, 1, "max_budget", None, q)]["retrieved_nodes"]
            for q in info))
    chk("matched: baseline expansions == M5 expansions (or fewer only when the baseline ran out)",
        all((x["expansions"] == R4[("M5", None, 4, "max_budget", None, x["query_id"])]["expansions"])
            or (x["expansions"] < R4[("M5", None, 4, "max_budget", None, x["query_id"])]["expansions"]
                and R4[(x["method"], x["variant"], 4, "max_budget", x["random_seed"], x["query_id"])]["expansions"] == x["expansions"])
            for x in real if x["comparison"] == "matched_M5_B4"))
    chk("matched rows are prefixes of the max-budget rows",
        all(R4[(x["method"], x["variant"], 4, "max_budget", x["random_seed"], x["query_id"])]["retrieved_nodes"][:len(x["retrieved_nodes"])]
            == x["retrieved_nodes"] for x in real if x["comparison"] == "matched_M5_B4"))
    slm_rows = [x for x in real if x["method"] in ("M5", "A1", "A3") or (x["method"] == "A2" and x["B"] == 8)]
    chk("SLM: step-summed calls == recorded slm_calls",
        all(x["_slm_calls_logged"] == x["slm_calls"] for x in slm_rows),
        mismatches=sum(x["_slm_calls_logged"] != x["slm_calls"] for x in slm_rows))
    tok_bad = sum(abs(slm_cost(r)["input_tokens"] - r["input_tokens"]) > 0 or abs(slm_cost(r)["output_tokens"] - r["output_tokens"]) > 0
                  for r in list(M5B4.values()) + [r for d in ("E3_B1_slm_sub300", "E3_B2_slm_sub300", "E3_B8_slm_sub300",
                                                               "E4b_nohistory_B4_sub300", "E4c_nobudget_B8_sub300") for r in R[d]])
    chk("SLM: step-summed tokens == recorded tokens (basis for A2 truncated cost)", tok_bad == 0, mismatches=tok_bad)
    chk("A2 truncated rows: expansions == min(B, A2@8 expansions)",
        all(R4[("A2", None, b, "max_budget", None, q)]["expansions"] == min(b, R4[("A2", None, 8, "max_budget", None, q)]["expansions"])
            for b in (1, 2, 4) for q in SETS["SUB-300"]))
    chk("M3_mean rows = mean of the three seeds", all(
        abs(m["id_recall"] - sum(R4[("M3", None, m["B"], m["comparison"], s, m["query_id"])]["id_recall"] for s in (0, 1, 2)) / 3) < 1e-12
        for m in rows if m["method"] == "M3_mean"))
    chk("SLM_STOP only in methods with adaptive stopping",
        all(x["termination_reason"] != "SLM_STOP" for x in real if x["method"] == "A3"))
    chk("aggregate counts per population", all(a["n"] == len(SETS[a["population"]]) for a in aggs),
        populations={P: sorted({f'{a["method"]}{"/" + a["variant"] if a["variant"] else ""} B{a["B"]} {a["comparison"]}'
                                 + (f' s{a["random_seed"]}' if a["random_seed"] is not None else "") for a in aggs if a["population"] == P})
                     for P in POPS})
    hashes_after = input_files()
    chk("no input file changed during scoring", hashes_before == hashes_after)
    inv["ALL_PASS"] = all(v["PASS"] for k, v in inv.items() if isinstance(v, dict) and "PASS" in v)

    # ---- write -------------------------------------------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    scalar = [k for k in rows[0] if k not in ("retrieved_nodes", "new_gold_ids", "selected_ranks") and not k.startswith("_")]
    with open(OUT / "per_question.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps({k: v for k, v in x.items() if not k.startswith("_")}) + "\n")
    with open(OUT / "per_question.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scalar + ["a3_source"], extrasaction="ignore")
        w.writeheader()
        for x in rows:
            w.writerow(x)
    with open(OUT / "question_info.jsonl", "w") as f:
        for q, i in info.items():
            f.write(json.dumps({k: (sorted(v) if isinstance(v, set) else v) for k, v in i.items() if k != "hop1"}) + "\n")
    (OUT / "aggregates.json").write_text(json.dumps(aggs, indent=1))
    flat_keys = [k for k in aggs[0] if not isinstance(aggs[0][k], dict)]
    allk = []
    for a in aggs:
        for k in a:
            if k not in allk and not isinstance(a[k], (dict, list)):
                allk.append(k)
    with open(OUT / "aggregates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allk, extrasaction="ignore")
        w.writeheader()
        for a in aggs:
            w.writerow(a)
    (OUT / "invariants.json").write_text(json.dumps(inv, indent=1))
    outputs = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.name != "scoring_manifest.json"}
    (OUT / "scoring_manifest.json").write_text(json.dumps({
        "verified_before_scoring": verified, "evaluator": "baselines/run_baselines_heldout.py::evaluate (imported unchanged)",
        "primary_metric": "id_recall", "diagnostic_only": ["text_recall_diag", "text_seed_recall_diag"],
        "dropped_unread_fields": sorted(GOLD_FIELDS), "populations": {P: len(SETS[P]) for P in POPS},
        "rows": len(rows), "aggregates": len(aggs), "inputs_sha256_before": hashes_before,
        "inputs_unchanged_after": hashes_before == hashes_after, "score_py_sha256": sha(Path(__file__)),
        "outputs_sha256": outputs}, indent=1))
    for k, v in inv.items():
        if isinstance(v, dict):
            print(("PASS " if v["PASS"] else "FAIL ") + k)
    print("ALL_PASS" if inv["ALL_PASS"] else "FAILED", f"({len(rows)} rows, {len(aggs)} aggregates)")
    sys.exit(0 if inv["ALL_PASS"] else 1)


if __name__ == "__main__":
    main()
