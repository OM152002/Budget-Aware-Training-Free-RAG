"""ANALYSIS_PLAN.md statistics: paired bootstrap CIs + exact sign tests + Holm within each family.

Reads ONLY the canonical scores (analysis/scores/per_question.jsonl), after checking their hashes against
analysis/scores/scoring_manifest.json. Recall is never recomputed here.
  bootstrap : 10,000 resamples of questions, percentile 95% CI, seed 20261102 (D7). One resample-index matrix per
              population size, shared by every comparison on that population.
  sign test : exact two-sided binomial on wins vs losses, ties dropped (|diff| <= 1e-12 is a tie).
  Holm      : on raw sign-test p-values within each family (D3). CIs are unadjusted.
  M3        : per-question mean over the 3 random seeds (D2); per-seed comparisons reported as supplementary.
Δ = (frozen full SLM M5) − (comparator); a "win" means M5 has higher recall on that question.

Outputs: analysis/stats/{comparisons.json, comparisons.csv, absolute_recall_ci.json, stats_manifest.json}

Usage: .venv-sage/bin/python -P -s analysis/stats.py
"""
import csv
import hashlib
import json
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
SCORES = PROJECT / "analysis" / "scores"
OUT = PROJECT / "analysis" / "stats"
SEED, N_BOOT, TIE = 20261102, 10_000, 1e-12
SETS = {k: v["ids"] for k, v in json.loads((PROJECT / "experiments/question_sets.json").read_text())["sets"].items()}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sign_test(w, l):
    n = w + l
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(w, l) + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2 ** n)))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main():
    man = json.loads((SCORES / "scoring_manifest.json").read_text())
    bad = {k: v for k, v in man["outputs_sha256"].items() if sha(SCORES / k) != v}
    if bad:
        raise SystemExit(f"canonical score files changed since scoring: {sorted(bad)}")
    inv = json.loads((SCORES / "invariants.json").read_text())
    assert inv["ALL_PASS"]

    R = {}
    for line in open(SCORES / "per_question.jsonl"):
        x = json.loads(line)
        R[(x["method"], x["variant"], x["B"], x["comparison"], x["random_seed"], x["query_id"])] = x["id_recall"]

    def vec(cond, pop):
        return np.array([R[(*cond, q)] for q in SETS[pop]], dtype=float)

    idx = {}
    def boot_idx(n):
        if n not in idx:
            idx[n] = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))
        return idx[n]

    def ci(v):
        bm = v[boot_idx(len(v))].mean(axis=1)
        return [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]

    M5 = lambda B, comp="max_budget": ("M5", None, B, "max_budget", None)
    cond = {
        "M0": lambda B, c: ("M0", None, 0, "max_budget", None),
        "M1": lambda B, c: ("M1", None, B, c, None),
        "M2 faithful": lambda B, c: ("M2", "faithful", B, c, None),
        "M3 (mean of 3 seeds)": lambda B, c: ("M3_mean", None, B, c, None),
        "M4": lambda B, c: ("M4", None, B, c, None),
    }
    abl = {"A1 no history": lambda B: ("A1", None, B, "max_budget", None),
           "A2 no budget input": lambda B: ("A2", None, B, "max_budget", None),
           "A3 no adaptive stop": lambda B: ("A3", None, B, "max_budget", None)}

    fams = {}  # family -> list of (label, ref_cond, other_cond, population, B, rq)
    for pop in ("TEST-1704", "NEW-1654"):
        fams[f"RQ1/T1 max-budget B=4 {pop}"] = [(f"M5 vs {m}", M5(4), f(4, "max_budget"), pop, 4, "RQ1") for m, f in cond.items()]
        fams[f"RQ1/T2 matched-expansion B=4 {pop}"] = [(f"M5 vs {m} (matched)", M5(4), f(4, "matched_M5_B4"), pop, 4, "RQ1")
                                                       for m, f in cond.items() if m != "M0"]
    for B in (1, 2, 4, 8):
        fams[f"RQ3/T3 budget B={B} SUB-300"] = [(f"M5 vs {m}", M5(B), f(B, "max_budget"), "SUB-300", B, "RQ3")
                                               for m, f in cond.items()]
    fams["RQ2+RQ3/T4 ablations"] = (
        [("M5 vs A1 no history", M5(4), abl["A1 no history"](4), "SUB-300", 4, "RQ2")]
        + [(f"M5 vs A2 no budget input", M5(B), abl["A2 no budget input"](B), "SUB-300", B, "RQ3") for B in (1, 2, 4, 8)]
        + [("M5 vs A3 no adaptive stop", M5(4), abl["A3 no adaptive stop"](4), "TEST-1704", 4, "RQ2"),
           ("M5 vs A3 no adaptive stop", M5(8), abl["A3 no adaptive stop"](8), "SUB-300", 8, "RQ2")])
    supp = (  # reported, not part of any Holm family
        [("M5 vs A3 no adaptive stop", M5(4), abl["A3 no adaptive stop"](4), "NEW-1654", 4, "RQ2"),
         ("M5 vs A3 no adaptive stop", M5(4), abl["A3 no adaptive stop"](4), "SUB-300", 4, "RQ2")]
        + [(f"M5 vs M2 dedup (diagnostic)", M5(4), ("M2", "dedup_diag", 4, c, None), p, 4, "RQ1")
           for p in ("TEST-1704", "NEW-1654") for c in ("max_budget",)]
        + [(f"M5 vs M3 seed {s}", M5(B), ("M3", None, B, "max_budget", s), p, B, "RQ1" if B == 4 and p != "SUB-300" else "RQ3")
           for s in (0, 1, 2) for B, ps in ((4, ("TEST-1704", "NEW-1654", "SUB-300")), (1, ("SUB-300",)), (2, ("SUB-300",)),
                                            (8, ("SUB-300",))) for p in ps]
    )

    def compare(label, ref, oth, pop, B, rq, family):
        a, b = vec(ref, pop), vec(oth, pop)
        d = a - b
        w, l = int((d > TIE).sum()), int((d < -TIE).sum())
        return {"family": family, "rq": rq, "comparison": label, "population": pop, "n": len(d), "B": B,
                "reference": "|".join(map(str, ref)), "comparator": "|".join(map(str, oth)),
                "recall_reference": float(a.mean()), "recall_comparator": float(b.mean()),
                "mean_diff": float(d.mean()), "ci95": ci(d), "wins": w, "ties": len(d) - w - l, "losses": l,
                "p_sign_raw": sign_test(w, l)}

    rows = []
    for fam, comps in fams.items():
        res = [compare(*c, family=fam) for c in comps]
        for r, p in zip(res, holm([r["p_sign_raw"] for r in res])):
            r["p_sign_holm"], r["family_size"] = p, len(res)
        rows += res
    for c in supp:
        r = compare(*c, family="supplementary (no Holm)")
        r["p_sign_holm"], r["family_size"] = None, None
        rows.append(r)

    # descriptive (plan §4): M5 vs A3 restricted to questions where M5 stopped; no test
    desc = []
    P = {}
    for line in open(SCORES / "per_question.jsonl"):
        x = json.loads(line)
        if x["method"] == "M5" and x["comparison"] == "max_budget":
            P[(x["B"], x["query_id"])] = x["termination_reason"]
    for B, pop in ((4, "TEST-1704"), (4, "NEW-1654"), (8, "SUB-300")):
        qs = [q for q in SETS[pop] if P[(B, q)] == "SLM_STOP"]
        d = np.array([R[(*M5(B), q)] - R[("A3", None, B, "max_budget", None, q)] for q in qs])
        desc.append({"comparison": "M5 vs A3 on M5-stopped questions only (descriptive, no test)", "population": pop, "B": B,
                     "n": len(qs), "mean_diff": float(d.mean()), "wins": int((d > TIE).sum()),
                     "ties": int((np.abs(d) <= TIE).sum()), "losses": int((d < -TIE).sum())})

    # absolute recall CIs for every condition used above
    absci, seen = [], set()
    for r in rows:
        for c in (r["reference"], r["comparator"]):
            k = (c, r["population"])
            if k in seen:
                continue
            seen.add(k)
            parts = c.split("|")
            key = (parts[0], None if parts[1] == "None" else parts[1], int(parts[2]), parts[3],
                   None if parts[4] == "None" else int(parts[4]))
            v = vec(key, r["population"])
            absci.append({"condition": c, "population": r["population"], "n": len(v), "id_recall": float(v.mean()), "ci95": ci(v)})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparisons.json").write_text(json.dumps({"comparisons": rows, "descriptive_a3_stopped_only": desc}, indent=1))
    with open(OUT / "comparisons.csv", "w", newline="") as f:
        cols = ["family", "family_size", "rq", "comparison", "population", "n", "B", "recall_reference", "recall_comparator",
                "mean_diff", "ci_low", "ci_high", "wins", "ties", "losses", "p_sign_raw", "p_sign_holm", "reference", "comparator"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "ci_low": r["ci95"][0], "ci_high": r["ci95"][1]})
    (OUT / "absolute_recall_ci.json").write_text(json.dumps(absci, indent=1))
    (OUT / "stats_manifest.json").write_text(json.dumps({
        "scores_verified_against": "analysis/scores/scoring_manifest.json", "scores_sha256": man["outputs_sha256"],
        "bootstrap": {"resamples": N_BOOT, "seed": SEED, "method": "percentile 95%", "rng": "numpy.random.default_rng",
                      "numpy": np.__version__, "resampling": "one index matrix per population size, shared across comparisons"},
        "sign_test": "exact two-sided binomial, ties dropped, tie tolerance 1e-12",
        "multiple_comparisons": "Holm on sign-test p within each family (D3); CIs unadjusted",
        "families": {k: len(v) for k, v in fams.items()}, "stats_py_sha256": sha(Path(__file__)),
        "outputs_sha256": {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.name != "stats_manifest.json"}}, indent=1))
    print(f"{len(rows)} comparisons ({sum(len(v) for v in fams.values())} in {len(fams)} Holm families), "
          f"{len(desc)} descriptive, {len(absci)} absolute CIs")


if __name__ == "__main__":
    main()
