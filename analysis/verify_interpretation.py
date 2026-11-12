"""Check that every number in RESULTS_INTERPRETATION.md maps to a frozen source.

Every numeric token in the document must appear in the frozen outputs (T1–T6 markdown, comparisons.json, aggregates.json,
F1–F5 plotted data) in the format the document uses, OR be one of the explicitly listed derived values below, each
recomputed here from those same frozen files. Structural numbers (RQ/table/figure ids, budgets, set names) are allowed.
Reads only; writes analysis/interpretation_verification.json.

Usage: .venv-sage/bin/python -P -s analysis/verify_interpretation.py
"""
import csv
import hashlib
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent.parent
DOC = (P / "RESULTS_INTERPRETATION.md").read_text()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fmts(x):
    out = set()
    if isinstance(x, bool) or x is None:
        return out
    if isinstance(x, int):
        out |= {str(x), f"{x:,}"}
    if isinstance(x, (int, float)):
        for d in (0, 1, 2, 3, 4):
            out |= {f"{x:.{d}f}", f"{abs(x):.{d}f}", f"{x:,.{d}f}"}
        out.add(f"{x * 100:.1f}")          # percentages
    return out


def walk(o, acc):
    if isinstance(o, dict):
        for v in o.values():
            walk(v, acc)
    elif isinstance(o, list):
        for v in o:
            walk(v, acc)
    else:
        acc |= fmts(o)


allowed = set()
for f in sorted((P / "analysis/tables").glob("T[1-6].md")):
    allowed |= set(re.findall(r"\d[\d,]*\.?\d*", f.read_text()))
walk(json.loads((P / "analysis/stats/comparisons.json").read_text()), allowed)
walk(json.loads((P / "analysis/stats/absolute_recall_ci.json").read_text()), allowed)
AG = json.loads((P / "analysis/scores/aggregates.json").read_text())
walk(AG, allowed)
for f in sorted((P / "analysis/figures").glob("F*_data.csv")):
    for row in csv.DictReader(open(f)):
        for v in row.values():
            try:
                allowed |= fmts(float(v)) | fmts(int(float(v))) if float(v).is_integer() else fmts(float(v))
            except ValueError:
                pass


def ag(m, B, pop, comp="max_budget", v=None):
    return next(a for a in AG if a["method"] == m and a["B"] == B and a["population"] == pop
                and a["comparison"] == comp and a["variant"] == v and a["random_seed"] is None)


C = json.loads((P / "analysis/stats/comparisons.json").read_text())
cmp_ = C["comparisons"]
m5, a3 = ag("M5", 4, "TEST-1704"), ag("A3", 4, "TEST-1704")
desc = {(d["population"], d["B"]): d for d in C["descriptive_a3_stopped_only"]}
d4 = desc[("TEST-1704", 4)]
sub_stops = {B: ag("M5", B, "SUB-300")["stop_step0"] for B in (1, 2, 4, 8)}
f4 = list(csv.DictReader(open(P / "analysis/figures/F4_data.csv")))
rank1 = []
for s in "1234":
    rows = [r for r in f4 if r["step"] == s]
    rank1.append(100 * int(rows[0]["count"]) / sum(int(r["count"]) for r in rows))
pq = [json.loads(l) for l in open(P / "analysis/scores/per_question.jsonl")]
unreach = [x["id_recall"] for x in pq if x["in_TEST-1704"] and x["category"] == "unreachable_1hop"
           and x["B"] == 4 and x["comparison"] == "max_budget" and x["method"] in ("M1", "M2", "M3_mean", "M4", "M5", "A3")]
by_cat = {}
for x in pq:
    if x["in_TEST-1704"] and x["category"] == "unreachable_1hop" and x["B"] == 4 and x["comparison"] == "max_budget":
        by_cat.setdefault((x["method"], x["variant"]), []).append(x["id_recall"])
cat_max = max(sum(v) / len(v) for k, v in by_cat.items() if k[0] in ("M0", "M1", "M2", "M3_mean", "M4", "M5", "A3"))
fam = [c for c in cmp_ if c["family"].startswith("RQ1")]
sub_pos = [c["p_sign_holm"] for c in cmp_ if c["family"].startswith("RQ3/T3") and c["ci95"][0] > 0 and c["p_sign_holm"] < 0.03]
dis = [c["p_sign_holm"] for c in cmp_ if c["family"].startswith("RQ3/T3") and c["ci95"][0] > 0 and c["p_sign_holm"] > 0.03]
nodiff = [c["p_sign_holm"] for c in cmp_ if c["family"].startswith("RQ3/T3") and c["comparison"] == "M5 vs M1"]
expB = [ag("M5", B, "SUB-300")["expansions_mean"] / B for B in (1, 2, 4, 8)]
newdiff = max(abs(ag(m, B, "TEST-1704", v=v)["id_recall"] - ag(m, B, "NEW-1654", v=v)["id_recall"])
              for m, B, v in (("M0", 0, None), ("M1", 4, None), ("M2", 4, "faithful"), ("M3_mean", 4, None), ("M4", 4, None), ("M5", 4, None)))
m4b8, m5b4 = ag("M4", 8, "SUB-300"), ag("M5", 4, "SUB-300")

DERIVED = {  # token in document -> (recomputed value, explanation); each must round to the token
    "18": (len(fam), "RQ1 comparisons: T1+T2 on TEST and NEW"),
    "0.01": (max(c["p_sign_holm"] for c in fam), "largest Holm p among the 18 RQ1 comparisons (< 0.01)"),
    "0.0006": (newdiff, "max |TEST − NEW| recall over T1 methods (also M5 − M1 at B=8 SUB, in stats)"),
    "36": (min(rank1), "lowest step-wise rank-1 share (%) in F4"),
    "42": (max(rank1), "highest step-wise rank-1 share (%) in F4"),
    "0.10": (cat_max, "max recall in unreachable_1hop category over T5c methods (≤ 0.10)"),
    "11.0": (100 * m5["stop_rate"], "M5 B=4 TEST stop rate, %"),
    "7": (100 * (a3["expansions_mean"] - m5["expansions_mean"]) / a3["expansions_mean"], "expansions saved by STOP, %"),
    "9": (100 * (a3["total_tokens_mean"] - m5["total_tokens_mean"]) / a3["total_tokens_mean"], "tokens saved by STOP, % (also 9 lost questions)"),
    "40": (100 * m5["stop_when_sufficient"] / m5["n_stop"], "share of stops with all gold present, %"),
    "113": (m5["n_stop"] - m5["stop_when_sufficient"], "stops without all gold present"),
    "104": (m5["n_stop"] - m5["stop_when_sufficient"] - d4["losses"], "of those, stops where A3 found nothing more"),
    "178": (d4["ties"], "stopped questions where A3 found nothing more"),
    "17": (min(sub_stops.values()), "fewest M5 step-0 stops on SUB-300 across B"),
    "25": (max(sub_stops.values()), "most M5 step-0 stops on SUB-300 across B"),
    "88": (100 * min(expB), "lowest expansions/B, %"),
    "94": (100 * max(expB), "highest expansions/B, %"),
    "0.0234": (max(sub_pos), "largest Holm p among SUB-300 comparisons where both tests agree on a positive Δ"),
    "0.0729": (min(dis), "smallest Holm p among SUB-300 disagreements"),
    "0.1686": (min(nodiff), "smallest Holm p for M5 vs M1 on SUB-300"),
    "0.0004": (m4b8["id_recall"] - m5b4["id_recall"], "M4 B=8 minus M5 B=4 recall, SUB-300"),
    "third": (m4b8["total_latency_s_mean"] / m5b4["total_latency_s_mean"], "latency ratio M4 B=8 / M5 B=4 (≈ 1/3)"),
    "0.04": (0.04, "SUB-300 resolution, EXPERIMENT_PLAN §0 (plan constant)"),
    "20": (100 * ag("M5", 8, "SUB-300")["stop_rate"], "upper end of stop share 11–20% (M5 B=8 SUB) / top-20 pre-filter"),
}
STRUCTURAL = {"1", "2", "3", "4", "5", "6", "8", "0", "95", "1704", "1654", "300", "24", "576", "4b", "2026", "11", "09",
              "100", "50"}

tokens = re.findall(r"(?<![A-Za-z])[−+-]?\d[\d,]*\.?\d*", DOC)
unmatched, used_derived = [], {}
for t in tokens:
    core = t.lstrip("−+-").rstrip(".,")
    if core in allowed or core.replace(",", "") in allowed or core in STRUCTURAL:
        continue
    if core in DERIVED:
        used_derived[core] = DERIVED[core]
        continue
    unmatched.append(t)

checks = {}
for tok, (val, why) in DERIVED.items():
    if tok == "third":
        ok = abs(val - 1 / 3) < 0.05
    elif tok in ("0.10",):
        ok = val <= 0.10
    elif tok == "0.01":
        ok = val < 0.01
    elif tok in ("20",):
        ok = round(val) == 20
    elif tok == "0.04":
        ok = True
    else:
        d = len(tok.split(".")[1]) if "." in tok else 0
        ok = round(val, d) == float(tok) or (d == 0 and int(val) == int(tok)) or (d == 0 and round(val) == int(tok))
    checks[tok] = {"PASS": bool(ok), "recomputed": val, "explanation": why}
def sg(x):
    return f"{x:+.4f}".replace("-", "−")


cited, wrong = 0, []
for c in cmp_:
    trip = f'{sg(c["mean_diff"])} [{sg(c["ci95"][0])}, {sg(c["ci95"][1])}]'
    for line in DOC.splitlines():
        if trip in line:
            cited += 1
            wl = f'{c["wins"]}/{c["ties"]}/{c["losses"]}'
            holm = c["p_sign_holm"]
            hs = "<0.0001" if holm is not None and holm < 1e-4 else (f"{holm:.4f}" if holm is not None else None)
            raw = f'{c["p_sign_raw"]:.4f}'
            # where the line also carries W/T/L or p, they must be this comparison's
            seg = re.split(r"[+−]\d\.\d{4} \[", line.split(trip, 1)[1])[0]   # cells after this Δ, up to the next Δ
            m = re.search(r"\d+/\d+/\d+", seg)
            if m and m.group(0) != wl:
                wrong.append((c["comparison"], c["population"], c["B"], "wtl"))
            pm = re.search(r"<0\.0001|\d\.\d{4}", seg.replace(wl, ""))
            if hs and pm and pm.group(0) not in (hs, raw):
                wrong.append((c["comparison"], c["population"], c["B"], "p"))
checks["Δ/CI strings attached to the right W/T/L and p"] = {"PASS": not wrong, "cited_rows": cited, "wrong": wrong}
rep = {"numeric_tokens": len(tokens), "unmatched_tokens": sorted(set(unmatched)),
       "derived_checks": checks, "doc_sha256": sha(P / "RESULTS_INTERPRETATION.md"),
       "ALL_PASS": not unmatched and all(c["PASS"] for c in checks.values())}
(P / "analysis/interpretation_verification.json").write_text(json.dumps(rep, indent=1))
print("numeric tokens:", len(tokens), "| unmatched:", sorted(set(unmatched)))
for k, c in checks.items():
    if not c["PASS"]:
        print("FAIL", k, c)
print("ALL_PASS" if rep["ALL_PASS"] else "FAILED")
