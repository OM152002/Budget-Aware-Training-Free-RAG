"""ANALYSIS_PLAN.md tables T1–T6, built ONLY from the canonical scores and statistics.

Inputs (hash-checked against their manifests before use):
  analysis/scores/{aggregates.json, per_question.jsonl}   (scoring_manifest.json)
  analysis/stats/{comparisons.json, absolute_recall_ci.json}   (stats_manifest.json)
No retrieval metric is recomputed. The only new arithmetic is grouping canonical per-question values (recall by post-hoc
category, hop-0 gold counts, descriptive budget-to-budget differences of canonical aggregates).

Frozen reporting decisions (2026-11-02): no binary significance label; each paired comparison shows Δ = M5 − method,
95% paired-bootstrap CI, W/T/L (W = M5 higher) and the Holm-adjusted sign-test p (raw p for supplementary rows).
ID Recall is primary; text-equivalent Recall is diagnostic only.

Outputs: analysis/tables/T1..T6 as .md (rounded) + .csv (full precision, with source keys), TABLES.md (all six),
tables_manifest.json, tables_validation.json.

Usage: .venv-sage/bin/python -P -s analysis/tables.py
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SC, ST, OUT = PROJECT / "analysis/scores", PROJECT / "analysis/stats", PROJECT / "analysis/tables"
POP_N = {"TEST-1704": 1704, "NEW-1654": 1654, "SUB-300": 300}
NAMES = {"M0": "M0 seeds only", "M1": "M1 flat top-(3+b)", "M2|faithful": "M2 SAGE one-shot (faithful)",
         "M2|dedup_diag": "M2 SAGE one-shot (dedup, diagnostic)", "M3_mean": "M3 random (mean of 3 seeds)",
         "M4": "M4 SAGE-score greedy", "M5": "**M5 SLM (frozen, full)**", "A1": "A1 no history",
         "A2": "A2 no budget input", "A3": "A3 no adaptive stop"}
CATS = ("all_gold_in_seeds", "reachable_top20", "reachable_filtered", "unreachable_1hop")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    for d, m in ((SC, "scoring_manifest.json"), (ST, "stats_manifest.json")):
        man = json.loads((d / m).read_text())
        bad = [k for k, v in man["outputs_sha256"].items() if sha(d / k) != v]
        if bad:
            raise SystemExit(f"{d.name}: files changed since they were produced: {bad}")
    st = json.loads((ST / "stats_manifest.json").read_text())
    sm = json.loads((SC / "scoring_manifest.json").read_text())
    if st["scores_sha256"] != sm["outputs_sha256"]:
        raise SystemExit("statistics were computed from different score files")


def ck(method, variant=None, B=4, comp="max_budget", seed=None):
    return f"{method}|{variant}|{B}|{comp}|{seed}"


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def fs(x):
    return "—" if x is None else f"{x:+.4f}"


def fp(p):
    return "—" if p is None else ("<0.0001" if p < 1e-4 else f"{p:.4f}")


def fci(c):
    return "—" if c is None else f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def fabs(c):
    return "" if c is None else f" [{c[0]:.4f}, {c[1]:.4f}]"


def num(x, d=1):
    """Counts: integers as integers; fractional values (M3 seed means) with d decimals."""
    if x is None:
        return "—"
    return str(int(round(x))) if float(x).is_integer() else f"{x:.{d}f}"


def md(title, notes, header, rows):
    s = [f"### {title}", ""] + ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    s += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    s += [""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(s)


def main():
    verify()
    AG = {}
    for a in json.loads((SC / "aggregates.json").read_text()):
        AG[(ck(a["method"], a["variant"], a["B"], a["comparison"], a["random_seed"]), a["population"])] = a
    STATS = json.loads((ST / "comparisons.json").read_text())
    CMP = {(c["comparator"], c["reference"], c["population"]): c for c in STATS["comparisons"]}
    ABS = {(a["condition"], a["population"]): a for a in json.loads((ST / "absolute_recall_ci.json").read_text())}
    PQ = [json.loads(l) for l in open(SC / "per_question.jsonl")]
    hop0 = {P: sum(x["gold_hop0"] for x in PQ if x["method"] == "M0" and x[f"in_{P}"]) for P in POP_N}
    ngold = {P: sum(x["n_gold"] for x in PQ if x["method"] == "M0" and x[f"in_{P}"]) for P in POP_N}

    csv_rows = defaultdict(list)       # table -> list of dict rows (full precision + source keys)
    sections = {}

    def label(key):
        m, v = key.split("|")[:2]
        return NAMES.get(f"{m}|{v}", NAMES.get(m, m))

    def cmp_for(key, P, ref=None):
        B = key.split("|")[2]
        ref = ref or ck("M5", None, int(B) if key.split("|")[0] != "M0" else 4, "max_budget")
        return CMP.get((key, ref, P))

    # ---------------------------------------------------------------- T1 ----
    t1 = []
    hdr = ["Method", "ID Recall [95% CI]", "Δ = M5 − method [95% CI]", "W/T/L", "Holm p (sign)", "Text Recall (diag.)",
           "Hit rate", "Questions gaining", "New gold", "Expansions", "SLM calls", "Tokens", "Latency (s)"]
    for P in ("TEST-1704", "NEW-1654"):
        rows = []
        for key in (ck("M0", B=0), ck("M1"), ck("M2", "faithful"), ck("M3_mean"), ck("M4"), ck("M5")):
            a, ab = AG[(key, P)], ABS.get((key, P))
            c = None if key.startswith("M5|") else CMP[(key, ck("M5"), P)]
            rec = f4(a["id_recall"]) + (f" ± {a['id_recall_seed_sd']:.4f} (seed SD)" if "id_recall_seed_sd" in a else "") + fabs(ab and ab["ci95"])
            rows.append([label(key), rec, fs(c and c["mean_diff"]) + (" " + fci(c["ci95"]) if c else ""),
                         f"{c['wins']}/{c['ties']}/{c['losses']}" if c else "—", fp(c and c["p_sign_holm"]),
                         f4(a["text_recall_diag"]), f4(a["hit_rate"]), num(a["questions_gaining"], 1), num(a["new_gold_total"], 1),
                         f"{a['expansions_mean']:.2f}", f"{a['slm_calls_mean']:.2f}" if "slm_calls_mean" in a else "0",
                         f"{a['total_tokens_mean']:.0f}" if "total_tokens_mean" in a else "0",
                         f"{a['total_latency_s_mean']:.1f}" if a["total_latency_s_mean"] is not None else "—"])
            csv_rows["T1"].append({"population": P, "B": 4, "condition": key, "id_recall": a["id_recall"],
                                   "id_recall_ci_low": ab and ab["ci95"][0], "id_recall_ci_high": ab and ab["ci95"][1],
                                   "mean_diff": c and c["mean_diff"], "ci_low": c and c["ci95"][0], "ci_high": c and c["ci95"][1],
                                   "wins": c and c["wins"], "ties": c and c["ties"], "losses": c and c["losses"],
                                   "p_sign_raw": c and c["p_sign_raw"], "p_sign_holm": c and c["p_sign_holm"],
                                   "text_recall_diag": a["text_recall_diag"], "hit_rate": a["hit_rate"],
                                   "questions_gaining": a["questions_gaining"], "new_gold_total": a["new_gold_total"],
                                   "expansions_mean": a["expansions_mean"], "slm_calls_mean": a.get("slm_calls_mean", 0),
                                   "total_tokens_mean": a.get("total_tokens_mean", 0), "total_latency_s_mean": a["total_latency_s_mean"],
                                   "n": a["n"]})
        t1.append(md(f"T1 ({P}{', PRIMARY' if P == 'TEST-1704' else ', never-previously-scored robustness set'}): main results, "
                     f"budget B = 4, maximum budget (n = {POP_N[P]})",
                     ["ID Recall is primary; Text Recall is diagnostic only. Δ, CI and W/T/L are paired per question; W = M5 higher.",
                      f"Holm correction within this table's family (m = 5, {P}). CIs are unadjusted; no significance label is assigned.",
                      "Baselines always use all 4 expansions (except when candidates run out); M5 may STOP earlier.",
                      "M3: per-question mean over random seeds {0,1,2}; ± is the SD of the three seed-level recalls.",
                      "Latency is wall time per question and only where measured exactly (D5): M4 at B=4 is a truncation of the "
                      "B=8 run, so its latency is not reported. M0/M1 add no retrieval cost beyond the shared seed retrieval.",
                      "M2 (dedup, diagnostic) is reported in the supplementary statistics, not here."],
                     hdr, rows))
    sections["T1"] = "\n".join(t1)

    # ---------------------------------------------------------------- T2 ----
    t2 = []
    hdr = ["Method (matched to M5's per-question expansions)", "ID Recall [95% CI]", "Δ = M5 − method [95% CI]", "W/T/L",
           "Holm p (sign)", "Text Recall (diag.)", "Expansions"]
    for P in ("TEST-1704", "NEW-1654"):
        rows = []
        m5 = AG[(ck("M5"), P)]
        rows.append([label(ck("M5")), f4(m5["id_recall"]) + fabs(ABS[(ck("M5"), P)]["ci95"]), "—", "—", "—",
                     f4(m5["text_recall_diag"]), f"{m5['expansions_mean']:.2f}"])
        for key in (ck("M1", comp="matched_M5_B4"), ck("M2", "faithful", comp="matched_M5_B4"), ck("M3_mean", comp="matched_M5_B4"),
                    ck("M4", comp="matched_M5_B4")):
            a, ab, c = AG[(key, P)], ABS[(key, P)], CMP[(key, ck("M5"), P)]
            rows.append([label(key), f4(a["id_recall"]) + fabs(ab["ci95"]), fs(c["mean_diff"]) + " " + fci(c["ci95"]),
                         f"{c['wins']}/{c['ties']}/{c['losses']}", fp(c["p_sign_holm"]), f4(a["text_recall_diag"]),
                         f"{a['expansions_mean']:.2f}"])
            csv_rows["T2"].append({"population": P, "condition": key, "id_recall": a["id_recall"], "mean_diff": c["mean_diff"],
                                   "ci_low": c["ci95"][0], "ci_high": c["ci95"][1], "wins": c["wins"], "ties": c["ties"],
                                   "losses": c["losses"], "p_sign_raw": c["p_sign_raw"], "p_sign_holm": c["p_sign_holm"],
                                   "text_recall_diag": a["text_recall_diag"], "expansions_mean": a["expansions_mean"], "n": a["n"]})
        t2.append(md(f"T2 ({P}): matched-expansion comparison, B = 4 (n = {POP_N[P]})",
                     ["Each baseline gets exactly the number of expansions M5 used on that question (fewer only where the "
                      "baseline ran out of candidates). Exact prefix truncation of the logged runs.",
                      f"Holm correction within this table's family (m = 4, {P}); exact two-sided sign test, ties dropped.",
                      "ID Recall primary; Text Recall diagnostic only."], hdr, rows))
    sections["T2"] = "\n".join(t2)

    # ---------------------------------------------------------------- T3 ----
    t3 = []
    hdr = ["Method", "B", "ID Recall [95% CI]", "Δ = M5 − method [95% CI]", "W/T/L", "Holm p (sign)", "Text Recall (diag.)",
           "Expansions", "Expansions / B", "Pre-filter calls", "SLM calls", "Tokens", "Latency (s)"]
    rows = []
    for B in (1, 2, 4, 8):
        for key in (ck("M0", B=0), ck("M1", B=B), ck("M2", "faithful", B=B), ck("M3_mean", B=B), ck("M4", B=B), ck("M5", B=B)):
            a, ab = AG[(key, "SUB-300")], ABS.get((key, "SUB-300"))
            c = None if key.startswith("M5|") else CMP[(key, ck("M5", B=B), "SUB-300")]
            rows.append([label(key), B, f4(a["id_recall"]) + (f" ± {a['id_recall_seed_sd']:.4f}" if "id_recall_seed_sd" in a else "")
                         + fabs(ab and ab["ci95"]), fs(c and c["mean_diff"]) + (" " + fci(c["ci95"]) if c else ""),
                         f"{c['wins']}/{c['ties']}/{c['losses']}" if c else "—", fp(c and c["p_sign_holm"]),
                         f4(a["text_recall_diag"]), f"{a['expansions_mean']:.2f}",
                         f"{a['expansions_mean'] / B:.2f}" if not key.startswith("M0") else "—",
                         f"{a['prefilter_calls_mean']:.2f}", f"{a['slm_calls_mean']:.2f}" if "slm_calls_mean" in a else "0",
                         f"{a['total_tokens_mean']:.0f}" if "total_tokens_mean" in a else "0",
                         f"{a['total_latency_s_mean']:.1f}" if a["total_latency_s_mean"] is not None else "—"])
            csv_rows["T3"].append({"panel": "a SUB-300", "population": "SUB-300", "B": B, "condition": key, "id_recall": a["id_recall"],
                                   "mean_diff": c and c["mean_diff"], "ci_low": c and c["ci95"][0], "ci_high": c and c["ci95"][1],
                                   "wins": c and c["wins"], "ties": c and c["ties"], "losses": c and c["losses"],
                                   "p_sign_holm": c and c["p_sign_holm"], "text_recall_diag": a["text_recall_diag"],
                                   "expansions_mean": a["expansions_mean"], "prefilter_calls_mean": a["prefilter_calls_mean"],
                                   "slm_calls_mean": a.get("slm_calls_mean", 0), "total_tokens_mean": a.get("total_tokens_mean", 0),
                                   "total_latency_s_mean": a["total_latency_s_mean"], "n": a["n"]})
    t3.append(md("T3a (SUB-300): budget sweep, every method at B ∈ {1, 2, 4, 8} (n = 300)",
                 ["M5 at B = 1/2/8 exists only on SUB-300; M5 at B = 4 on SUB-300 = 291 questions from E1 + 9 from PILOT-50.",
                  "Holm correction within each budget's family (m = 5 per B, SUB-300). CIs unadjusted; no significance label.",
                  "M0 (no expansion) is repeated in each block as the per-B family reference; its Δ differs because M5 differs.",
                  "SUB-300 resolves paired differences of roughly ≥ 0.04 (EXPERIMENT_PLAN §0); smaller Δ are inconclusive at this n.",
                  "Latency only where measured exactly (D5): M3 at B = 4 (E2r) and B = 8 (E2r8), M4 at B = 8, M2 faithful, all SLM rows.",
                  "ID Recall primary; Text Recall diagnostic only."], hdr, rows))
    rows = []
    for B in (1, 2, 4, 8):
        keys = [ck("M1", B=B), ck("M2", "faithful", B=B), ck("M4", B=B)] + ([ck("M3_mean", B=B)] if B <= 4 else [])
        if B == 4:
            keys.append(ck("M5"))
        for key in keys:
            a = AG[(key, "TEST-1704")]
            rows.append([label(key), B, f4(a["id_recall"]) + (f" ± {a['id_recall_seed_sd']:.4f}" if "id_recall_seed_sd" in a else ""),
                         f4(a["text_recall_diag"]), f"{a['expansions_mean']:.2f}", f"{a['prefilter_calls_mean']:.2f}",
                         f"{a['total_latency_s_mean']:.1f}" if a["total_latency_s_mean"] is not None else "—"])
            csv_rows["T3"].append({"panel": "b TEST-1704", "population": "TEST-1704", "B": B, "condition": key,
                                   "id_recall": a["id_recall"], "text_recall_diag": a["text_recall_diag"],
                                   "expansions_mean": a["expansions_mean"], "prefilter_calls_mean": a["prefilter_calls_mean"],
                                   "total_latency_s_mean": a["total_latency_s_mean"], "n": a["n"]})
    m0 = AG[(ck("M0", B=0), "TEST-1704")]
    rows.insert(0, [label(ck("M0", B=0)), 0, f4(m0["id_recall"]), f4(m0["text_recall_diag"]), "0.00", "0.00", "—"])
    t3.append(md("T3b (TEST-1704): baselines across budgets (n = 1,704), descriptive",
                 ["M5 exists on TEST-1704 only at B = 4 (paired tests in T1). M3 at B = 8 exists only on SUB-300 (T3a).",
                  "No paired tests in this panel (none preregistered for TEST-1704 at B ≠ 4).",
                  "ID Recall primary; Text Recall diagnostic only."],
                 ["Method", "B", "ID Recall", "Text Recall (diag.)", "Expansions", "Pre-filter calls", "Latency (s)"], rows))
    rows, prev = [], None
    for B in (1, 2, 4, 8):
        a = AG[(ck("M5", B=B), "SUB-300")]
        rows.append([B, f4(a["id_recall"]), fs(a["id_recall"] - prev["id_recall"]) if prev else "—", f"{a['expansions_mean']:.2f}",
                     f"{a['stop_rate']:.3f}", f"{a['slm_calls_mean']:.2f}", f"{a['total_tokens_mean']:.0f}",
                     f'{a["total_tokens_mean"] - prev["total_tokens_mean"]:+.0f}' if prev else "—",
                     f"{a['total_latency_s_mean']:.1f}"])
        csv_rows["T3"].append({"panel": "c M5 by budget (descriptive)", "population": "SUB-300", "B": B, "condition": ck("M5", B=B),
                               "id_recall": a["id_recall"], "recall_change_from_previous_B": a["id_recall"] - prev["id_recall"] if prev else None,
                               "expansions_mean": a["expansions_mean"], "stop_rate": a["stop_rate"], "slm_calls_mean": a["slm_calls_mean"],
                               "total_tokens_mean": a["total_tokens_mean"], "total_latency_s_mean": a["total_latency_s_mean"], "n": a["n"]})
        prev = a
    t3.append(md("T3c (SUB-300): M5 across budgets, descriptive only (n = 300)",
                 ["Changes between budgets are differences of canonical means; no cross-budget significance test (frozen decision 3)."],
                 ["B", "ID Recall", "Change vs previous B", "Expansions", "STOP rate", "SLM calls", "Tokens", "Token change", "Latency (s)"],
                 rows))
    sections["T3"] = "\n".join(t3)

    # ---------------------------------------------------------------- T4 ----
    hdr = ["Comparison", "Population", "B", "M5 ID Recall", "Ablation ID Recall [95% CI]", "Δ = M5 − ablation [95% CI]", "W/T/L",
           "p (sign)", "Text Recall abl. (diag.)", "Expansions M5 / abl.", "STOP rate M5 / abl.", "SLM calls M5 / abl.",
           "Tokens M5 / abl.", "Latency (s) M5 / abl."]
    rows = []
    spec = [("A1", 4, "SUB-300", True), ("A2", 1, "SUB-300", True), ("A2", 2, "SUB-300", True), ("A2", 4, "SUB-300", True),
            ("A2", 8, "SUB-300", True), ("A3", 4, "TEST-1704", True), ("A3", 8, "SUB-300", True),
            ("A3", 4, "NEW-1654", False), ("A3", 4, "SUB-300", False)]
    for m, B, P, fam in spec:
        key, ref = ck(m, B=B), ck("M5", B=B)
        a, r, ab, c = AG[(key, P)], AG[(ref, P)], ABS[(key, P)], CMP[(key, ref, P)]
        calls_a = a["slm_calls_actual_mean"] if m == "A3" else a["slm_calls_mean"]
        rows.append([label(key) + ("" if fam else " (supplementary)"), P, B, f4(r["id_recall"]), f4(a["id_recall"]) + fabs(ab["ci95"]),
                     fs(c["mean_diff"]) + " " + fci(c["ci95"]), f"{c['wins']}/{c['ties']}/{c['losses']}",
                     (fp(c["p_sign_holm"]) + " (Holm)") if fam else (fp(c["p_sign_raw"]) + " (raw)"),
                     f4(a["text_recall_diag"]), f"{r['expansions_mean']:.2f} / {a['expansions_mean']:.2f}",
                     f"{r['stop_rate']:.3f} / {a['stop_rate']:.3f}", f"{r['slm_calls_mean']:.2f} / {calls_a:.2f}",
                     f"{r['total_tokens_mean']:.0f} / {a['total_tokens_mean']:.0f}",
                     f"{r['total_latency_s_mean']:.1f} / {a['total_latency_s_mean']:.1f}"])
        csv_rows["T4"].append({"population": P, "B": B, "condition": key, "family_member": fam, "m5_id_recall": r["id_recall"],
                               "id_recall": a["id_recall"], "mean_diff": c["mean_diff"], "ci_low": c["ci95"][0], "ci_high": c["ci95"][1],
                               "wins": c["wins"], "ties": c["ties"], "losses": c["losses"], "p_sign_raw": c["p_sign_raw"],
                               "p_sign_holm": c["p_sign_holm"], "text_recall_diag": a["text_recall_diag"],
                               "expansions_m5": r["expansions_mean"], "expansions_abl": a["expansions_mean"],
                               "stop_rate_m5": r["stop_rate"], "stop_rate_abl": a["stop_rate"],
                               "slm_calls_m5": r["slm_calls_mean"], "slm_calls_abl": calls_a,
                               "tokens_m5": r["total_tokens_mean"], "tokens_abl": a["total_tokens_mean"],
                               "latency_m5": r["total_latency_s_mean"], "latency_abl": a["total_latency_s_mean"], "n": a["n"]})
    desc = STATS["descriptive_a3_stopped_only"]
    drows = [[d["population"], d["B"], d["n"], fs(d["mean_diff"]), f"{d['wins']}/{d['ties']}/{d['losses']}"] for d in desc]
    sections["T4"] = md("T4: ablations, M5 (full controller) vs A1 no history, A2 no budget input, A3 no adaptive stopping",
                        ["Holm correction within the ablation family (m = 7: A1 B4; A2 B1/2/4/8; A3 B4 TEST-1704; A3 B8 SUB-300). "
                         "Rows marked supplementary are outside the family and show the raw sign-test p.",
                         "A2 removes the budget VALUE from the prompt; the fixed v2 system prompt still says the remaining budget is a "
                         "maximum. A2 at B = 1/2/4 is an exact truncation of the B = 8 run (verified gold-free).",
                         "A3 SLM calls include the STOP-suppression re-prompts (D6); A3 tokens and latency include them too.",
                         "A2 at B = 1 and 2 differs from M5 on a single question each; reported as computed (frozen decision 4).",
                         "SUB-300 resolves paired differences of roughly ≥ 0.04; smaller Δ are inconclusive at this n.",
                         "ID Recall primary; Text Recall diagnostic only."], hdr, rows) + "\n" + md(
        "T4 (continued): M5 vs A3 restricted to the questions where M5 stopped (descriptive, no test)",
        ["On every other question A3 is identical to M5 (verified), so the full-population Δ comes only from these questions."],
        ["Population", "B", "Questions where M5 stopped", "Δ = M5 − A3", "W/T/L"], drows)
    for d in desc:
        csv_rows["T4"].append({"population": d["population"], "B": d["B"], "condition": "A3 on M5-stopped questions (descriptive)",
                               "mean_diff": d["mean_diff"], "wins": d["wins"], "ties": d["ties"], "losses": d["losses"], "n": d["n"]})

    # ---------------------------------------------------------------- T5 ----
    t5 = []
    for P in ("TEST-1704", "NEW-1654"):
        a, a3 = AG[(ck("M5"), P)], AG[(ck("A3"), P)]
        stop_rows = [["STOP rate", f"{a['stop_rate']:.3f} ({a['n_stop']} of {a['n']})"],
                     ["STOP at step 0 (no expansion)", str(a["stop_step0"])],
                     ["STOP when all gold already in evidence (post-hoc)", f"{a['stop_when_sufficient']} of {a['n_stop']}"],
                     ["Expansions saved by STOP, per question (A3 mean − M5 mean)", f"{a3['expansions_mean'] - a['expansions_mean']:.2f}"],
                     ["Terminations", ", ".join(f"{k} {v}" for k, v in a["terminations"].items())]]
        t5.append(md(f"T5a ({P}): M5 stopping behavior, B = 4 (n = {POP_N[P]})",
                     ["'All gold already in evidence' uses gold labels after the fact; the controller never sees gold."],
                     ["Measure", "Value"], stop_rows))
        rows = []
        for key in (ck("M1"), ck("M2", "faithful"), ck("M3_mean"), ck("M4"), ck("M5"), ck("A3")):
            x = AG[(key, P)]
            rows.append([label(key), num(x["gold_hop1_total"], 1), num(x["gold_hop2plus_total"], 1), num(x["new_gold_total"], 1),
                         f"{x['wasted_expansions_mean']:.2f}", f"{x['expansions_mean']:.2f}"])
            csv_rows["T5"].append({"panel": "hop depth + wasted", "population": P, "B": 4, "condition": key,
                                   "gold_hop0": hop0[P], "gold_hop1": x["gold_hop1_total"], "gold_hop2plus": x["gold_hop2plus_total"],
                                   "new_gold_total": x["new_gold_total"], "wasted_expansions_mean": x["wasted_expansions_mean"],
                                   "expansions_mean": x["expansions_mean"], "n": x["n"]})
        t5.append(md(f"T5b ({P}): discovered gold by hop depth and wasted expansions, B = 4 maximum budget",
                     [f"Hop 0 = gold already in the seeds: {hop0[P]} of {ngold[P]} gold passages (identical for every method).",
                      "Hop 1 = discovered gold that is a direct graph neighbor of a seed; hop 2+ = discovered gold that is not "
                      "(genuine multi-hop discovery for the traversal methods M3/M4/M5/A3).",
                      "M1 does not use the graph: its 'hop 2+' means 'not a 1-hop neighbor of the seeds', including passages "
                      "with no graph connection. M2 is one hop by construction.",
                      "Wasted expansions = expansions taken after the last gold passage was found (all expansions if none was found).",
                      "M3 counts are means over the three random seeds."],
                     ["Method", "Gold found at hop 1", "Gold found at hop 2+", "Total new gold", "Wasted expansions (mean)",
                      "Expansions (mean)"], rows))
        # recall by post-hoc category (grouping canonical per-question recall)
        grp = defaultdict(list)
        for x in PQ:
            if x[f"in_{P}"] and x["comparison"] == "max_budget" and (
                    (x["B"] == 4 and (x["method"], x["variant"]) in {("M1", None), ("M2", "faithful"), ("M3_mean", None),
                                                                     ("M4", None), ("M5", None), ("A3", None)})
                    or x["method"] == "M0"):
                grp[(ck(x["method"], x["variant"], x["B"]), x["category"])].append(x["id_recall"])
        keys = (ck("M0", B=0), ck("M1"), ck("M2", "faithful"), ck("M3_mean"), ck("M4"), ck("M5"), ck("A3"))
        ncat = {c: len(grp[(keys[0], c)]) for c in CATS}
        rows = []
        for key in keys:
            vals = {c: sum(grp[(key, c)]) / len(grp[(key, c)]) for c in CATS}
            rows.append([label(key)] + [f4(vals[c]) for c in CATS])
            csv_rows["T5"].append({"panel": "recall by category", "population": P, "B": 4, "condition": key,
                                   **{f"id_recall_{c}": vals[c] for c in CATS}, **{f"n_{c}": ncat[c] for c in CATS}})
        t5.append(md(f"T5c ({P}): ID Recall by post-hoc category, B = 4 maximum budget",
                     ["Categories (held-out definitions, D5), from the seeds' raw 1-hop neighbors and the step-1 top-20 pre-filter "
                      "list including text-duplicate members: all gold in seeds / some remaining gold in the top-20 / reachable in "
                      "one hop but filtered out of the top-20 / not reachable in one hop.",
                      "Category membership is defined post hoc with gold and is identical for every method."],
                     ["Method"] + [f"{c} (n = {ncat[c]})" for c in CATS], rows))
    sections["T5"] = "\n".join(t5)

    # ---------------------------------------------------------------- T6 ----
    rows = []
    spec = [("M5", 1, "SUB-300"), ("M5", 2, "SUB-300"), ("M5", 4, "TEST-1704"), ("M5", 4, "NEW-1654"), ("M5", 4, "SUB-300"),
            ("M5", 8, "SUB-300"), ("A1", 4, "SUB-300"), ("A2", 8, "SUB-300"), ("A3", 4, "TEST-1704"), ("A3", 8, "SUB-300")]
    for m, B, P in spec:
        a = AG[(ck(m, B=B), P)]
        calls = a["n"] * a["slm_calls_actual_mean"]
        rows.append([label(ck(m, B=B)), P, B, a["n"], f"{calls:.0f}", a["invalid_first_attempts"], a["retries"], a["controller_errors"],
                     f"{a['controller_errors'] / a['n']:.4f}", ", ".join(f"{k} {v}" for k, v in a["terminations"].items())])
        csv_rows["T6"].append({"population": P, "B": B, "condition": ck(m, B=B), "n": a["n"], "slm_calls_total": calls,
                               "invalid_first_attempts": a["invalid_first_attempts"], "retries": a["retries"],
                               "controller_errors": a["controller_errors"], "terminations": json.dumps(a["terminations"])})
    sections["T6"] = md("T6: controller reliability, all SLM runs, by budget",
                        ["SLM calls include A3's STOP-suppression re-prompts (D6).",
                         "A2 is shown for the B = 8 run only; its B = 1/2/4 rows are truncations of that same run, not separate runs.",
                         "M5 B = 4 rows overlap (SUB-300 and NEW-1654 are subsets of TEST-1704); they are not additive.",
                         "Invalid first attempt = the first reply at a step failed the parser; a retry is the single deterministic "
                         "re-prompt; CONTROLLER_ERROR = the retry also failed."],
                        ["Run", "Population", "B", "Questions", "SLM calls", "Invalid first attempts", "Retries", "CONTROLLER_ERROR",
                         "Error rate", "Terminations"], rows)

    # ---------------------------------------------------------------- write + validate ----
    OUT.mkdir(parents=True, exist_ok=True)
    for t, text in sections.items():
        (OUT / f"{t}.md").write_text(f"## {t}\n\n" + text)
        cols = []
        for r in csv_rows[t]:
            cols += [k for k in r if k not in cols]
        with open(OUT / f"{t}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(csv_rows[t])
    (OUT / "TABLES.md").write_text("# Tables T1–T6 (canonical scores and statistics; no interpretation)\n\n"
                                   + "\n".join(f"## {t}\n\n{s}" for t, s in sections.items()))

    val = validate(AG, CMP, ABS, csv_rows, PQ, hop0, ngold)
    (OUT / "tables_validation.json").write_text(json.dumps(val, indent=1))
    (OUT / "tables_manifest.json").write_text(json.dumps({
        "inputs_sha256": {**{f"scores/{k}": v for k, v in json.loads((SC / "scoring_manifest.json").read_text())["outputs_sha256"].items()},
                          **{f"stats/{k}": v for k, v in json.loads((ST / "stats_manifest.json").read_text())["outputs_sha256"].items()}},
        "tables_py_sha256": sha(Path(__file__)),
        "outputs_sha256": {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.name != "tables_manifest.json"}}, indent=1))
    for k, v in val.items():
        if isinstance(v, dict):
            print(("PASS " if v["PASS"] else "FAIL ") + k)
    print("ALL_PASS" if val["ALL_PASS"] else "FAILED")


def validate(AG, CMP, ABS, csv_rows, PQ, hop0, ngold):
    v = {}

    def chk(n, c, **d):
        v[n] = {"PASS": bool(c), **d}

    # every recall/Δ/CI/WTL/p cell equals its canonical source exactly
    bad = 0
    for t in ("T1", "T2", "T3", "T4"):
        for r in csv_rows[t]:
            if r["condition"].startswith("A3 on"):
                continue
            a = AG[(r["condition"], r["population"])]
            bad += abs(r["id_recall"] - a["id_recall"]) > 0 if "id_recall" in r else 0
            bad += r["n"] != a["n"] or r["n"] != POP_N[r["population"]]
            if r.get("mean_diff") is not None:
                ref = "M5|None|" + str(r["condition"].split("|")[2] if not r["condition"].startswith("M0") else r.get("B", 4)) + "|max_budget|None"
                c = CMP[(r["condition"], ref, r["population"])]
                bad += any(r[k] != c[k] for k in ("mean_diff", "wins", "ties", "losses", "p_sign_holm"))
                bad += (r["ci_low"], r["ci_high"]) != tuple(c["ci95"])
                bad += abs((AG[(ref, r["population"])]["id_recall"] - a["id_recall"]) - r["mean_diff"]) > 1e-12
    chk("all recall / Δ / CI / W-T-L / Holm p cells equal canonical sources", bad == 0, mismatches=bad)
    chk("population sizes", all(r["n"] == POP_N[r["population"]] for t in ("T1", "T2", "T3", "T6") for r in csv_rows[t]))
    chk("W + T + L = n for every paired row", all(r["wins"] + r["ties"] + r["losses"] == r["n"] for t in ("T1", "T2", "T3", "T4")
                                                  for r in csv_rows[t] if r.get("wins") is not None))
    chk("T4: Holm p present for family rows, absent for supplementary rows",
        all((r["p_sign_holm"] is not None) == r["family_member"] for r in csv_rows["T4"] if "family_member" in r))
    chk("T1/T2 cover both TEST-1704 and NEW-1654; T3a/T4 SUB-300 rows exist for every budget",
        {r["population"] for r in csv_rows["T1"]} == {"TEST-1704", "NEW-1654"} == {r["population"] for r in csv_rows["T2"]}
        and {r["B"] for r in csv_rows["T3"] if r["panel"] == "a SUB-300"} == {1, 2, 4, 8})
    chk("T5: hop1 + hop2+ = new gold for every method",
        all(abs(r["gold_hop1"] + r["gold_hop2plus"] - r["new_gold_total"]) < 1e-9 for r in csv_rows["T5"] if r["panel"] == "hop depth + wasted"))
    chk("T5: category sizes sum to the population", all(sum(r[f"n_{c}"] for c in CATS) == POP_N[r["population"]]
                                                       for r in csv_rows["T5"] if r["panel"] == "recall by category"))
    # category recall re-weighted by size reproduces the canonical overall recall
    ok = True
    for r in csv_rows["T5"]:
        if r["panel"] == "recall by category":
            tot = sum(r[f"id_recall_{c}"] * r[f"n_{c}"] for c in CATS) / POP_N[r["population"]]
            ok &= abs(tot - AG[(r["condition"], r["population"])]["id_recall"]) < 1e-12
    chk("T5: size-weighted category recall equals canonical overall recall", ok)
    chk("T5: all_gold_in_seeds category has recall 1.0 for every method",
        all(abs(r["id_recall_all_gold_in_seeds"] - 1.0) < 1e-12 for r in csv_rows["T5"] if r["panel"] == "recall by category"))
    chk("T6: zero-error claims match aggregates", all(
        r["invalid_first_attempts"] == AG[(r["condition"], r["population"])]["invalid_first_attempts"]
        and r["controller_errors"] == AG[(r["condition"], r["population"])]["controller_errors"] for r in csv_rows["T6"]))
    chk("ID Recall labelled primary and Text Recall labelled diagnostic in every table",
        all("ID Recall" in (OUT / f"T{i}.md").read_text() for i in range(1, 6))
        and all("diag" in (OUT / f"T{i}.md").read_text() for i in (1, 2, 3, 4)))
    chk("no binary significance label in any table",
        not any(w in (OUT / f"T{i}.md").read_text().lower() for i in range(1, 7) for w in ("significant", "✓", "*p", "sig.")))
    v["ALL_PASS"] = all(x["PASS"] for x in v.values() if isinstance(x, dict))
    return v


if __name__ == "__main__":
    main()
