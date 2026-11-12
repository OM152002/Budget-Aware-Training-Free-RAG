"""T7 (APPENDIX, development set only): DEV-24, v1 -> v2 controller-interface changes.

DEV-24 is excluded from every primary number and from the canonical scores (analysis/scores/). It is scored here, once,
with the same approved evaluator (`baselines/run_baselines_heldout.py::evaluate`, imported unchanged; its hash is
checked against the approved integrity report first) into a SEPARATE appendix file. The pilots' old gold-derived
`eval` fields are dropped unread; the pilots' own recorded summaries are used only as a reproduction check.

DEV-24 is a stratified development sample (8 questions per stratum, strata defined with gold), and the v2 fixes were
made while looking at it. It is NOT representative and supports no test; T7 is descriptive.

Outputs: analysis/appendix_dev24/{dev24_scores.jsonl, T7.md, T7.csv, t7_validation.json, t7_manifest.json}
Usage: OMP_NUM_THREADS=1 PYTHONHASHSEED=0 .venv-sage/bin/python -P -s analysis/appendix_t7.py
"""
import csv
import hashlib
import inspect
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "analysis" / "appendix_dev24"
RUNS = {"v1": PROJECT / "adaptive_traversal/runs/pilot_k3_m20_B4", "v2": PROJECT / "adaptive_traversal/runs/pilot_k3_m20_B4_v2"}
for p in (PROJECT / "baselines", PROJECT / "traversal_adapter", PROJECT / "candidate_prefilter", PROJECT / "adaptive_traversal",
          PROJECT / "slm_controller"):
    sys.path.insert(0, str(p))
K = 3


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        r.pop("eval", None)
        out.append(r)
    return out


def main():
    import run_baselines_heldout as RB
    approved = json.loads((PROJECT / "analysis/integrity/integrity_report.json").read_text())["5_evaluator_hashes"]
    now = {"baselines/run_baselines_heldout.py": sha(PROJECT / "baselines/run_baselines_heldout.py"),
           "evaluate()_source_sha256": hashlib.sha256(inspect.getsource(RB.evaluate).encode()).hexdigest(),
           "adaptive_traversal/traversal.py (normalize_text)": sha(PROJECT / "adaptive_traversal/traversal.py")}
    if any(approved[k] != v for k, v in now.items()):
        raise SystemExit("evaluator hash differs from the approved integrity report")
    inputs_before = {str(f.relative_to(PROJECT)): sha(f) for d in RUNS.values() for f in sorted(d.iterdir()) if f.is_file()}
    from graph_seed_adapter import GraphSeedAdapter
    from traversal import normalize_text
    A = GraphSeedAdapter()
    dev = json.loads((PROJECT / "experiments/question_sets.json").read_text())["sets"]["DEV-24"]["ids"]
    strata = {q["query_id"]: q["stratum"] for q in json.loads((RUNS["v2"] / "comparison_vs_v1.json").read_text())["per_question"]}

    rows, agg = [], {}
    for ver, d in RUNS.items():
        recs = {r["query_id"]: r for r in load(d / "results.jsonl")}
        steps = {}
        for s in load(d / "steps.jsonl"):
            steps.setdefault(s["query_id"], []).append(s)
        assert set(recs) == set(dev), f"{ver}: question set is not DEV-24"
        rs = []
        for q in dev:
            r = recs[q]
            e = RB.evaluate(A, q, r["seed_nodes"], r["retrieved_nodes"])
            seen, dup = {normalize_text(A.node(c).content) for c in r["seed_nodes"]}, 0
            for c in r["retrieved_nodes"][K:]:
                t = normalize_text(A.node(c).content)
                dup += t in seen
                seen.add(t)
            calls = [c for s in steps.get(q, []) for c in s.get("calls", [])]
            first = [s["calls"][0] for s in steps.get(q, []) if s.get("calls")]
            row = {"version": ver, "query_id": q, "stratum": strata[q], "id_recall": e["id_recall"],
                   "id_seed_recall": e["id_seed_recall"], "text_recall_diag": e["text_recall"],
                   "gained": e["id_recall"] > e["id_seed_recall"], "n_new_gold": len(e["new_gold_ids"]),
                   "expansions": len(r["retrieved_nodes"]) - K, "duplicate_text_explorations": dup,
                   "termination_reason": r["termination_reason"], "slm_calls": r["slm_calls"], "calls_logged": len(calls),
                   "first_attempts": len(first), "invalid_first_attempts": sum(not c["valid"] for c in first),
                   "retries": r["retries"], "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
                   "total_latency_ms": r["total_latency_ms"]}
            rs.append(row)
        rows += rs
        n = len(rs)
        m = lambda f: sum(x[f] for x in rs) / n
        agg[ver] = {"n": n, "id_seed_recall": m("id_seed_recall"), "id_recall": m("id_recall"), "text_recall_diag": m("text_recall_diag"),
                    "questions_gaining": sum(x["gained"] for x in rs), "new_gold": sum(x["n_new_gold"] for x in rs),
                    "expansions_mean": m("expansions"), "explored_total": sum(x["expansions"] for x in rs),
                    "duplicate_text_explorations": sum(x["duplicate_text_explorations"] for x in rs),
                    "stop": sum(x["termination_reason"] == "SLM_STOP" for x in rs),
                    "controller_errors": sum(x["termination_reason"] == "CONTROLLER_ERROR" for x in rs),
                    "first_attempts": sum(x["first_attempts"] for x in rs),
                    "invalid_first_attempts": sum(x["invalid_first_attempts"] for x in rs), "retries": sum(x["retries"] for x in rs),
                    "slm_calls_mean": m("slm_calls"), "input_tokens_mean": m("input_tokens"), "output_tokens_mean": m("output_tokens"),
                    "total_latency_s_mean": m("total_latency_ms") / 1000,
                    "id_recall_by_stratum": {s: sum(x["id_recall"] for x in rs if x["stratum"] == s) / sum(x["stratum"] == s for x in rs)
                                             for s in ("reachable_top20", "reachable_filtered", "unreachable_1hop")}}

    # ---- validation: reproduce the pilots' own recorded numbers ------------------------------------------------
    rec = json.loads((RUNS["v2"] / "comparison_vs_v1.json").read_text())
    val = {}
    for ver, key in (("v1", "before"), ("v2", "after")):
        R, a = rec[key], agg[ver]
        chk = {"recall_seed": round(a["id_seed_recall"], 3) == R["recall_seed"],
               "recall_final": round(a["id_recall"], 3) == R["recall_final"],
               "invalid_first_attempts": f'{a["invalid_first_attempts"]}/{a["first_attempts"]}' == R["invalid_first_attempts"],
               "retries": a["retries"] == R["retries"], "stops": a["stop"] == R["termination"].get("SLM_STOP", 0),
               "duplicate_text_explorations": a["duplicate_text_explorations"] == R["duplicate_text_explorations"],
               "explored_nodes": a["explored_total"] == R["explored_nodes"], "new_gold": a["new_gold"] == R["new_gold_items"],
               "questions_gaining": a["questions_gaining"] == R["queries_with_new_gold"],
               "avg_slm_calls": round(a["slm_calls_mean"], 3) == R["avg_slm_calls"],
               "stratum_recall": all(round(a["id_recall_by_stratum"][s], 3) == R["recall_by_stratum"][s]["final"]
                                     for s in R["recall_by_stratum"])}
        val[f"{ver} reproduces the pilot's recorded summary"] = {"PASS": all(chk.values()), **chk}
    val["both versions = DEV-24 (24 questions), disjoint from TEST-1704"] = {"PASS": len(dev) == 24 and not set(dev) & set(
        json.loads((PROJECT / "experiments/question_sets.json").read_text())["sets"]["TEST-1704"]["ids"])}
    val["seed recall identical across versions"] = {"PASS": all(
        x["id_seed_recall"] == y["id_seed_recall"] for x, y in zip(rows[:24], rows[24:]))}
    inputs_after = {str(f.relative_to(PROJECT)): sha(f) for d in RUNS.values() for f in sorted(d.iterdir()) if f.is_file()}
    val["pilot run files unchanged"] = {"PASS": inputs_before == inputs_after}
    val["ALL_PASS"] = all(v["PASS"] for v in val.values())

    # ---- T7 ------------------------------------------------------------------------------------------------------
    v1, v2 = agg["v1"], agg["v2"]
    lines = [
        ("Prompt interface", "v1: evidence labeled E1.., candidates C1..", "v2: evidence unlabeled, candidates A1..; budget stated as a maximum"),
        ("Duplicate-text handling", "off", "on (same-text candidates collapsed; evidence-text duplicates dropped)"),
        ("ID Recall, seeds only", f'{v1["id_seed_recall"]:.4f}', f'{v2["id_seed_recall"]:.4f}'),
        ("ID Recall, final (primary)", f'{v1["id_recall"]:.4f}', f'{v2["id_recall"]:.4f}'),
        ("Text Recall, final (diagnostic)", f'{v1["text_recall_diag"]:.4f}', f'{v2["text_recall_diag"]:.4f}'),
        ("ID Recall, stratum reachable_top20 (n = 8)", f'{v1["id_recall_by_stratum"]["reachable_top20"]:.4f}', f'{v2["id_recall_by_stratum"]["reachable_top20"]:.4f}'),
        ("ID Recall, stratum reachable_filtered (n = 8)", f'{v1["id_recall_by_stratum"]["reachable_filtered"]:.4f}', f'{v2["id_recall_by_stratum"]["reachable_filtered"]:.4f}'),
        ("ID Recall, stratum unreachable_1hop (n = 8)", f'{v1["id_recall_by_stratum"]["unreachable_1hop"]:.4f}', f'{v2["id_recall_by_stratum"]["unreachable_1hop"]:.4f}'),
        ("Questions gaining gold over seeds", v1["questions_gaining"], v2["questions_gaining"]),
        ("New gold passages found", v1["new_gold"], v2["new_gold"]),
        ("Expansions per question", f'{v1["expansions_mean"]:.2f}', f'{v2["expansions_mean"]:.2f}'),
        ("Duplicate-text explorations (total)", v1["duplicate_text_explorations"], v2["duplicate_text_explorations"]),
        ("Invalid first attempts / first attempts", f'{v1["invalid_first_attempts"]}/{v1["first_attempts"]}', f'{v2["invalid_first_attempts"]}/{v2["first_attempts"]}'),
        ("Retries", v1["retries"], v2["retries"]),
        ("CONTROLLER_ERROR", v1["controller_errors"], v2["controller_errors"]),
        ("SLM_STOP (questions)", v1["stop"], v2["stop"]),
        ("SLM calls per question", f'{v1["slm_calls_mean"]:.2f}', f'{v2["slm_calls_mean"]:.2f}'),
        ("Input tokens per question", f'{v1["input_tokens_mean"]:.0f}', f'{v2["input_tokens_mean"]:.0f}'),
        ("Total latency per question (s)", f'{v1["total_latency_s_mean"]:.1f}', f'{v2["total_latency_s_mean"]:.1f}'),
    ]
    md = ["## T7 — APPENDIX / DEVELOPMENT SET ONLY: DEV-24, v1 → v2 interface changes", "",
          "**Not part of any primary result.** DEV-24 (24 questions) was used to develop the v2 fixes and is excluded from "
          "TEST-1704, NEW-1654 and SUB-300. It is a gold-stratified sample (8 questions per stratum), not a random sample, "
          "so it is not representative. Descriptive only; no statistical test. B = 4, k = 3, m = 20, same model, same seeds.", "",
          "| Measure | v1 (pilot_k3_m20_B4) | v2 (frozen; pilot_k3_m20_B4_v2) |", "|---|---|---|"]
    md += [f"| {a} | {b} | {c} |" for a, b, c in lines]
    md += ["", "- Scored with the same approved evaluator as the canonical scores, into a separate appendix file; the pilots' "
           "old `eval` fields were dropped unread. Both columns reproduce the pilots' own recorded summaries (t7_validation.json).",
           "- ID Recall is primary; Text Recall is diagnostic only.",
           "- A 'duplicate-text exploration' is an explored passage whose normalized text equals that of a passage already in the evidence.", ""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "T7.md").write_text("\n".join(md))
    with open(OUT / "T7.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measure", "v1", "v2"])
        w.writerows(lines)
    with open(OUT / "dev24_scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (OUT / "t7_aggregates.json").write_text(json.dumps(agg, indent=1))
    (OUT / "t7_validation.json").write_text(json.dumps(val, indent=1))
    (OUT / "t7_manifest.json").write_text(json.dumps({"evaluator_hashes_verified": now, "inputs_sha256": inputs_before,
                                                      "script_sha256": sha(Path(__file__)),
                                                      "outputs_sha256": {p.name: sha(p) for p in sorted(OUT.iterdir())
                                                                         if p.name != "t7_manifest.json"}}, indent=1))
    for k, v in val.items():
        if isinstance(v, dict):
            print(("PASS " if v["PASS"] else "FAIL ") + k, "" if v["PASS"] else {a: b for a, b in v.items() if b is False})
    print("ALL_PASS" if val["ALL_PASS"] else "FAILED")


if __name__ == "__main__":
    main()
