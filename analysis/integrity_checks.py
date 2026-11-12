"""ANALYSIS_PLAN.md §6: gold-free integrity / alignment checks. Reads no gold label and computes no metric.

Inputs: run manifests + results, the PILOT-50 held-out SLM run (its gold-derived `eval` field is never read),
the held-out baseline runs (retrieved nodes only), question_sets.json, FROZEN.json and the frozen controller code.
Output: analysis/integrity/integrity_report.json (+ printed summary). Exit code 1 if any check fails.

Usage: OMP_NUM_THREADS=1 PYTHONHASHSEED=0 .venv-sage/bin/python -P -s analysis/integrity_checks.py
"""
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "experiments" / "runs"
HELDOUT = PROJECT / "adaptive_traversal" / "runs" / "heldout50_k3_m20_B4_v2"
HELDOUT_BASE = PROJECT / "baselines" / "runs" / "heldout50_baselines" / "runs.jsonl"
OUT = PROJECT / "analysis" / "integrity"
for p in (PROJECT / "experiments", PROJECT / "slm_controller", PROJECT / "adaptive_traversal",
          PROJECT / "baselines", PROJECT / "traversal_adapter", PROJECT / "candidate_prefilter"):
    sys.path.insert(0, str(p))

FROZEN = json.loads((PROJECT / "experiments" / "FROZEN.json").read_text())
QS = json.loads((PROJECT / "experiments" / "question_sets.json").read_text())["sets"]
SETS = {k: v["ids"] for k, v in QS.items()}
GOLD_KEYS = {"eval"}          # gold-derived fields that must never be read


def load(path, strip=GOLD_KEYS):
    out = []
    for line in open(path):
        r = json.loads(line)
        for k in strip & r.keys():
            del r[k]
        out.append(r)
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def top_ids(step):
    return [c["chunk_id"] if isinstance(c, dict) else c for c in step.get("top_m", [])]


def m4_step1(e2rec):
    st = e2rec["M4"]["steps"]
    return st[0]["top_m"] if st else []


def ok(cond, **detail):
    return {"PASS": bool(cond), **detail}


def main():
    from run_experiment import fingerprints
    from mlx_backend import MODEL_REPO, MODEL_REVISION
    R = {d: load(RUNS / d / "results.jsonl") for d in FROZEN["canonical_run_dirs"].values()}
    M = {d: json.loads((RUNS / d / "manifest.json").read_text()) for d in FROZEN["canonical_run_dirs"].values()}
    pilot = load(HELDOUT / "results.jsonl")
    pilot_steps = {}
    for s in load(HELDOUT / "steps.jsonl"):
        pilot_steps.setdefault(s["query_id"], []).append(s)
    for r in pilot:
        r["steps"] = sorted(pilot_steps.get(r["query_id"], []), key=lambda s: s["step"])
    E2 = {r["query_id"]: r for r in R["E2_baselines_B8_test1704"]}
    M0 = {q: r["M0_seeds"] for q, rec in [(q, E2[q]) for q in E2] for r in [rec]}
    rep = {}

    # ---- 1. fingerprints -------------------------------------------------------------------------------------
    fp_now = fingerprints(MODEL_REPO, MODEL_REVISION)
    fp1 = {d: [k for k in fp_now if m["fingerprints"].get(k) != fp_now[k]] for d, m in M.items()}
    vf = subprocess.run([sys.executable, "-P", "-s", str(PROJECT / "experiments" / "run_experiment.py"), "--verify-frozen"],
                        capture_output=True, text=True, cwd=PROJECT)
    held_cfg = json.loads((HELDOUT / "summary.json").read_text())["config"]["frozen_code_sha256"]
    rep["1_fingerprints"] = {
        "manifests_match_current": ok(all(not v for v in fp1.values()), mismatches=fp1, runs=len(M)),
        "verify_frozen": ok(vf.returncode == 0 and "frozen code verified" in vf.stdout, stdout=vf.stdout.strip()[-200:]),
        "pilot50_code_hashes_match_FROZEN": ok(all(FROZEN["code"].get(k) == v for k, v in held_cfg.items()),
                                              files=len(held_cfg)),
        "pilot50_config": ok((held_cfg is not None) and all(json.loads((HELDOUT / "summary.json").read_text())["config"][k] == v
                             for k, v in {"k": 3, "m": 20, "budget": 4, "max_retries": 1, "dedup_text": True,
                                          "prompt_version": "v2", "temperature": 0.0}.items())),
    }

    # ---- 2. question sets / order ----------------------------------------------------------------------------
    order = {}
    for exp, d in FROZEN["canonical_run_dirs"].items():
        s = FROZEN["experiments"][exp]["set"]
        ids = [r["query_id"] for r in R[d]]
        order[d] = ok(ids == SETS[s] and len(set(ids)) == len(ids) and M[d]["question_set"]["name"] == s,
                      set=s, n=len(ids))
    pilot_ids = [r["query_id"] for r in pilot]
    test, new, p50, sub, dev = (set(SETS[k]) for k in ("TEST-1704", "NEW-1654", "PILOT-50", "SUB-300", "DEV-24"))
    sub_m5_b4 = [q for q in SETS["SUB-300"] if q in {r["query_id"] for r in R["E1_slm_B4_new1654"]} | set(pilot_ids)]
    rep["2_question_sets"] = {
        "run_order_equals_frozen_set": order,
        "pilot50_records_equal_PILOT50_order": ok(pilot_ids == SETS["PILOT-50"]),
        "NEW_union_PILOT_equals_TEST_disjoint": ok(new | p50 == test and not new & p50 and len(test) == 1704),
        "SUB300_subset_of_TEST": ok(sub <= test, overlap_pilot50=len(sub & p50)),
        "DEV24_disjoint_from_every_run": ok(not dev & test and not dev & set(pilot_ids)),
        "M5_B4_covers_SUB300": ok(len(sub_m5_b4) == 300, from_E1=len(sub - p50), from_PILOT50=len(sub & p50)),
        "no_gold_field_in_experiment_runs": ok(not any(GOLD_KEYS & set(json.loads(open(RUNS / d / "results.jsonl").readline()))
                                                       for d in R)),
    }

    # ---- 3. structure + prefix / alignment ---------------------------------------------------------------------
    def fixed_chain_ok(r, seeds, greedy):
        sel = [s["selected"] for s in r["steps"]]
        return (r["seed_nodes"] == seeds and r["retrieved_nodes"] == seeds + sel and r["explore_steps"] == len(sel)
                and all(s["selected"] == s["top_m"][s["selected_rank"] - 1] for s in r["steps"])
                and (not greedy or all(s["selected_rank"] == 1 for s in r["steps"])))

    def slm_chain_ok(r, seeds):
        expl = [h[2] for h in r["traversal_history"] if h[1] == "EXPLORE"]
        return (r["seed_nodes"] == seeds and r["retrieved_nodes"] == seeds + expl
                and r["explore_steps"] == len(expl) == r["budget_used"] <= r["budget_allocated"])

    slm_runs = {"PILOT-50 (M5 B4)": pilot, **{d: R[d] for d in ["E1_slm_B4_new1654", "E3_B1_slm_sub300", "E3_B2_slm_sub300",
                "E3_B8_slm_sub300", "E4a_B4_nostop_test1704", "E4a_B8_nostop_sub300", "E4b_nohistory_B4_sub300",
                "E4c_nobudget_B8_sub300"]}}
    step1 = {}
    for name, recs in slm_runs.items():
        bad_seed = [r["query_id"] for r in recs if not slm_chain_ok(r, M0[r["query_id"]])]
        steps_present = [r for r in recs if r["steps"]]
        mism = [r["query_id"] for r in steps_present if top_ids(r["steps"][0]) != m4_step1(E2[r["query_id"]])] \
            if name != "E4a_B4_nostop_test1704" else None
        no_steps = [r["query_id"] for r in recs if not r["steps"]]
        step1[name] = ok(not bad_seed and not mism, n=len(recs), seeds_or_chain_bad=bad_seed[:5],
                         step1_topm_mismatch_vs_M4=(mism[:5] if mism is not None else "n/a (E4a: steps checked via source runs)"),
                         questions_without_steps=no_steps)
    e2bad = [q for q, r in E2.items() if not (
        fixed_chain_ok(r["M4"], r["M0_seeds"], True) and len(r["M1_flat_top"]) == 11 and r["M1_flat_top"][:3] == r["M0_seeds"]
        and len(r["M2_faithful_top"]) <= 8 and not {x["chunk_id"] for x in r["M2_faithful_top"]} & set(r["M0_seeds"])
        and len({x["chunk_id"] for x in r["M2_faithful_top"]}) == len(r["M2_faithful_top"])
        and all(a["score"] >= b["score"] for a, b in zip(r["M2_faithful_top"], r["M2_faithful_top"][1:]))   # ranked -> prefix-truncatable
        and r["M2_dedup_top"] == (r["M4"]["steps"][0]["top_m"][:8] if r["M4"]["steps"] else []))]
    m3bad = [r["query_id"] for d in ("E2r_random_B4_test1704", "E2r8_random_B8_sub300") for r in R[d]
             if not all(fixed_chain_ok(v, M0[r["query_id"]], False) for v in r["random"].values())]
    m3_step1 = [r["query_id"] for d in ("E2r_random_B4_test1704", "E2r8_random_B8_sub300") for r in R[d]
                for v in r["random"].values() if v["steps"] and v["steps"][0]["top_m"] != m4_step1(E2[r["query_id"]])]
    E2r = {r["query_id"]: r for r in R["E2r_random_B4_test1704"]}
    m3_prefix = [r["query_id"] for r in R["E2r8_random_B8_sub300"] for s in "012"
                 if r["random"][s]["retrieved_nodes"][:3 + E2r[r["query_id"]]["random"][s]["explore_steps"]]
                 != E2r[r["query_id"]]["random"][s]["retrieved_nodes"]]
    # cross-run determinism: held-out baseline runs at B=4 (PILOT-50) vs E2 / E2r prefixes
    hb = [json.loads(l) for l in open(HELDOUT_BASE)]
    hb4 = [r for r in hb if r.get("comparison") == "max_budget"]
    m4x = [r["query_id"] for r in hb4 if r["policy"] == "sage_score"
           and r["retrieved_nodes"] != E2[r["query_id"]]["M4"]["retrieved_nodes"][:3 + r["explore_steps"]]]
    m3x = [r["query_id"] for r in hb4 if r["policy"] == "random"
           and r["retrieved_nodes"] != E2r[r["query_id"]]["random"]["0"]["retrieved_nodes"][:3 + r["explore_steps"]]]
    hb_matched = [r for r in hb if r.get("comparison") == "matched_expansion"]
    m4_matched = [r["query_id"] for r in hb_matched if r["policy"] == "sage_score"
                  and r["retrieved_nodes"] != E2[r["query_id"]]["M4"]["retrieved_nodes"][:3 + r["explore_steps"]]]
    # A3 vs its source: reused identical; reruns equal the source up to the source's STOP
    src = {r["query_id"]: r for r in R["E1_slm_B4_new1654"] + pilot}
    src8 = {r["query_id"]: r for r in R["E3_B8_slm_sub300"]}

    def a3_check(recs, S):
        bad = []
        for r in recs:
            s = S[r["query_id"]]
            if r["a3_source"] == "reused_M5":
                good = s["termination_reason"] != "SLM_STOP" and r["retrieved_nodes"] == s["retrieved_nodes"]
            else:
                good = (s["termination_reason"] == "SLM_STOP" and r["termination_reason"] != "SLM_STOP"
                        and r["retrieved_nodes"][:len(s["retrieved_nodes"])] == s["retrieved_nodes"])
            if not good:
                bad.append(r["query_id"])
        return bad
    a3b4, a3b8 = a3_check(R["E4a_B4_nostop_test1704"], src), a3_check(R["E4a_B8_nostop_sub300"], src8)
    # SLM determinism across budgets: M5 B=1/2 first choice vs B=4/B=8 first choice is NOT expected equal (budget
    # text differs), so it is only reported, not checked.
    rep["3_alignment"] = {
        "slm_runs_seeds_chain_and_step1_topm_equal_M4": step1,
        "E2_internal (M4 greedy chain, M1 prefix, M2 faithful/dedup)": ok(not e2bad, bad=e2bad[:5], n=len(E2)),
        "M3_chains_valid": ok(not m3bad, bad=m3bad[:5]),
        "M3_step1_topm_equal_M4": ok(not m3_step1, bad=m3_step1[:5]),
        "M3_B4_is_prefix_of_B8_on_SUB300": ok(not m3_prefix, bad=m3_prefix[:5], checked=300 * 3),
        "M4_B4_heldout_run_is_prefix_of_E2_B8": ok(not m4x, checked=sum(r["policy"] == "sage_score" for r in hb4)),
        "M4_matched_heldout_run_is_prefix_of_E2_B8": ok(not m4_matched, checked=sum(r["policy"] == "sage_score" for r in hb_matched)),
        "M3_seed0_heldout_run_equals_E2r_seed0": ok(not m3x, checked=sum(r["policy"] == "random" for r in hb4)),
        "A3_B4_vs_sources": ok(not a3b4, bad=a3b4[:5], reruns=sum(r["a3_source"] == "rerun_no_stop" for r in R["E4a_B4_nostop_test1704"])),
        "A3_B8_vs_E3_B8": ok(not a3b8, bad=a3b8[:5], reruns=sum(r["a3_source"] == "rerun_no_stop" for r in R["E4a_B8_nostop_sub300"])),
    }

    # ---- 4. A2 truncation validity (code level, no SLM call) ----------------------------------------------------
    from controller import ControllerConfig, HistoryStep, Passage, build_prompt
    q = "Which team did the player join?"
    ev = [Passage("e1", "Seed passage one."), Passage("e2", "Seed passage two.")]
    ca = [Passage("c1", "Candidate one."), Passage("c2", "Candidate two.")]
    hi = [HistoryStep(1, "EXPLORE", "e2")]
    nob = ControllerConfig(include_budget=False)
    texts = {b: build_prompt(q, ev, ca, hi, b, nob)[:2] for b in range(0, 9)}
    withb = {b: build_prompt(q, ev, ca, hi, b, ControllerConfig())[:2] for b in (1, 8)}
    import traversal as T
    src_trav = inspect.getsource(T.traverse)
    rep["4_A2_truncation"] = {
        "no_budget_prompt_identical_for_B_0_to_8": ok(len(set(texts.values())) == 1),
        "with_budget_prompt_differs (sanity)": ok(withb[1] != withb[8]),
        "no_digit_budget_line_when_disabled": ok("Remaining budget (maximum" not in texts[4][1]),
        "note": ("The fixed v2 SYSTEM prompt still says 'The remaining budget is a maximum, not a target' in A2; it carries "
                 "no budget value, so it does not break truncation. A2 = 'no budget value', not 'no mention of budget'."),
        "traverse_passes_only_remaining_budget_to_controller": ok("remaining_budget" in src_trav),
    }

    # ---- 5. evaluation code hashes (recorded before any gold is read) -----------------------------------------
    import run_baselines_heldout as RB
    rep["5_evaluator_hashes"] = {
        "baselines/run_baselines_heldout.py": sha(PROJECT / "baselines" / "run_baselines_heldout.py"),
        "evaluate()_source_sha256": hashlib.sha256(inspect.getsource(RB.evaluate).encode()).hexdigest(),
        "adaptive_traversal/traversal.py (normalize_text)": sha(PROJECT / "adaptive_traversal" / "traversal.py"),
        "normalize_text_is_frozen_file": ok(sha(PROJECT / "adaptive_traversal" / "traversal.py")
                                            == FROZEN["code"]["adaptive_traversal/traversal.py"]),
        "analysis/integrity_checks.py": sha(Path(__file__)),
        "ANALYSIS_PLAN.md": sha(PROJECT / "ANALYSIS_PLAN.md"),
    }

    def collect(node, path=""):
        if isinstance(node, dict):
            if "PASS" in node:
                yield path, node["PASS"]
            for k, v in node.items():
                yield from collect(v, f"{path}/{k}")
    flat = list(collect(rep))
    rep["ALL_PASS"] = all(v for _, v in flat)
    rep["n_checks"], rep["failed"] = len(flat), [p for p, v in flat if not v]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "integrity_report.json").write_text(json.dumps(rep, indent=2))
    for p, v in flat:
        print(("PASS " if v else "FAIL ") + p)
    print(f"{'ALL_PASS' if rep['ALL_PASS'] else 'FAILED'} ({len(flat)} checks)")
    sys.exit(0 if rep["ALL_PASS"] else 1)


if __name__ == "__main__":
    main()
