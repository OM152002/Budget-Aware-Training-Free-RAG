"""Resumable experiment driver for EXPERIMENT_PLAN.md. Adds checkpointing/dispatch only; every method is imported
unchanged (frozen). One JSONL record per question; gold is never read here (evaluation is a separate step).

Experiments (set; budgets; methods)
  E0        PILOT-50[:10]  B=4  M5 frozen SLM                     reproducibility check (done)
  E1        NEW-1654       B=4  M5                                PILOT-50 B=4 reused from the held-out run
  E2        TEST-1704      B=8  M0 seeds, M1 flat top-11, M2 SAGE one-shot (faithful no-dedup + dedup diag),
                                M4 SAGE-score greedy (B=1/2/4 by prefix truncation)
  E2r       TEST-1704      B=4  M3 random, seeds {0,1,2}          (B=1/2 by truncation)
  E2r8      SUB-300        B=8  M3 random, seeds {0,1,2}
  E3_B1/B2/B8 SUB-300   B=1/2/8 M5                                (B=4 = E1 + PILOT-50 records)
  E4a_B4    TEST-1704      B=4  A3 no adaptive stopping: M5 record reused unless it ended in SLM_STOP, else rerun
  E4a_B8    SUB-300        B=8  A3, source = E3_B8
  E4b       SUB-300        B=4  A1 no history
  E4c       SUB-300        B=8  A2 no budget input (B=1/2/4 by truncation)

Usage:
  python -P -s run_experiment.py <EXP> <run_dir> [--max-questions N]
  python -P -s run_experiment.py <EXP> <run_dir> --smoke-ids q1,q2 [--source-dir DIR]   (tiny validation only)
  python -P -s run_experiment.py --verify-frozen
Re-running the same command on the same run_dir resumes; any fingerprint/config mismatch aborts.
"""
import argparse
import importlib.metadata as md
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
for p in (HERE, PROJECT / "adaptive_traversal", PROJECT / "slm_controller", PROJECT / "traversal_adapter",
          PROJECT / "candidate_prefilter", PROJECT / "baselines"):
    sys.path.insert(0, str(p))
from runner import ResumableRun, sha256_dir, sha256_file  # noqa: E402
import question_sets  # noqa: E402

FROZEN_CODE = ["slm_controller/controller.py", "slm_controller/mlx_backend.py", "adaptive_traversal/traversal.py",
               "candidate_prefilter/sage_prefilter.py", "traversal_adapter/graph_seed_adapter.py",
               "baselines/fixed_traversal.py", "experiments/ablations.py",
               "sage/scripts/analysis/graph_enhanced_accuracy_analyzer.py"]
RUNNER_CODE = ["experiments/runner.py", "experiments/run_experiment.py", "experiments/question_sets.py"]
FROZEN_FILE = HERE / "FROZEN.json"
GRAPH_RUN = "graph_reconstruction/runs/docgraph_k200_q99_20261023/output"
ARTIFACT_FILES = [f"{GRAPH_RUN}/analysis_cache/knowledge_graph/unique_edges_20261023_053523.json",
                  f"{GRAPH_RUN}/indexing_filtered.csv",
                  "seed_retrieval/runs/seeds_fusionA_top100_seed0_runA/seeds_top100.csv",
                  "seed_retrieval/runs/seeds_fusionA_top100_seed0_runA/seeds_top100_scores.jsonl",
                  "experiments/question_sets.json"]
ARTIFACT_DIRS = ["sage_hf_artifacts/extracted/docs_chunks_1"]
LIBS = ["mlx", "mlx-lm", "torch", "sentence-transformers", "transformers", "numpy", "pandas"]
HELDOUT = PROJECT / "adaptive_traversal/runs/heldout50_k3_m20_B4_v2"
RUNS = HERE / "runs"

SLM = "M5_slm_full"
EXPERIMENTS = {
    "E0": {"method": SLM, "set": "PILOT-50", "slice": [0, 10], "B": 4},
    "E1": {"method": SLM, "set": "NEW-1654", "B": 4},
    "E2": {"method": "E2_bundle_M0_M1_M2_M4", "set": "TEST-1704", "B": 8},
    "E2r": {"method": "M3_random", "set": "TEST-1704", "B": 4, "random_seeds": [0, 1, 2]},
    "E2r8": {"method": "M3_random", "set": "SUB-300", "B": 8, "random_seeds": [0, 1, 2]},
    "E3_B1": {"method": SLM, "set": "SUB-300", "B": 1},
    "E3_B2": {"method": SLM, "set": "SUB-300", "B": 2},
    "E3_B8": {"method": SLM, "set": "SUB-300", "B": 8},
    "E4a_B4": {"method": "A3_no_adaptive_stop", "set": "TEST-1704", "B": 4,
               "sources": ["heldout:PILOT-50", "run:E1_slm_B4_new1654"]},
    "E4a_B8": {"method": "A3_no_adaptive_stop", "set": "SUB-300", "B": 8, "sources": ["run:E3_B8_slm_sub300"]},
    "E4b": {"method": "A1_no_history", "set": "SUB-300", "B": 4, "include_history": False},
    "E4c": {"method": "A2_no_budget_input", "set": "SUB-300", "B": 8, "include_budget": False},
}
# canonical run directories (used by the README commands and as E4a sources)
RUN_DIRS = {"E1": "E1_slm_B4_new1654", "E2": "E2_baselines_B8_test1704", "E2r": "E2r_random_B4_test1704",
            "E2r8": "E2r8_random_B8_sub300", "E3_B1": "E3_B1_slm_sub300", "E3_B2": "E3_B2_slm_sub300",
            "E3_B8": "E3_B8_slm_sub300", "E4a_B4": "E4a_B4_nostop_test1704", "E4a_B8": "E4a_B8_nostop_sub300",
            "E4b": "E4b_nohistory_B4_sub300", "E4c": "E4c_nobudget_B8_sub300"}


def code_hashes():
    return {"code": {p: sha256_file(PROJECT / p) for p in FROZEN_CODE},
            "runner_code": {p: sha256_file(PROJECT / p) for p in RUNNER_CODE}}


def fingerprints(model_repo, model_rev):
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(model_repo, revision=model_rev, local_files_only=True))
    return {**code_hashes(),
            "artifacts": {**{p: sha256_file(PROJECT / p) for p in ARTIFACT_FILES},
                          **{p: sha256_dir(PROJECT / p) for p in ARTIFACT_DIRS}},
            "model": {"repo": model_repo, "revision": model_rev,
                      "files": {f: sha256_file(snap / f) for f in ("model.safetensors", "config.json", "tokenizer.json",
                                                                   "chat_template.jinja")}},
            "libraries": {l: md.version(l) for l in LIBS},
            "python": sys.version.split()[0]}


def verify_frozen():
    """Abort unless all method + runner code still equals the frozen record (if one exists)."""
    if not FROZEN_FILE.exists():
        return None
    frozen = json.loads(FROZEN_FILE.read_text())
    cur = code_hashes()
    bad = [f"{p}" for grp in ("code", "runner_code") for p, h in frozen[grp].items() if cur[grp].get(p) != h]
    bad += [p for grp in ("code", "runner_code") for p in cur[grp] if p not in frozen[grp]]
    if bad:
        raise SystemExit(f"FROZEN code changed, refusing to run: {bad}")
    return frozen


# ------------------------------------------------------------------ method dispatch
def load_sources(spec, smoke_source):
    """M5 records keyed by question id, for A3 reuse (heldout PILOT-50 and/or complete runner runs)."""
    recs = {}
    smoke = [smoke_source if smoke_source.startswith("heldout:") else f"dir:{smoke_source}"] if smoke_source else None
    for src in (smoke or spec["sources"]):
        kind, val = src.split(":", 1)
        if kind == "heldout":
            steps = {}
            for l in open(HELDOUT / "steps.jsonl"):
                s = json.loads(l); steps.setdefault(s.pop("query_id"), []).append(s)
            for l in open(HELDOUT / "results.jsonl"):
                r = json.loads(l); r["steps"] = steps[r["query_id"]]; r.pop("eval", None)
                recs[r["query_id"]] = (r, "adaptive_traversal/runs/heldout50_k3_m20_B4_v2")
        else:
            d = RUNS / val if kind == "run" else Path(val)
            man = json.loads((d / "manifest.json").read_text())
            prog = json.loads((d / "progress.json").read_text())
            if kind == "run" and prog["remaining"] != 0:
                raise SystemExit(f"source run {d} is not complete ({prog['remaining']} remaining)")
            if man["method"] != SLM or man["config"]["B"] != spec["B"]:
                raise SystemExit(f"source run {d} is not M5 at B={spec['B']}")
            for l in open(d / "results.jsonl"):
                r = json.loads(l); recs[r["query_id"]] = (r, str(d.relative_to(PROJECT)) if d.is_relative_to(PROJECT) else str(d))
    return recs


def make_process(exp, spec, smoke_source):
    from controller import ControllerConfig, TraversalController
    from traversal import TraversalConfig, traverse
    state = {}

    def common():
        if "A" not in state:
            from graph_seed_adapter import GraphSeedAdapter
            from sage_prefilter import SagePrefilter
            A = GraphSeedAdapter()
            state.update(A=A, P=SagePrefilter(A, use_cache=False))
        return state["A"], state["P"]

    def slm(cfg_kwargs):
        if "C" not in state:
            from mlx_backend import MLXBackend
            state["C"] = TraversalController(MLXBackend(), ControllerConfig(**cfg_kwargs))
        return state["C"]

    tcfg = TraversalConfig(budget=spec["B"])
    ccfg_kwargs = {k: spec[k] for k in ("include_history", "include_budget") if k in spec}
    m = spec["method"]

    if m in (SLM, "A1_no_history", "A2_no_budget_input"):
        def process(q):
            A, P = common()
            return traverse(q, A, P, slm(ccfg_kwargs), tcfg)

    elif m == "A3_no_adaptive_stop":
        sources = load_sources(spec, smoke_source)

        def process(q):
            if q not in sources:
                raise SystemExit(f"no M5 source record for {q}")
            src, where = sources[q]
            if src["termination_reason"] != "SLM_STOP":        # trajectory provably identical to M5 -> reuse
                return {**src, "a3_source": "reused_M5", "a3_source_run": where}
            from ablations import NoStopController
            A, P = common()
            if "NS" not in state:
                state["NS"] = NoStopController(slm({}))
            ns = state["NS"]; ns.reset()
            r = traverse(q, A, P, ns, tcfg)
            return {**r, "a3_source": "rerun_no_stop", "a3_source_run": where,
                    "m5_termination": src["termination_reason"], "m5_explore_steps": src["explore_steps"],
                    "a3_suppressed_stops": ns.events, "a3_extra_reprompt_calls": ns.extra_calls,
                    "slm_calls_actual": r["slm_calls"] + ns.extra_calls}

    elif m == "M3_random":
        from fixed_traversal import traverse_fixed

        def process(q):
            A, P = common()
            return {"query_id": q, "budget": spec["B"],
                    "random": {str(s): traverse_fixed(q, A, P, "random", spec["B"], seed=s) for s in spec["random_seeds"]}}

    elif m == "E2_bundle_M0_M1_M2_M4":
        from fixed_traversal import traverse_fixed
        import time

        def process(q):
            A, P = common()
            seeds = [s.node.chunk_id for s in A.get_seeds(q, 3)[0]]
            t = time.perf_counter()
            raw = [n.node.chunk_id for n in A.get_neighbors(seeds)]          # Variant-B: no text dedup
            m2 = P.score(A.questions[q]["question"], raw)[:spec["B"]]
            m2_ms = (time.perf_counter() - t) * 1000
            m4 = traverse_fixed(q, A, P, "sage_score", spec["B"])
            return {"query_id": q, "budget": spec["B"], "M0_seeds": seeds,
                    "M1_flat_top": A.questions[q]["ranking"][:3 + spec["B"]],
                    "M2_faithful_top": [{"chunk_id": c, "score": s} for c, s in m2], "M2_faithful_latency_ms": round(m2_ms, 1),
                    "M2_dedup_top": m4["steps"][0]["top_m"][:spec["B"]] if m4["steps"] else [],
                    "M4": m4}
    else:
        raise ValueError(m)
    return process


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", nargs="?", choices=sorted(EXPERIMENTS)); ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--smoke-ids", default=None, help="comma-separated ids (subset of the set) for tiny validation")
    ap.add_argument("--source-dir", default=None, help="A3 only, with --smoke-ids: M5 source run directory")
    ap.add_argument("--verify-frozen", action="store_true")
    a = ap.parse_args()
    frozen = verify_frozen()
    if a.verify_frozen:
        print("frozen code verified" if frozen else "no FROZEN.json yet"); return
    if a.source_dir and not a.smoke_ids:
        raise SystemExit("--source-dir is only allowed together with --smoke-ids")
    spec = EXPERIMENTS[a.exp]

    from controller import ControllerConfig
    from traversal import TraversalConfig
    from mlx_backend import MODEL_REPO, MODEL_REVISION
    tcfg = TraversalConfig(budget=spec["B"])            # k=3, m=20, max_retries=1, dedup_text=True (frozen defaults)
    ccfg = ControllerConfig(**{k: spec[k] for k in ("include_history", "include_budget") if k in spec})
    qs = question_sets.load()["sets"][spec["set"]]
    qids = qs["ids"] if "slice" not in spec else qs["ids"][spec["slice"][0]:spec["slice"][1]]
    if a.smoke_ids:
        sm = a.smoke_ids.split(",")
        if not set(sm) <= set(qids):
            raise SystemExit("smoke ids must belong to the experiment's question set")
        qids = [q for q in qids if q in set(sm)]
    manifest = {"experiment": a.exp, "method": spec["method"],
                "config": {"k": tcfg.k, "m": tcfg.m, "B": tcfg.budget, "max_retries": tcfg.max_retries,
                           "dedup_text": tcfg.dedup_text, "prompt_version": ccfg.prompt_version,
                           "temperature": ccfg.temperature, "max_new_tokens": ccfg.max_new_tokens,
                           "max_passage_chars": ccfg.max_passage_chars, "include_history": ccfg.include_history,
                           "include_budget": ccfg.include_budget, "random_seeds": spec.get("random_seeds"),
                           "a3_sources": spec.get("sources") if not a.source_dir else [a.source_dir],
                           "prefilter": "SAGE _calculate_hybrid_similarities, cache-free"},
                "question_set": {"name": spec["set"], "slice": spec.get("slice"), "parent_ids_sha256": qs["ids_sha256"],
                                 "smoke": bool(a.smoke_ids)},
                "fingerprints": fingerprints(MODEL_REPO, MODEL_REVISION)}
    run = ResumableRun(a.run_dir, qids, manifest)
    process = make_process(a.exp, spec, a.source_dir)

    def logged(q):
        r = process(q)
        print(f"  {q} done", flush=True)
        return r

    print(json.dumps(run.run(logged, max_questions=a.max_questions)))


if __name__ == "__main__":
    main()
