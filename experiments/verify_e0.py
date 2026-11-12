"""E0 verification: the resumable runner reproduces the original held-out SLM runs exactly (timing excluded),
the frozen code hashes equal those recorded in the held-out summary, and a real manifest mismatch aborts.

Usage: python -P -s verify_e0.py <E0_run_dir>
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
from runner import FingerprintMismatch, ResumableRun  # noqa: E402

HELD = PROJECT / "adaptive_traversal/runs/heldout50_k3_m20_B4_v2"
TIMING = {"latency_ms", "prefilter_ms", "slm_latency_ms", "prefilter_latency_ms", "total_latency_ms", "updated_at"}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in TIMING and k != "eval"}
    if isinstance(o, list):
        return [strip(x) for x in o]
    return o


def main():
    run = Path(sys.argv[1]).resolve()
    new = [json.loads(l) for l in open(run / "results.jsonl")]
    old = {json.loads(l)["query_id"]: json.loads(l) for l in open(HELD / "results.jsonl")}
    old_steps = {}
    for l in open(HELD / "steps.jsonl"):
        s = json.loads(l); old_steps.setdefault(s.pop("query_id"), []).append(s)
    manifest = json.loads((run / "manifest.json").read_text())
    rep = {"questions": len(new), "identical_excluding_timing": 0, "mismatches": []}
    for r in new:
        o = dict(old[r["query_id"]]); o["steps"] = old_steps[r["query_id"]]
        a, b = strip(r), strip(o)
        if a == b:
            rep["identical_excluding_timing"] += 1
        else:
            rep["mismatches"].append({"q": r["query_id"], "keys": sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))})
    # field-level spot summary
    rep["fields_compared"] = sorted(strip(new[0]).keys())
    rep["raw_outputs_identical"] = all(
        [c["raw_output"] for s in r["steps"] for c in s.get("calls", [])]
        == [c["raw_output"] for s in old_steps[r["query_id"]] for c in s.get("calls", [])] for r in new)
    rep["order_matches_pilot50_prefix"] = [r["query_id"] for r in new] == json.loads(
        (HELD / "questions.json").read_text())["question_ids"][:10]
    # frozen code hashes vs those recorded when the held-out pilot ran
    recorded = json.loads((HELD / "summary.json").read_text())["config"]["frozen_code_sha256"]
    current = manifest["fingerprints"]["code"]
    rep["frozen_code_hashes_match_heldout"] = {p: current.get(p) == h for p, h in recorded.items()}
    rep["manifest_fingerprints"] = {k: (len(v) if isinstance(v, dict) else v) for k, v in manifest["fingerprints"].items()}
    # real mismatch must abort without touching results
    before = (run / "results.jsonl").read_bytes()
    qids = [r["query_id"] for r in new]
    tampered = {**{k: v for k, v in manifest.items() if k != "created_at"}}
    tampered["config"] = {**tampered["config"], "B": 8}
    try:
        ResumableRun(run, qids, tampered).run(lambda q: {"query_id": q}, log=lambda *a: None)
        rep["tampered_config_aborts"] = False
    except FingerprintMismatch as e:
        rep["tampered_config_aborts"] = (run / "results.jsonl").read_bytes() == before
        rep["abort_message_first_line"] = str(e).splitlines()[1].strip()
    rep["ALL_PASS"] = (rep["identical_excluding_timing"] == len(new) == 10 and rep["raw_outputs_identical"]
                       and rep["order_matches_pilot50_prefix"] and all(rep["frozen_code_hashes_match_heldout"].values())
                       and rep["tampered_config_aborts"])
    (run / "e0_verification.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
