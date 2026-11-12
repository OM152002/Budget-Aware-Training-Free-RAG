"""Checkpoint/resume tests with a mock (deterministic, model-free) experiment.

  T1 uninterrupted reference run
  T2 hard crash INSIDE a question (os._exit in a subprocess, record never written) + a simulated torn partial
     line appended to results.jsonl -> resume reruns only that question; final file byte-identical to T1
  T3 several crash/resume cycles, incl. max_questions stops -> byte-identical to T1, no duplicates
  T4 resuming a finished run processes nothing
  T5 manifest mismatch (config, code hash, question set) -> FingerprintMismatch, results untouched
  T6 duplicate IDs in results.jsonl -> CorruptResults
  T7 progress.json reports completed/remaining/last_completed and no in_progress at the end

Usage: python -P -s test_runner.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runner import CorruptResults, FingerprintMismatch, ResumableRun  # noqa: E402

QIDS = [f"q{i:03d}" for i in range(20)]
MANIFEST = {"experiment": "MOCK", "config": {"k": 3, "m": 20, "B": 4, "seed": 0},
            "fingerprints": {"code": {"x.py": "abc"}}, "question_set": {"name": "MOCK-20"}}


def mock_process(qid):
    h = hashlib.sha256(qid.encode()).hexdigest()
    return {"query_id": qid, "retrieved": [h[i:i + 8] for i in range(0, 32, 8)], "steps": [{"step": 1, "x": h[:4]}]}


CHILD = r"""
import os, sys
sys.path.insert(0, {here!r})
from runner import ResumableRun
from test_runner import QIDS, MANIFEST, mock_process
crash_at = {crash_at!r}
def proc(q):
    if q == crash_at:
        os._exit(17)          # hard crash mid-question: no record, no cleanup
    return mock_process(q)
ResumableRun({run_dir!r}, QIDS, MANIFEST).run(proc, log=lambda *a: None)
"""


def child_run(run_dir, crash_at):
    code = CHILD.format(here=str(HERE), crash_at=crash_at, run_dir=str(run_dir))
    return subprocess.run([sys.executable, "-P", "-s", "-c", code]).returncode


def main():
    tmp = Path(tempfile.mkdtemp(prefix="runner_test_"))
    checks = {}
    try:
        ref = tmp / "ref"
        ResumableRun(ref, QIDS, MANIFEST).run(mock_process, log=lambda *a: None)
        ref_bytes = (ref / "results.jsonl").read_bytes()

        # T2: crash inside q007, then a torn partial line, then resume
        a = tmp / "crash"
        rc = child_run(a, "q007")
        after_crash = [json.loads(l)["query_id"] for l in open(a / "results.jsonl")]
        prog = json.loads((a / "progress.json").read_text())
        with open(a / "results.jsonl", "ab") as f:
            f.write(b'{"query_id": "q007", "retrieved": ["dead')       # torn write, no newline
        ResumableRun(a, QIDS, MANIFEST).run(mock_process, log=lambda *a: None)
        log = [json.loads(l) for l in open(a / "resume_log.jsonl")]
        checks["T2 crash exit code seen"] = rc == 17
        checks["T2 only completed questions persisted before crash"] = after_crash == QIDS[:7]
        checks["T2 progress marked interrupted question"] = prog["in_progress"] == "q007" and prog["completed"] == 7
        checks["T2 resume reran interrupted question and truncated torn bytes"] = (
            log[-1]["interrupted_question_rerun"] == "q007" and log[-1]["torn_bytes_truncated"] > 0)
        checks["T2 final results byte-identical to uninterrupted run"] = (a / "results.jsonl").read_bytes() == ref_bytes

        # T3: multiple crashes + partial runs
        b = tmp / "multi"
        child_run(b, "q003")
        ResumableRun(b, QIDS, MANIFEST).run(mock_process, max_questions=4, log=lambda *a: None)
        child_run(b, "q012")
        child_run(b, "q012")                     # crash again on the same question
        ResumableRun(b, QIDS, MANIFEST).run(mock_process, log=lambda *a: None)
        ids = [json.loads(l)["query_id"] for l in open(b / "results.jsonl")]
        checks["T3 repeated crash/resume byte-identical, ordered, no duplicates"] = (
            (b / "results.jsonl").read_bytes() == ref_bytes and ids == QIDS)

        # T4: resume of finished run
        r = ResumableRun(ref, QIDS, MANIFEST).run(mock_process, log=lambda *a: None)
        checks["T4 finished run processes nothing"] = r["processed_now"] == 0 and (ref / "results.jsonl").read_bytes() == ref_bytes

        # T5: mismatches abort and leave results untouched
        for name, man, qids in [("config B", {**MANIFEST, "config": {**MANIFEST["config"], "B": 8}}, QIDS),
                                ("code hash", {**MANIFEST, "fingerprints": {"code": {"x.py": "zzz"}}}, QIDS),
                                ("question set", MANIFEST, QIDS[:-1])]:
            try:
                ResumableRun(ref, qids, man).run(mock_process, log=lambda *a: None)
                checks[f"T5 mismatch ({name}) aborts"] = False
            except FingerprintMismatch:
                checks[f"T5 mismatch ({name}) aborts"] = (ref / "results.jsonl").read_bytes() == ref_bytes

        # T6: duplicates detected
        d = tmp / "dup"
        shutil.copytree(ref, d)
        with open(d / "results.jsonl", "a") as f:
            f.write(json.dumps(mock_process("q000")) + "\n")
        try:
            ResumableRun(d, QIDS, MANIFEST).run(mock_process, log=lambda *a: None)
            checks["T6 duplicate IDs abort"] = False
        except CorruptResults:
            checks["T6 duplicate IDs abort"] = True

        # T7: progress file
        p = json.loads((a / "progress.json").read_text())
        checks["T7 progress complete"] = (p["completed"] == 20 and p["remaining"] == 0 and p["last_completed"] == "q019"
                                         and p["in_progress"] is None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("ALL_PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
