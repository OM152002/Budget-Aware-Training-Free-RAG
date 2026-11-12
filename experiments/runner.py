"""Crash-safe, resumable per-question experiment runner. Method-agnostic: it never changes what is computed.

Run directory layout:
  manifest.json      experiment, config, question-set identity, frozen code/artifact/model/library fingerprints
  results.jsonl      append-only; exactly one complete JSON record per finished question, flushed + fsync'd
  progress.json      completed / remaining / last_completed / in_progress (atomic replace on every update)
  resume_log.jsonl   one line per start/resume (incl. any interrupted question that will be rerun)

Guarantees:
  - a question is persisted only when its full record is written (line + "\n" + fsync); a torn final line from a
    crash mid-write is truncated away on resume, so that question is rerun from the beginning;
  - completed question IDs are skipped; a duplicate ID in results.jsonl aborts (never silently mixed);
  - questions are processed in the manifest's fixed order;
  - on resume the stored manifest must equal the current one (config, question set, all hashes), else abort.
"""
import hashlib
import json
import os
import time
from pathlib import Path


class FingerprintMismatch(RuntimeError):
    pass


class CorruptResults(RuntimeError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_dir(path) -> str:
    """Order-independent content hash of a directory tree (relative path + file hash per file, sorted)."""
    path, h = Path(path), hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(path)).encode()); h.update(b"\0"); h.update(sha256_file(p).encode())
    return h.hexdigest()


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def _atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _diff(a, b, prefix=""):
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k, "<missing>"), b.get(k, "<missing>")
        if isinstance(x, dict) and isinstance(y, dict):
            out += _diff(x, y, f"{prefix}{k}.")
        elif x != y:
            out.append(f"{prefix}{k}: stored={x!r} current={y!r}")
    return out


class ResumableRun:
    def __init__(self, run_dir, question_ids, manifest: dict):
        self.dir = Path(run_dir)
        self.qids = list(question_ids)
        if len(set(self.qids)) != len(self.qids):
            raise ValueError("question list contains duplicates")
        self.manifest = {**manifest, "question_set": {**manifest.get("question_set", {}),
                                                      "n": len(self.qids), "ids_sha256": ids_sha256(self.qids)}}
        self.results = self.dir / "results.jsonl"
        self.progress = self.dir / "progress.json"

    # ---------------- setup / verification ----------------
    def _verify_or_create_manifest(self):
        mpath = self.dir / "manifest.json"
        if mpath.exists():
            stored = json.loads(mpath.read_text())
            stored.pop("created_at", None)
            diffs = _diff(stored, self.manifest)
            if diffs:
                raise FingerprintMismatch("refusing to resume: manifest differs\n  " + "\n  ".join(diffs))
            return "resume"
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(mpath, {**self.manifest, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        return "new"

    def _load_completed(self):
        """Return (completed ids in file order, truncated_bytes). Truncates a torn final line."""
        if not self.results.exists():
            return [], 0
        data = self.results.read_bytes()
        cut = data.rfind(b"\n") + 1                      # end of the last complete line
        truncated = len(data) - cut
        if truncated:
            with open(self.results, "r+b") as f:
                f.truncate(cut); f.flush(); os.fsync(f.fileno())
        done = []
        for i, line in enumerate(data[:cut].splitlines()):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise CorruptResults(f"line {i + 1} is not valid JSON: {e}")
            done.append(rec["query_id"])
        if len(set(done)) != len(done):
            raise CorruptResults("duplicate question IDs in results.jsonl")
        unknown = set(done) - set(self.qids)
        if unknown:
            raise CorruptResults(f"results contain IDs outside the question set: {sorted(unknown)[:3]}")
        return done, truncated

    def _write_progress(self, done_set, last, in_progress):
        _atomic_write_json(self.progress, {"completed": len(done_set), "remaining": len(self.qids) - len(done_set),
                                           "total": len(self.qids), "last_completed": last,
                                           "in_progress": in_progress,
                                           "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    # ---------------- main loop ----------------
    def run(self, process_fn, max_questions=None, log=print):
        mode = self._verify_or_create_manifest()
        done, truncated = self._load_completed()
        prev = json.loads(self.progress.read_text()) if self.progress.exists() else {}
        interrupted = prev.get("in_progress") if prev.get("in_progress") not in done else None
        done_set = set(done)
        with open(self.dir / "resume_log.jsonl", "a") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": mode, "completed": len(done),
                                "interrupted_question_rerun": interrupted, "torn_bytes_truncated": truncated}) + "\n")
            f.flush(); os.fsync(f.fileno())
        log(f"[runner] {mode}: {len(done)}/{len(self.qids)} done"
            + (f"; rerunning interrupted {interrupted}" if interrupted else "")
            + (f"; truncated {truncated} torn bytes" if truncated else ""))
        last = done[-1] if done else None
        processed = 0
        with open(self.results, "a") as out:
            for qid in self.qids:
                if qid in done_set:
                    continue
                if max_questions is not None and processed >= max_questions:
                    break
                self._write_progress(done_set, last, qid)
                rec = process_fn(qid)
                if rec.get("query_id") != qid:
                    raise ValueError(f"process_fn returned record for {rec.get('query_id')!r}, expected {qid!r}")
                out.write(json.dumps(rec) + "\n"); out.flush(); os.fsync(out.fileno())
                done_set.add(qid); last = qid; processed += 1
                self._write_progress(done_set, last, None)
        self._write_progress(done_set, last, None)
        return {"completed": len(done_set), "remaining": len(self.qids) - len(done_set), "processed_now": processed}
