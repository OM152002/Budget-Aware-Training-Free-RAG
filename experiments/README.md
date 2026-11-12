# Resumable experiment runner: FROZEN

**Status: FROZEN.** `FROZEN.json` (sha256 `b03749456ef8…`) records the hashes of all method code, runner code, artifacts, model files, libraries and the experiment registry. `run_experiment.py` refuses to run if any frozen method or runner file has changed (`--verify-frozen` checks this without running anything). Every run directory's `manifest.json` repeats these fingerprints, and resuming aborts on any mismatch.

| File | Role |
|---|---|
| `runner.py` | `ResumableRun`: append-only, fsync'd `results.jsonl`; atomic `progress.json`; `resume_log.jsonl`; manifest verification; torn-line truncation; duplicate detection |
| `run_experiment.py` | Experiment registry and dispatch (E0, E1, E2, E2r, E2r8, E3_B1/B2/B8, E4a_B4/B8, E4b, E4c) |
| `ablations.py` | A3 `NoStopController` (wrapper around the unmodified controller; logs suppressed STOPs and extra calls) |
| `question_sets.py` / `.json` | Frozen DEV-24, TEST-1704, PILOT-50, NEW-1654, SUB-300 |
| `test_runner.py`, `test_ablations.py` | Crash/resume and A3 tests |
| `smoke_all.sh`, `verify_smoke.py` | Tiny smoke runs of every definition (stop after 1 question, then resume) and schema checks. Output is in `smoke/`, not reported |
| `verify_e0.py` | E0 reproducibility check (done) |

## Experiments (EXPERIMENT_PLAN.md)
| Exp | Set | B | Method(s) | Notes |
|---|---|---|---|---|
| E1 | NEW-1654 | 4 | M5 frozen SLM | PILOT-50 B=4 comes from the held-out run (reused) |
| E2 | TEST-1704 | 8 | M0 seeds, M1 flat top-(3+b), M2 SAGE one-shot (faithful no-dedup + dedup diagnostic), M4 SAGE-score greedy | B=1/2/4 by prefix truncation |
| E2r | TEST-1704 | 4 | M3 random, seeds 0/1/2 | B=1/2 by truncation |
| E2r8 | SUB-300 | 8 | M3 random, seeds 0/1/2 | |
| E3_B1 / E3_B2 / E3_B8 | SUB-300 | 1 / 2 / 8 | M5 | B=4 comes from E1 + PILOT-50 |
| E4a_B4 | TEST-1704 | 4 | A3 no adaptive stopping | Reuses the M5 record unless it ended in SLM_STOP. **Run after E1** |
| E4a_B8 | SUB-300 | 8 | A3 | Source is E3_B8. **Run after E3_B8** |
| E4b | SUB-300 | 4 | A1 no history | |
| E4c | SUB-300 | 8 | A2 no budget input | B=1/2/4 by truncation |

## Commands (run in this order; each one resumes when re-run unchanged)
Keep the Mac plugged in with the lid open. `caffeinate -ims` prevents idle, disk and system sleep while on AC power.

```bash
cd /Users/ompatel/Downloads/CSE576/Project
```

```bash
caffeinate -ims .venv-sage/bin/python -P -s experiments/run_experiment.py E2 experiments/runs/E2_baselines_B8_test1704 2>> experiments/runs/E2_baselines_B8_test1704.stderr
```

```bash
caffeinate -ims .venv-sage/bin/python -P -s experiments/run_experiment.py E2r experiments/runs/E2r_random_B4_test1704 2>> experiments/runs/E2r_random_B4_test1704.stderr
```

```bash
caffeinate -ims .venv-sage/bin/python -P -s experiments/run_experiment.py E2r8 experiments/runs/E2r8_random_B8_sub300 2>> experiments/runs/E2r8_random_B8_sub300.stderr
```

```bash
caffeinate -ims .venv-sage/bin/python -P -s experiments/run_experiment.py E1 experiments/runs/E1_slm_B4_new1654 2>> experiments/runs/E1_slm_B4_new1654.stderr
```

```bash
caffeinate -ims .venv-sage/bin/python -P -s experiments/run_experiment.py E4a_B4 experiments/runs/E4a_B4_nostop_test1704 2>> experiments/runs/E4a_B4_nostop_test1704.stderr
```

```bash
caffeinate -ims .venv-sage/bin/python -P -s experiments/run_experiment.py E3_B1 experiments/runs/E3_B1_slm_sub300 2>> experiments/runs/E3_B1_slm_sub300.stderr
```

```bash
caffeinate -ims .venv-sage/bin/python -P -s experiments/run_experiment.py E3_B2 experiments/runs/E3_B2_slm_sub300 2>> experiments/runs/E3_B2_slm_sub300.stderr
```

```bash
caffeinate -ims .venv-sage/bin/python -P -s experiments/run_experiment.py E3_B8 experiments/runs/E3_B8_slm_sub300 2>> experiments/runs/E3_B8_slm_sub300.stderr
```

```bash
caffeinate -ims .venv-sage/bin/python -P -s experiments/run_experiment.py E4c experiments/runs/E4c_nobudget_B8_sub300 2>> experiments/runs/E4c_nobudget_B8_sub300.stderr
```

```bash
caffeinate -ims .venv-sage/bin/python -P -s experiments/run_experiment.py E4b experiments/runs/E4b_nohistory_B4_sub300 2>> experiments/runs/E4b_nohistory_B4_sub300.stderr
```

```bash
caffeinate -ims .venv-sage/bin/python -P -s experiments/run_experiment.py E4a_B8 experiments/runs/E4a_B8_nostop_sub300 2>> experiments/runs/E4a_B8_nostop_sub300.stderr
```

To check progress:

```bash
cat experiments/runs/E1_slm_B4_new1654/progress.json
```

To check the frozen code without running anything:

```bash
.venv-sage/bin/python -P -s experiments/run_experiment.py --verify-frozen
```

Estimates are in EXPERIMENT_PLAN.md §6: E2 ≈ 6 h, E2r ≈ 5.3 h, E2r8 ≈ 3 h (CPU); E1 ≈ 17 h; E3 ≈ 9.4 h; E4 ≈ 11.5 h (Qwen). The run-directory names above are the ones E4a uses as its sources (`RUN_DIRS` in `run_experiment.py`).
