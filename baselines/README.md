# Equal-budget non-SLM traversal baselines

`fixed_traversal.py` shares every component of the frozen v2 adaptive method: the same seeds (k=3), unvisited 1-hop neighbors of the current evidence, the frozen duplicate handling (imported from `adaptive_traversal/traversal.py`) and the SAGE pre-filter (top-20). Only the decision rule differs:
- `sage_score`: take the highest-scoring eligible candidate.
- `random`: pick uniformly from the eligible top 20, seeded per (seed, question_id).

Neither ever stops early.

```bash
cd Project/baselines
../.venv-sage/bin/python -P -s run_baselines_heldout.py runs/<new_dir> 2> runs/<new_dir>.stderr   # ~12 min
```

Results for the held-out 50 are in `runs/heldout50_baselines/`: `summary.json`, `per_question.jsonl` and `runs.jsonl` (every baseline path). The frozen SLM results are read from `adaptive_traversal/runs/heldout50_k3_m20_B4_v2/` and not re-run.

**Checks (50/50 each):**
- matched-expansion runs are exact prefixes of the B=4 runs;
- the SAGE-score step-1 pick equals the recorded `baseline_inputs.jsonl` rank-1 candidate;
- seeds are identical to the SLM run.
