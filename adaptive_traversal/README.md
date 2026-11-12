# SLM-controlled multi-hop traversal: pilot (k=3, m=20, B=4)

| File | Role |
|---|---|
| `traversal.py` | `traverse(qid, adapter, prefilter, controller, TraversalConfig)`: the PRD §21 loop. It is gold-free and returns a PRD §23 result plus per-step records |
| `test_traversal.py` | Loop tests with a scripted controller on the real graph: STOP, budget exhaustion, exactly-once decrement, visited exclusion, history, B=0, retry, CONTROLLER_ERROR, NO_CANDIDATES |
| `run_pilot.py` | 24-question pilot (3 strata × 8). Gold is used only for sampling and the post-hoc evaluation |

**Loop:**
1. Candidates are the unvisited 1-hop neighbors of the current evidence.
2. Take the top 20 by the SAGE pre-filter (cache-free).
3. Call Qwen.
4. A valid EXPLORE adds the node to evidence, visited and history, and decrements the budget by 1.
5. Stop on SLM_STOP, BUDGET_EXHAUSTED or NO_CANDIDATES.

An invalid output gets one retry (the invalid reply is replayed, followed by a fixed correction message). A second invalid output ends the traversal with CONTROLLER_ERROR.

```bash
cd Project/adaptive_traversal
../.venv-sage/bin/python -P -s test_traversal.py
../.venv-sage/bin/python -P -s run_pilot.py runs/<new_dir> 2> runs/<new_dir>.stderr   # ~20 min
```

**Interface fixes (v2, now the default):**
- Evidence is shown as unlabeled, non-selectable context, and candidates use the action aliases A1..Am. The prompt states that only those aliases or STOP are valid.
- Before pre-filtering, candidates whose normalized text matches an evidence passage are dropped, and exact-duplicate texts are collapsed into one action (representative = smallest chunk_id; all members are logged; the graph is untouched).
- The prompt states that the budget is a maximum, not a target, and gives both STOP conditions.

The accepted earlier pilot is reproducible with `ControllerConfig(prompt_version="v1")` and `TraversalConfig(dedup_text=False)`. To compare two runs: `python -P -s compare_pilots.py runs/pilot_k3_m20_B4 runs/pilot_k3_m20_B4_v2`.

Output of the accepted earlier pilot is in `runs/pilot_k3_m20_B4/`, and the v2 re-run is in `runs/pilot_k3_m20_B4_v2/` (with `comparison_vs_v1.json`). Each contains: `summary.json`, `results.jsonl` (per query), `steps.jsonl` (per step: evidence, candidate count, top-m with scores, attempts with raw output, tokens and latency, retry flag, budget before and after, history, termination) and `traces.md`.
