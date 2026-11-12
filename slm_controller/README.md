# Standalone SLM controller (Milestone 3): one decision per call, no traversal loop

| File | Role |
|---|---|
| `controller.py` | Model-agnostic `TraversalController.choose_action(query, evidence, candidates, history, remaining_budget)`: prompt builder, strict parser/validator, deterministic fallback |
| `mlx_backend.py` | `MLXBackend`: `mlx-community/Qwen3-4B-Instruct-2507-4bit` @ `50d4277` (4-bit, group size 64), greedy decoding |
| `test_parser.py` | Parser and prompt unit tests (PRD §32), no model needed |
| `run_standalone.py` | 24-question stratified standalone test (k=3, m=20, budget 4, empty history) |

**Contract:**
- Output is exactly `{"action":"EXPLORE","node_id":"C<i>"}` or `{"action":"STOP","node_id":null}`.
- The prompt shows passages under aliases (E1..Ek for evidence, C1..Cm for candidates). A valid alias is mapped back to its chunk ID.
- Invalid output (EMPTY_OUTPUT, MALFORMED_JSON, MULTIPLE_ACTIONS, INVALID_ACTION, MISSING_NODE_ID, CONFLICTING_STOP, VISITED_NODE, NODE_NOT_IN_CANDIDATES) falls back to effective STOP with termination `CONTROLLER_ERROR`, which is distinct from `SLM_STOP`. There is no retry.
- Passage text is whitespace-collapsed and clipped to 300 characters. No gold labels or chunk IDs appear in the prompt.

```bash
cd Project/slm_controller
../.venv-sage/bin/python -P -s test_parser.py
../.venv-sage/bin/python -P -s run_standalone.py runs/<new_dir> 2> runs/<new_dir>.stderr
```

Results are in `runs/standalone_k3_m20/` (`summary.json`, and `decisions.jsonl` with the raw output per question).
