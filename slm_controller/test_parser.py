"""Parser/validation unit tests (PRD §32 controller-output parser), no model needed.

Usage: python -P -s test_parser.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from controller import (ControllerConfig, HistoryStep, Passage, TraversalController, build_prompt,  # noqa: E402
                        parse_action)

AM = {"evidence": {"E1": "seed_a", "E2": "seed_b"}, "candidates": {"C1": "cand_x", "C2": "cand_y"}}
CASES = [
    # (name, raw, expect_valid, expect_action/error, expect_node)
    ("valid EXPLORE", '{"action":"EXPLORE","node_id":"C2"}', True, "EXPLORE", "cand_y"),
    ("valid STOP", '{"action":"STOP","node_id":null}', True, "STOP", None),
    ("STOP without node_id key", '{"action":"STOP"}', True, "STOP", None),
    ("EXPLORE with reason", '{"action":"EXPLORE","node_id":"C1","reason":"needed"}', True, "EXPLORE", "cand_x"),
    ("code fence", '```json\n{"action":"EXPLORE","node_id":"C1"}\n```', True, "EXPLORE", "cand_x"),
    ("empty", "   ", False, "EMPTY_OUTPUT", None),
    ("malformed JSON", '{"action":"EXPLORE","node_id":"C1"', False, "MALFORMED_JSON", None),
    ("prose only", "I would explore C1.", False, "MALFORMED_JSON", None),
    ("invalid action", '{"action":"EXPAND","node_id":"C1"}', False, "INVALID_ACTION", None),
    ("lowercase action", '{"action":"explore","node_id":"C1"}', False, "INVALID_ACTION", None),
    ("nonexistent node", '{"action":"EXPLORE","node_id":"C9"}', False, "NODE_NOT_IN_CANDIDATES", None),
    ("raw chunk id not an alias", '{"action":"EXPLORE","node_id":"cand_x"}', False, "NODE_NOT_IN_CANDIDATES", None),
    ("visited (evidence) node", '{"action":"EXPLORE","node_id":"E1"}', False, "VISITED_NODE", None),
    ("EXPLORE missing node", '{"action":"EXPLORE","node_id":null}', False, "MISSING_NODE_ID", None),
    ("non-string node", '{"action":"EXPLORE","node_id":1}', False, "NODE_NOT_IN_CANDIDATES", None),
    ("STOP with node", '{"action":"STOP","node_id":"C1"}', False, "CONFLICTING_STOP", None),
    ("multiple actions", '{"action":"EXPLORE","node_id":"C1"} {"action":"STOP","node_id":null}', False,
     "MULTIPLE_ACTIONS", None),
]


class FakeBackend:
    name = "fake"

    def __init__(self, text):
        self.text, self.seen = text, None

    def generate(self, system, user, max_new_tokens, temperature):
        self.seen = (system, user)
        return {"text": self.text, "prompt_tokens": 1, "output_tokens": 1, "latency_ms": 0.0}


def main():
    ok = True
    for name, raw, valid, what, node in CASES:
        d = parse_action(raw, AM)
        good = d.valid == valid and ((d.action == what) if valid else (d.error == what)) and d.node_id == node
        if not valid:
            good &= d.effective_action == "STOP" and d.termination == "CONTROLLER_ERROR"
        elif what == "STOP":
            good &= d.termination == "SLM_STOP"
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name}: valid={d.valid} action={d.action} error={d.error} node={d.node_id}")

    # prompt contains no gold / chunk ids, history + budget rendering, ablation switches
    v1 = ControllerConfig(prompt_version="v1")
    ev, ca = [Passage("seed_a", "Alpha text")], [Passage("cand_x", "Beta text"), Passage("cand_y", "Gamma")]
    s, u, am = build_prompt("Who?", ev, ca, [HistoryStep(1, "EXPLORE", "cand_x")], 3, v1)
    checks = {
        "v1 aliases only (no chunk ids in prompt)": "cand_x" not in u and "seed_a" not in u,
        "v1 history rendered with alias": "step 1: EXPLORE C1" in u,
        "v1 budget rendered": "Remaining budget: 3" in u,
    }
    # v2 (default): unlabeled evidence, A-aliases, history by snippet, explicit valid-action list, budget-as-maximum
    ev2 = [Passage("seed_a", "Alpha text"), Passage("cand_z", "Delta explored passage")]
    s2, u2, am2 = build_prompt("Who?", ev2, ca, [HistoryStep(1, "EXPLORE", "cand_z")], 3, ControllerConfig())
    checks.update({
        "v2 no chunk ids in prompt": not any(x in u2 for x in ("cand_x", "cand_y", "cand_z", "seed_a")),
        "v2 evidence has no selectable labels": "- Alpha text" in u2 and "[E" not in u2 and "E1" not in u2,
        "v2 candidates labeled A1..Am": "[A1] Beta text" in u2 and "[A2] Gamma" in u2 and am2["candidates"] == {"A1": "cand_x", "A2": "cand_y"},
        "v2 states valid actions": "valid actions: A1 to A2 or STOP" in u2 and "Only the candidate labels" in s2,
        "v2 budget is a maximum": "maximum, not a target" in s2 and "maximum further explorations): 3" in u2,
        "v2 history by text snippet": 'step 1: EXPLORE "Delta explored passage"' in u2,
        "v2 rejects C-style id": parse_action('{"action":"EXPLORE","node_id":"C1"}', am2).error == "NODE_NOT_IN_CANDIDATES",
        "v2 rejects E-style id": parse_action('{"action":"EXPLORE","node_id":"E1"}', am2).error == "NODE_NOT_IN_CANDIDATES",
        "no-history ablation": "History" not in build_prompt("Who?", ev, ca, [], 3, ControllerConfig(include_history=False))[1],
        "no-budget ablation": "budget" not in build_prompt("Who?", ev, ca, [], 3, ControllerConfig(include_budget=False))[1],
    })
    fb = FakeBackend('{"action":"EXPLORE","node_id":"A2"}')
    d = TraversalController(fb).choose_action("Who?", ev, ca, [], 3)
    checks["controller maps alias -> chunk id"] = d.valid and d.node_id == "cand_y"
    for bad_input in ([], [Passage("seed_a", "dup")]):
        try:
            TraversalController(fb).choose_action("Who?", ev, bad_input, [], 3); checks[f"rejects candidates={len(bad_input)}"] = False
        except ValueError:
            checks[f"rejects candidates={len(bad_input)} (empty or overlapping evidence)"] = True
    for k, v in checks.items():
        ok &= v
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("ALL_PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
