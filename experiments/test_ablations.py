"""Tests for the A3 NoStopController wrapper (scripted backend, real graph, frozen traverse()).

Usage: python -P -s test_ablations.py
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
for p in (Path(__file__).resolve().parent, PROJECT / "adaptive_traversal", PROJECT / "slm_controller",
          PROJECT / "traversal_adapter"):
    sys.path.insert(0, str(p))
from ablations import NoStopController  # noqa: E402
from controller import TraversalController  # noqa: E402
from graph_seed_adapter import GraphSeedAdapter  # noqa: E402
from traversal import TraversalConfig, traverse  # noqa: E402


class AdapterOrderPrefilter:
    def score(self, q, ids):
        return [(c, 0.0) for c in ids]


class Scripted:
    name = "scripted"

    def __init__(self, replies):
        self.replies, self.msgs = list(replies), []

    def generate(self, system, user, max_new_tokens, temperature, extra_messages=None):
        self.msgs.append((user, extra_messages))
        return {"text": self.replies.pop(0), "prompt_tokens": 10, "output_tokens": 5, "latency_ms": 1.0}


def run(A, q, replies, wrap):
    be = Scripted(replies)
    c = TraversalController(be)
    return traverse(q, A, AdapterOrderPrefilter(), NoStopController(c) if wrap else c, TraversalConfig()), be


def main():
    A = GraphSeedAdapter()
    q = next(x for x in A.question_ids if len(A.get_neighbors(A.questions[x]["ranking"][:3])) >= 30)
    ex = lambda a: f'{{"action":"EXPLORE","node_id":"{a}"}}'
    stop = '{"action":"STOP","node_id":null}'
    checks = {}
    # identical to M5 when no STOP occurs
    r0, b0 = run(A, q, [ex("A1"), ex("A2"), ex("A1"), ex("A3")], wrap=False)
    r1, b1 = run(A, q, [ex("A1"), ex("A2"), ex("A1"), ex("A3")], wrap=True)
    checks["no STOP -> identical trajectory and prompts to M5"] = (r0["retrieved_nodes"] == r1["retrieved_nodes"]
                                                                  and b0.msgs == b1.msgs)
    # STOP is suppressed by one re-prompt, budget then exhausted
    r, be = run(A, q, [ex("A1"), stop, ex("A2"), ex("A1"), ex("A1")], wrap=True)
    s2 = r["steps"][1]["calls"][0]
    checks["STOP -> one no-stop re-prompt -> EXPLORE"] = (r["termination_reason"] == "BUDGET_EXHAUSTED"
                                                         and r["budget_used"] == 4 and r["slm_calls"] == 4
                                                         and "STOP is not available" in be.msgs[2][1][1]["content"]
                                                         and be.msgs[2][1][0]["content"] == stop)
    checks["re-prompt tokens/latency summed into the step"] = s2["prompt_tokens"] == 20 and s2["output_tokens"] == 10
    be = Scripted([ex("A1"), stop, ex("A2"), ex("A1"), ex("A1")])
    ns = NoStopController(TraversalController(be))
    r = traverse(q, A, AdapterOrderPrefilter(), ns, TraversalConfig())
    checks["suppressed STOP logged + extra call counted"] = (ns.extra_calls == 1 and len(ns.events) == 1
                                                            and ns.events[0]["suppressed_stop_raw"] == stop
                                                            and ns.events[0]["history_len"] == 1
                                                            and r["slm_calls"] + ns.extra_calls == len(be.msgs) == 5)
    ns.reset()
    checks["reset clears per-question log"] = ns.events == [] and ns.extra_calls == 0
    # STOP twice -> invalid -> frozen single retry -> base returns STOP -> re-prompt -> STOP -> CONTROLLER_ERROR
    r, be = run(A, q, [stop, stop, stop, stop], wrap=True)
    checks["persistent STOP -> CONTROLLER_ERROR (never SLM_STOP)"] = (r["termination_reason"] == "CONTROLLER_ERROR"
                                                                     and r["retries"] == 1)
    # invalid STOP re-prompt recovered by the frozen retry
    r, be = run(A, q, [stop, "garbage", ex("A2"), ex("A1"), ex("A1"), ex("A1")], wrap=True)
    checks["invalid re-prompt -> frozen retry recovers"] = (r["termination_reason"] == "BUDGET_EXHAUSTED"
                                                           and r["retries"] == 1 and r["budget_used"] == 4)
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("ALL_PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
