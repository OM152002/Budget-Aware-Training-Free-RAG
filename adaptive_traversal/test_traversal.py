"""Traversal-loop tests (PRD §32 Budget + Traversal) with a scripted controller backend and the real graph.

Usage: python -P -s test_traversal.py
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
for p in (Path(__file__).resolve().parent, PROJECT / "slm_controller", PROJECT / "traversal_adapter"):
    sys.path.insert(0, str(p))
from controller import ControllerConfig, TraversalController  # noqa: E402
from graph_seed_adapter import GraphSeedAdapter  # noqa: E402
from traversal import TraversalConfig, traverse  # noqa: E402


class AdapterOrderPrefilter:
    """Test stand-in for the pre-filter: keeps the adapter order (score 0)."""
    def score(self, q, ids):
        return [(c, 0.0) for c in ids]


class Scripted:
    name = "scripted"

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, system, user, max_new_tokens, temperature, extra_messages=None):
        self.prompts.append((user, extra_messages))
        return {"text": self.replies.pop(0), "prompt_tokens": 10, "output_tokens": 5, "latency_ms": 1.0}


def run(A, qid, replies, prompt_version="v2", **cfg):
    be = Scripted(replies)
    ctl = TraversalController(be, ControllerConfig(prompt_version=prompt_version))
    return traverse(qid, A, AdapterOrderPrefilter(), ctl, TraversalConfig(**cfg)), be


def main():
    A = GraphSeedAdapter()
    qs = A.question_ids
    q = next(x for x in qs if len(A.get_neighbors(A.questions[x]["ranking"][:3])) >= 30)
    ex = lambda a: f'{{"action":"EXPLORE","node_id":"{a}"}}'
    stop = '{"action":"STOP","node_id":null}'
    norm = lambda c: " ".join(A.node(c).content.lower().split())
    checks = {}

    r, _ = run(A, q, [stop])
    checks["STOP terminates as SLM_STOP, budget untouched"] = (r["termination_reason"] == "SLM_STOP"
                                                               and r["budget_used"] == 0 and r["slm_calls"] == 1)
    r, be = run(A, q, [ex("A1")] * 4)
    checks["budget exhaustion after B EXPLOREs"] = (r["termination_reason"] == "BUDGET_EXHAUSTED"
                                                    and r["budget_used"] == 4 and len(r["retrieved_nodes"]) == 7)
    checks["budget decremented exactly once per EXPLORE"] = [s["remaining_budget_before"] - s["remaining_budget_after"]
                                                            for s in r["steps"] if "selected_action" in s] == [1, 1, 1, 1]
    checks["explored node becomes evidence (never re-offered)"] = all(
        s["selected_node_id"] not in [c["chunk_id"] for t in r["steps"][i + 1:] for c in t.get("top_m", [])]
        for i, s in enumerate(r["steps"]) if s.get("selected_node_id"))
    checks["v2 history rendered by snippet, evidence unlabeled"] = ('step 1: EXPLORE "' in be.prompts[1][0]
                                                                   and "[E" not in be.prompts[1][0])
    checks["budget never negative / B=0 makes no calls"] = run(A, q, [], budget=0)[0]["slm_calls"] == 0
    r, be = run(A, q, [ex("E1"), ex("A2"), stop])
    checks["invalid then valid retry continues"] = (r["retries"] == 1 and r["invalid_outputs"] == 1
                                                    and r["explore_steps"] == 1 and r["termination_reason"] == "SLM_STOP"
                                                    and be.prompts[1][1] is not None
                                                    and "A1 to A" in be.prompts[1][1][1]["content"])
    # dedup: find a question whose first-step neighbors contain duplicate texts or evidence-text duplicates
    dq = None
    for x in qs:
        s3 = A.questions[x]["ranking"][:3]
        nb = [n.node.chunk_id for n in A.get_neighbors(s3)]
        ev = {norm(c) for c in s3}
        texts = [norm(c) for c in nb]
        if len(set(texts)) < len(texts) and any(t in ev for t in texts):
            dq = x; break
    r, _ = run(A, dq, [ex("A1")] * 4)
    first = r["steps"][0]
    shown = [c["chunk_id"] for c in first["top_m"]]
    checks["dedup: evidence-text duplicates dropped"] = bool(first["dropped_same_text_as_evidence"]) and not (
        set(first["dropped_same_text_as_evidence"]) & set(shown))
    checks["dedup: one action per text, representative = min id, members logged"] = (
        bool(first["collapsed_groups"]) and len({norm(c) for c in shown}) == len(shown)
        and all(rep == min(g) for rep, g in first["collapsed_groups"].items()))
    checks["dedup: no explored node repeats evidence text"] = all(
        norm(v) not in {norm(c) for c in r["retrieved_nodes"][:i]} for i, v in enumerate(r["retrieved_nodes"]) if i >= 3)
    checks["dedup: graph untouched (all raw neighbors accounted for)"] = (
        first["candidate_count_raw"] == len(first["dropped_same_text_as_evidence"])
        + sum(len(g) for g in first["collapsed_groups"].values())
        + first["candidate_count"] - len(first["collapsed_groups"]))
    r1, be1 = run(A, q, [ex("C1")] * 4, prompt_version="v1", dedup_text=False)
    checks["v1 + dedup off still runs (C aliases, E-labeled evidence)"] = (r1["budget_used"] == 4
                                                                           and "[E1]" in be1.prompts[0][0])
    r, _ = run(A, q, ["nonsense", ex("A999")])
    checks["two invalid -> CONTROLLER_ERROR (not SLM_STOP)"] = (r["termination_reason"] == "CONTROLLER_ERROR"
                                                               and r["budget_used"] == 0 and r["slm_calls"] == 2)
    iso_q = next((x for x in qs if not A.get_neighbors(A.questions[x]["ranking"][:1])), None)
    if iso_q:
        r, _ = run(A, iso_q, [], k=1)
        checks["no candidates -> NO_CANDIDATES without SLM call"] = (r["termination_reason"] == "NO_CANDIDATES"
                                                                    and r["slm_calls"] == 0)
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("ALL_PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
