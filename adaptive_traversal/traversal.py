"""SLM-controlled multi-hop traversal (PRD §11-23). Gold-free: this module never sees gold labels.

Per step:
  1. candidates = unvisited 1-hop neighbors of the current evidence E   (GraphSeedAdapter.get_neighbors; V == E)
  2. top-m by the accepted SAGE pre-filter                              (SagePrefilter.score, cache-free)
  3. controller(query, E, top-m, history, remaining budget)            (TraversalController)
  4. valid EXPLORE(v): E += v, V += v, history += v, budget -= 1 (exactly once)
     invalid output: ONE deterministic retry with a fixed correction; second invalid -> CONTROLLER_ERROR
  5. repeat until SLM_STOP / BUDGET_EXHAUSTED / NO_CANDIDATES (or CONTROLLER_ERROR)

dedup_text (default on; off reproduces the accepted v1 pilot), applied between steps 1 and 2:
  - drop candidates whose normalized text (lowercased, whitespace-collapsed) equals that of any evidence passage;
  - collapse candidates with identical normalized text into one action whose representative is the smallest chunk_id
    of the group. The graph is untouched; all member node IDs are logged, and only the representative enters E.
"""
import time
from dataclasses import dataclass
from typing import List

from controller import HistoryStep, Passage

TERMINATIONS = ("SLM_STOP", "BUDGET_EXHAUSTED", "NO_CANDIDATES", "CONTROLLER_ERROR")


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def dedup_candidates(cand_ids, evidence_ids, text_of):
    """-> (representatives in adapter order, {rep: [all members]}, [ids dropped as evidence-text duplicates])."""
    ev_norm = {normalize_text(text_of(c)) for c in evidence_ids}
    groups, dropped = {}, []
    for c in cand_ids:
        t = normalize_text(text_of(c))
        if t in ev_norm:
            dropped.append(c)
        else:
            groups.setdefault(t, []).append(c)
    reps, members = [], {}
    for g in groups.values():                       # dict preserves adapter order of first appearance
        rep = min(g)
        reps.append(rep); members[rep] = sorted(g)
    return reps, members, dropped


@dataclass
class TraversalConfig:
    k: int = 3
    m: int = 20
    budget: int = 4
    max_retries: int = 1
    dedup_text: bool = True


def _call_record(d, attempt):
    return {"attempt": attempt, "raw_output": d.raw_output, "valid": d.valid, "error": d.error, "action": d.action,
            "alias": d.alias, "node_id": d.node_id, "prompt_tokens": d.prompt_tokens, "output_tokens": d.output_tokens,
            "latency_ms": round(d.latency_ms, 1)}


def traverse(question_id: str, adapter, prefilter, controller, cfg: TraversalConfig, step_log=None) -> dict:
    t_start = time.perf_counter()
    q = adapter.questions[question_id]["question"]
    seeds, unresolved = adapter.get_seeds(question_id, cfg.k)          # strict: raises on unresolved seeds
    S = [s.node.chunk_id for s in seeds]
    E: List[str] = list(S)
    V = set(S)
    H: List[HistoryStep] = []
    b = cfg.budget
    steps, calls, termination = [], [], None

    def passage(cid):
        return Passage(cid, adapter.node(cid).content)

    step = 0
    while b > 0:
        step += 1
        t0 = time.perf_counter()
        neighbors = adapter.get_neighbors(E)                      # excludes E (== V); deduplicated
        raw_ids = [n.node.chunk_id for n in neighbors]
        assert not (set(raw_ids) & V)
        if cfg.dedup_text:
            cand_ids, members, dropped = dedup_candidates(raw_ids, E, lambda c: adapter.node(c).content)
        else:
            cand_ids, members, dropped = raw_ids, {c: [c] for c in raw_ids}, []
        dedup_log = {"candidate_count_raw": len(raw_ids), "dropped_same_text_as_evidence": dropped,
                     "collapsed_groups": {r: g for r, g in members.items() if len(g) > 1}}
        if not cand_ids:
            termination = "NO_CANDIDATES"
            steps.append({"step": step, "candidate_count": 0, **dedup_log, "termination": termination})
            break
        ranked = prefilter.score(q, cand_ids)
        top = ranked[:cfg.m]
        prefilter_ms = (time.perf_counter() - t0) * 1000
        ev_p, ca_p = [passage(c) for c in E], [passage(c) for c, _ in top]
        before = b
        d = controller.choose_action(q, ev_p, ca_p, H, b)
        attempts = [_call_record(d, 1)]
        retried = False
        if not d.valid and cfg.max_retries > 0:
            retried = True
            d = controller.choose_action(q, ev_p, ca_p, H, b, previous_invalid=d.raw_output)
            attempts.append(_call_record(d, 2))
        calls += attempts
        alias_prefix = "A" if controller.cfg.prompt_version == "v2" else "C"
        rec = {"step": step, "evidence_ids": list(E), "candidate_count": len(cand_ids), **dedup_log,
               "top_m": [{"rank": i + 1, "alias": f"{alias_prefix}{i + 1}", "chunk_id": c, "prefilter_score": round(s, 6),
                          "text_duplicate_members": members[c]}
                         for i, (c, s) in enumerate(top)],
               "history_before": [(h.step, h.action, h.chunk_id) for h in H],
               "remaining_budget_before": before, "prefilter_ms": round(prefilter_ms, 1),
               "retried": retried, "calls": attempts}
        if not d.valid:
            termination = "CONTROLLER_ERROR"
            rec.update(selected_action="INVALID", selected_node_id=None, remaining_budget_after=b, termination=termination)
            steps.append(rec)
            break
        if d.action == "STOP":
            H.append(HistoryStep(step, "STOP", None))
            termination = "SLM_STOP"
            rec.update(selected_action="STOP", selected_node_id=None, remaining_budget_after=b, termination=termination)
            steps.append(rec)
            break
        v = d.node_id
        assert v in {c for c, _ in top} and v not in V, "validator guarantees a fresh supplied candidate"
        E.append(v); V.add(v); H.append(HistoryStep(step, "EXPLORE", v)); b -= 1
        rec.update(selected_action="EXPLORE", selected_node_id=v, selected_alias=d.alias,
                   selected_text_duplicate_members=members[v],
                   selected_prefilter_rank=[c for c, _ in top].index(v) + 1, remaining_budget_after=b,
                   history_after=[(h.step, h.action, h.chunk_id) for h in H])
        steps.append(rec)
        if step_log:
            step_log(question_id, rec)
    if termination is None:
        termination = "BUDGET_EXHAUSTED"
        steps.append({"step": step + 1, "termination": termination, "remaining_budget": b})
    if step_log and steps and steps[-1].get("termination"):
        step_log(question_id, steps[-1])
    assert b >= 0 and cfg.budget - b == len(E) - len(S)

    return {
        "query_id": question_id, "query": q, "seed_nodes": S, "retrieved_nodes": E,
        "explored_text_duplicate_members": {s["selected_node_id"]: s["selected_text_duplicate_members"]
                                            for s in steps if s.get("selected_action") == "EXPLORE"},
        "traversal_history": [(h.step, h.action, h.chunk_id) for h in H], "termination_reason": termination,
        "budget_allocated": cfg.budget, "budget_used": cfg.budget - b, "explore_steps": len(E) - len(S),
        "slm_calls": len(calls), "retries": sum(c["attempt"] == 2 for c in calls),
        "invalid_outputs": sum(not c["valid"] for c in calls),
        "input_tokens": sum(c["prompt_tokens"] for c in calls), "output_tokens": sum(c["output_tokens"] for c in calls),
        "slm_latency_ms": round(sum(c["latency_ms"] for c in calls), 1),
        "prefilter_latency_ms": round(sum(s.get("prefilter_ms", 0) for s in steps), 1),
        "total_latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
        "steps": steps,
    }
