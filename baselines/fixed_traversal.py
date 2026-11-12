"""Non-SLM traversal baselines that share every component of the frozen v2 adaptive method except the decision rule.

Per step (identical to adaptive_traversal/traversal.py up to the decision):
  1. candidates = unvisited 1-hop neighbors of the current evidence E          (GraphSeedAdapter.get_neighbors)
  2. dedup_text: drop evidence-text duplicates, collapse identical texts       (traversal.dedup_candidates)
  3. rank by the SAGE pre-filter, keep top-m                                   (SagePrefilter.score, cache-free)
  4. decision rule:
       sage_score : the highest-scoring eligible candidate (top-m rank 1)
       random     : uniform over the eligible top-m, seeded per question (seed, question_id) -> order-independent
  5. E += v, budget -= 1; stop at budget 0 or NO_CANDIDATES (no other stopping).
Gold-free: no gold label is read here.
"""
import hashlib
import random
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "adaptive_traversal"))
sys.path.insert(0, str(PROJECT / "slm_controller"))
from traversal import dedup_candidates  # noqa: E402  (frozen v2 duplicate handling, imported unchanged)

POLICIES = ("sage_score", "random")


def question_rng(seed: int, question_id: str) -> random.Random:
    h = hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def traverse_fixed(question_id, adapter, prefilter, policy: str, budget: int, k: int = 3, m: int = 20,
                   dedup_text: bool = True, seed: int = 0) -> dict:
    assert policy in POLICIES and budget >= 0
    t0 = time.perf_counter()
    q = adapter.questions[question_id]["question"]
    seeds, _ = adapter.get_seeds(question_id, k)
    S = [s.node.chunk_id for s in seeds]
    E, b, steps, termination = list(S), budget, [], None
    rng = question_rng(seed, question_id) if policy == "random" else None
    while b > 0:
        raw = [n.node.chunk_id for n in adapter.get_neighbors(E)]
        if dedup_text:
            cands, members, _ = dedup_candidates(raw, E, lambda c: adapter.node(c).content)
        else:
            cands, members = raw, {c: [c] for c in raw}
        if not cands:
            termination = "NO_CANDIDATES"
            break
        top = prefilter.score(q, cands)[:m]
        idx = 0 if policy == "sage_score" else rng.randrange(len(top))
        v = top[idx][0]
        E.append(v); b -= 1
        steps.append({"step": len(steps) + 1, "candidate_count_raw": len(raw), "candidate_count": len(cands),
                      "top_m": [c for c, _ in top], "selected": v, "selected_rank": idx + 1,
                      "selected_text_duplicate_members": members[v]})
    return {"query_id": question_id, "policy": policy, "budget_allocated": budget, "seed_nodes": S,
            "retrieved_nodes": E, "explore_steps": len(E) - len(S),
            "termination_reason": termination or "BUDGET_EXHAUSTED", "steps": steps,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
