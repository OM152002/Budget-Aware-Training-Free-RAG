"""Frozen question-set definitions (EXPERIMENT_PLAN.md §0). Deterministic; only uses "has document gold".

  DEV-24     questions of the two development pilots (excluded from all primary evaluation)
  TEST-1704  all document-gold questions minus DEV-24 (sorted by question_id)
  PILOT-50   held-out pilot questions already evaluated with the frozen SLM (in their original sampling order)
  NEW-1654   TEST-1704 minus PILOT-50 (sorted)
  SUB-300    random.Random(20261101).sample(TEST-1704, 300), in sampled order

Usage: python -P -s question_sets.py   -> writes question_sets.json (refuses to overwrite a different definition)
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(PROJECT / "traversal_adapter"))
from runner import ids_sha256  # noqa: E402

OUT = HERE / "question_sets.json"
SUB_SEED = 20261101


def build():
    from graph_seed_adapter import GraphSeedAdapter
    A = GraphSeedAdapter()
    doc = sorted(q for q in A.question_ids if A.questions[q]["eval_group"] == "doc")
    dev = sorted(json.loads(l)["query_id"] for l in open(PROJECT / "adaptive_traversal/runs/pilot_k3_m20_B4_v2/results.jsonl"))
    pilot = json.loads((PROJECT / "adaptive_traversal/runs/heldout50_k3_m20_B4_v2/questions.json").read_text())["question_ids"]
    test = [q for q in doc if q not in set(dev)]
    new = [q for q in test if q not in set(pilot)]
    sub = random.Random(SUB_SEED).sample(test, 300)
    sets = {"DEV-24": dev, "TEST-1704": test, "PILOT-50": pilot, "NEW-1654": new, "SUB-300": sub}
    assert (len(doc), len(dev), len(test), len(pilot), len(new), len(sub)) == (1728, 24, 1704, 50, 1654, 300)
    assert not set(dev) & set(test) and set(pilot) <= set(test) and set(sub) <= set(test)
    return {"definition": __doc__.strip().splitlines()[2:8], "sub_seed": SUB_SEED,
            "sets": {k: {"n": len(v), "ids_sha256": ids_sha256(v), "ids": v} for k, v in sets.items()},
            "overlaps": {"SUB-300 ∩ PILOT-50": len(set(sub) & set(pilot))}}


def load():
    return json.loads(OUT.read_text())


if __name__ == "__main__":
    d = build()
    if OUT.exists() and load()["sets"] != d["sets"]:
        raise SystemExit("question_sets.json exists with a DIFFERENT definition; refusing to overwrite")
    OUT.write_text(json.dumps(d, indent=1))
    print({k: (v["n"], v["ids_sha256"][:12]) for k, v in d["sets"].items()}, d["overlaps"])
