"""Validate hybrid_search_standalone against SAGE's ORIGINAL agent/ott_qa_api_server.py::hybrid_search.

The original module imports Flask, which is not installed. For this validation only, a minimal stub `flask`
module is placed in sys.modules so that SAGE's unmodified source can be imported and executed. The stub
provides Flask(...).route (identity decorator), request.get_json() and jsonify (identity). Nothing is installed.

Usage: OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python -I validate_seed.py <out_dir>
"""
import json
import math
import random
import sys
import time
import types
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_common as C  # noqa: E402


class _Req:
    payload = None

    def get_json(self):
        return self.payload


class _App:
    def __init__(self, *a, **kw):
        pass

    def route(self, *a, **kw):
        return lambda f: f


def install_flask_stub():
    stub = types.ModuleType("flask")
    stub.Flask, stub.request, stub.jsonify = _App, _Req(), (lambda x: x)
    sys.modules["flask"] = stub
    return stub


def main():
    C.check_env()
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    loader, hnsw, bm25 = C.build_indices()
    graph_ids = set(pd.read_csv(C.GRAPH_IDS_CSV).chunk_id)
    model = C.load_model()

    stub = install_flask_stub()
    import agent.ott_qa_api_server as S  # unmodified SAGE source
    S.hnsw_index, S.chunk_ids_list = hnsw.index, hnsw.chunk_ids
    S.bm25_index, S.chunks_data, S.embedding_model = C.bm25_dict(bm25), loader.chunks, model

    q = C.load_questions()
    rng = random.Random(0)
    cases = [(i, 100) for i in rng.sample(range(len(q)), 50)] + [(i, 20) for i in rng.sample(range(len(q)), 10)] \
        + [(i, 200) for i in rng.sample(range(len(q)), 10)]
    rep = {"corpus_size": len(loader.chunks), "corpus_equals_graph_nodes": set(loader.chunks) == graph_ids,
           "hnsw_ntotal": hnsw.index.ntotal, "hnsw_params": {"M": 32, "efConstruction": hnsw.index.hnsw.efConstruction,
                                                             "efSearch": hnsw.index.hnsw.efSearch},
           "cases": len(cases), "order_mismatch": 0, "length_mismatch": 0, "field_mismatch": 0, "max_score_abs_diff": 0.0}
    for i, k in cases:
        query = q.question[i]
        stub.request.payload = {"query": query, "k": k}
        orig = S.hybrid_search()
        if isinstance(orig, tuple):
            raise SystemExit(f"original hybrid_search errored: {orig}")
        mine = C.hybrid_search_standalone(query, k, model, hnsw.index, hnsw.chunk_ids, C.bm25_dict(bm25), loader.chunks)
        rep["length_mismatch"] += len(orig) != len(mine)
        rep["order_mismatch"] += [r["chunk_id"] for r in orig] != [r["chunk_id"] for r in mine]
        for a, b in zip(orig, mine):
            for f in ("combined_score", "hnsw_score", "bm25_score"):
                rep["max_score_abs_diff"] = max(rep["max_score_abs_diff"], abs(a[f] - b[f]))
            rep["field_mismatch"] += any(a[f] != b[f] for f in ("rank", "content", "source_name"))
        rep.setdefault("returned_len_by_k", {}).setdefault(str(k), set()).add(len(orig))
    rep["returned_len_by_k"] = {k: sorted(v) for k, v in rep["returned_len_by_k"].items()}
    rep["PASS"] = (rep["corpus_size"] == 21742 and rep["corpus_equals_graph_nodes"] and rep["order_mismatch"] == 0
                   and rep["length_mismatch"] == 0 and rep["field_mismatch"] == 0 and rep["max_score_abs_diff"] <= 1e-12)
    rep["elapsed_s"] = round(time.time() - t0, 1)
    (out / "validation_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
