"""Small validation of the CPU port before the full build (plan §5: V1, V3, V4, V6, V7, plus load checks).

Usage: python -I validate_port.py <run_dir> [--pairs 2000]
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sage_cpu_runner as R  # noqa: E402
import reference_features as ref  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--pairs", type=int, default=2000)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    ids = R.prepare_run_dir(run_dir)
    R.setup_logging(run_dir)
    R.apply_patches()
    os.chdir(run_dir)
    out, t0 = {"checks": {}}, time.time()

    p = R.new_pipeline()
    p.load_chunks()
    n_doc = sum(c.chunk_type == "document" for c in p.chunks)
    n_tab = sum(c.chunk_type == "table" for c in p.chunks)
    out["nodes"] = {"documents": n_doc, "tables": n_tab, "indexing_filtered_rows": len(ids)}
    out["checks"]["L0_docs_only_load"] = n_doc == 21742 and n_tab == 0

    # V4: stored embeddings used verbatim
    raw = {}
    for f in os.listdir(R.DOCS_DIR):
        with open(R.DOCS_DIR / f) as fh:
            o = json.load(fh)
        raw[o["chunk_id"]] = o
    same = sum(c.content_embedding == raw[c.chunk_id]["embedding"] for c in p.chunks)
    out["checks"]["V4_stored_embeddings_identical"] = same == len(p.chunks)
    out["V4_identical_count"] = same

    # V3 (CPU): SAGE EmbeddingService topic vectors vs direct SentenceTransformer encode
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    rng = random.Random(0)
    samp = rng.sample(range(len(p.chunks)), 1000)
    direct = st.encode([p.chunks[i].topic for i in samp], normalize_embeddings=True, batch_size=64)
    sage_t = np.array([p.chunks[i].topic_embedding for i in samp], dtype=np.float32)
    cos = (direct * sage_t).sum(1)
    out["V3_topic_cos_min"] = float(cos.min())
    out["checks"]["V3_topic_embeddings"] = bool(cos.min() >= 0.99999)

    # HNSW + candidate pairs via SAGE's own methods; V6 determinism with a second independent build
    p.build_hnsw_index()
    nb1 = p._find_all_neighbors_parallel(200)
    pairs = sorted(p._generate_unique_pairs(nb1))
    out["candidate_pairs"] = len(pairs)
    p2 = R.new_pipeline()
    p2.chunks, p2.chunk_index = p.chunks, p.chunk_index
    p2.build_hnsw_index()
    nb2 = p2._find_all_neighbors_parallel(200)
    diff = sum(nb1[k] != nb2[k] for k in nb1)
    out["V6_neighbor_lists_differing"] = diff
    out["checks"]["V6_hnsw_deterministic"] = diff == 0

    # V7: HNSW vs exact top-200 on 1000 queries
    import faiss
    E = np.array([c.content_embedding for c in p.chunks], dtype=np.float32)
    faiss.normalize_L2(E)
    flat = faiss.IndexFlatIP(E.shape[1]); flat.add(E)
    q = rng.sample(range(len(E)), 1000)
    _, I = flat.search(E[q], 201)
    rec = []
    for r, i in enumerate(q):
        exact = [p.chunks[j].chunk_id for j in I[r] if j != i][:200]
        rec.append(len(set(exact) & set(nb1[p.chunks[i].chunk_id])) / 200)
    out["V7_hnsw_recall_at_200"] = float(np.mean(rec))
    out["checks"]["V7_hnsw_recall"] = out["V7_hnsw_recall_at_200"] >= 0.998

    # V1: SAGE calculator (exact P2 code path) on sampled pairs vs independent reimplementation
    sample = rng.sample(pairs, args.pairs)
    calc = R.gap.GPUAcceleratedSimilarityCalculator(chunks=p.chunks, embedding_service=p.embedding_service,
                                                    batch_size=10000, gpu_id=0)
    calc.precompute_embeddings()
    calc._precompute_all_similarities(sample)
    res = calc.calculate_similarities_batch(sample)
    idx = {c.chunk_id: i for i, c in enumerate(p.chunks)}
    topic = np.array([c.topic_embedding for c in p.chunks], dtype=np.float32)
    dmax = {"content_similarity": 0.0, "topic_similarity": 0.0, "entity_relationship_overlap": 0.0}
    mism = {"entity_count": 0, "event_count": 0, "edge_type": 0, "orientation": 0}
    for r in res:
        a, b = raw[r.source_chunk_id], raw[r.target_chunk_id]
        f = ref.pair_features(a, b, E, topic, idx)
        for k in dmax:
            dmax[k] = max(dmax[k], abs(getattr(r, k) - f[k]))
        mism["entity_count"] += r.entity_count != f["entity_count"]
        mism["event_count"] += r.event_count != f["event_count"]
        mism["edge_type"] += r.edge_type != "doc-doc"
        mism["orientation"] += not (r.source_chunk_id < r.target_chunk_id)
    out["V1_pairs_compared"] = len(res)
    out["V1_max_abs_diff"] = dmax
    out["V1_mismatches"] = mism
    out["V1_sample_entity_count_gt0"] = sum(r.entity_count > 0 for r in res)
    out["checks"]["V1_float_features"] = (dmax["content_similarity"] <= 1e-6 and dmax["topic_similarity"] <= 1e-6
                                          and dmax["entity_relationship_overlap"] <= 1e-12)
    out["checks"]["V1_integer_features_exact"] = all(v == 0 for v in mism.values()) and len(res) == args.pairs

    out["patch_calls"] = dict(R.PATCH_LOG)
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["ALL_PASS"] = all(out["checks"].values())
    (run_dir / "validation_report.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
