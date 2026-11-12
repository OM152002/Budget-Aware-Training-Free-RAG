"""Shared setup for SAGE seed regeneration over the 21,742-document graph corpus.

Indices are built with SAGE's own, unmodified classes from experiments/ott_qa/ott_qa_retrieval_pipeline.py
(ChunkLoader, HNSWIndex, BM25Index), exactly as its main() steps 1, 3, 4 do (L537-584), restricted to
document chunks (decision G1: no table chunks, so no generate_table_embeddings call).

hybrid_search_standalone() is a line-for-line transcription of agent/ott_qa_api_server.py::hybrid_search
(L272-344) without Flask: request JSON -> function arguments, jsonify(results) -> return results.
Must run with OMP_NUM_THREADS=1 and PYTHONHASHSEED=0 (see README).
"""
import ast
import os
import re
import sys
from pathlib import Path

import numpy as np
# Import order matters on this Mac (faiss-cpu and torch bundle separate libomp copies): faiss, then torch, then
# sentence_transformers, all before any faiss work; other orders abort with "OMP: Error #15" (tested).
import faiss  # noqa: F401
import torch  # noqa: F401
import sentence_transformers  # noqa: F401

PROJECT = Path(__file__).resolve().parent.parent
SAGE_ROOT = PROJECT / "sage"
DOCS_DIR = PROJECT / "sage_hf_artifacts" / "extracted" / "docs_chunks_1"
RAW = PROJECT / "sage_hf_artifacts" / "raw"
QUESTIONS_CSV = RAW / "bm25_only_results.csv"           # question_id, question, gold_docs (S3: V1-compatible gold)
AUTHOR_RANKING_CSV = RAW / "dense_sparse_average_results.csv"  # diagnostic only (S6)
GRAPH_IDS_CSV = (PROJECT / "graph_reconstruction" / "runs" / "docgraph_k200_q99_20261023" / "output"
                 / "indexing_filtered.csv")
MODEL_NAME = "all-MiniLM-L6-v2"   # as in ott_qa_api_server.get_embedding_model (L164)

sys.path.insert(0, str(SAGE_ROOT))


def check_env():
    """hybrid_search builds candidates in a Python set and sorts stably, so tied scores are ordered by string-hash
    iteration order. PYTHONHASHSEED=0 fixes that order. Run with `python -P -s` (not -I: -I implies -E, which
    ignores PYTHONHASHSEED; verified)."""
    for k, v in (("OMP_NUM_THREADS", "1"), ("PYTHONHASHSEED", "0")):
        if os.environ.get(k) != v:
            raise SystemExit(f"set {k}={v}")
    if sys.flags.hash_randomization:
        raise SystemExit("hash randomization is active (PYTHONHASHSEED ignored); run with `python -P -s`, not -I")


def build_indices():
    """SAGE ChunkLoader -> HNSWIndex.build (M=32, efC=200, efS=100) -> BM25Index.build (BM25Okapi defaults)."""
    import faiss
    from experiments.ott_qa.ott_qa_retrieval_pipeline import ChunkLoader, HNSWIndex, BM25Index
    faiss.omp_set_num_threads(1)
    loader = ChunkLoader()
    loader.load_document_chunks(DOCS_DIR)
    chunk_ids = loader.get_all_chunk_ids()
    hnsw = HNSWIndex()
    hnsw.build(chunk_ids, loader.get_embedding_matrix(chunk_ids))
    bm25 = BM25Index()
    bm25.build(chunk_ids, loader.bm25_texts)
    return loader, hnsw, bm25


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device="cpu")  # CPU-only decision; original code used default device


def embed_query(model, query):
    """ott_qa_api_server.embed_query (L168-171)."""
    return model.encode(query, convert_to_numpy=True).astype(np.float32)


def hybrid_search_standalone(query, k, model, hnsw_index, chunk_ids_list, bm25_index, chunks_data):
    """Transcription of ott_qa_api_server.hybrid_search (L272-344). hnsw_index = raw faiss index,
    bm25_index = {'bm25', 'chunk_ids', 'tokenized_corpus'} (the BM25Index.save pickle layout)."""
    import faiss

    k = min(k, 200)

    if not query:
        return []

    # HNSW search
    query_emb = embed_query(model, query).reshape(1, -1)
    faiss.normalize_L2(query_emb)
    hnsw_distances, hnsw_indices = hnsw_index.search(query_emb, k * 2)

    hnsw_scores = {}
    for dist, idx in zip(hnsw_distances[0], hnsw_indices[0]):
        if idx >= 0 and idx < len(chunk_ids_list):
            chunk_id = chunk_ids_list[idx]
            hnsw_scores[chunk_id] = 1.0 / (1.0 + float(dist))

    # BM25 search
    query_tokens = re.findall(r'\w+', query.lower())
    bm25_obj = bm25_index['bm25']
    bm25_chunk_ids = bm25_index['chunk_ids']
    scores = bm25_obj.get_scores(query_tokens)

    # Normalize BM25 scores
    max_bm25 = max(scores) if len(scores) > 0 else 1.0
    bm25_scores = {}
    for idx, score in enumerate(scores):
        if score > 0:
            chunk_id = bm25_chunk_ids[idx]
            bm25_scores[chunk_id] = score / max_bm25 if max_bm25 > 0 else 0

    # Combine scores
    all_chunks = set(hnsw_scores.keys()) | set(bm25_scores.keys())
    combined = []

    for chunk_id in all_chunks:
        h_score = hnsw_scores.get(chunk_id, 0)
        b_score = bm25_scores.get(chunk_id, 0)

        # Boost if in both
        boost = 1.5 if chunk_id in hnsw_scores and chunk_id in bm25_scores else 1.0
        combined_score = boost * (0.5 * h_score + 0.5 * b_score)

        combined.append((chunk_id, combined_score, h_score, b_score))

    # Sort by combined score
    combined.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, (chunk_id, c_score, h_score, b_score) in enumerate(combined[:k]):
        chunk = chunks_data.get(chunk_id, {})
        results.append({
            'chunk_id': chunk_id,
            'combined_score': c_score,
            'hnsw_score': h_score,
            'bm25_score': b_score,
            'rank': i + 1,
            'content': chunk.get('content', '')[:500],
            'source_name': chunk.get('source_name', '') or chunk.get('source_info', {}).get('source_name', '')
        })

    return results


def bm25_dict(bm25):
    """In-memory equivalent of BM25Index.save()'s pickle (ott_qa_retrieval_pipeline.py L294-302)."""
    return {'bm25': bm25.bm25, 'chunk_ids': bm25.chunk_ids, 'tokenized_corpus': bm25.tokenized_corpus}


def load_questions():
    import pandas as pd
    df = pd.read_csv(QUESTIONS_CSV, usecols=["question_id", "question", "gold_docs"])
    df["gold_list"] = df.gold_docs.map(ast.literal_eval)
    return df
