"""Regenerate SAGE Fusion-A (hybrid_search) top-100 seed rankings for all 2,212 questions, document corpus only.

The number of seeds actually used for traversal is NOT fixed here: we save the full top-100, and consumers
choose k (Variant B reads the first k of columns 1st..100th via get_csv_retrieved_docs).

Usage: OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python -I run_seeds.py <out_dir>
"""
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_common as C  # noqa: E402

TOP_K = 100


def ordinal(i):
    return f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th"  # as L489


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    C.check_env()
    out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=False)
    proc, peak, stop = psutil.Process(), [0], [False]
    def watch():
        while not stop[0]:
            peak[0] = max(peak[0], proc.memory_info().rss); time.sleep(0.5)
    threading.Thread(target=watch, daemon=True).start()

    t0 = time.time()
    loader, hnsw, bm25 = C.build_indices()
    t_index = time.time() - t0
    model = C.load_model()
    bm = C.bm25_dict(bm25)
    corpus = set(loader.chunks)
    q = C.load_questions()

    rows, t1 = [], time.time()
    with open(out / "seeds_top100_scores.jsonl", "w") as fs:
        for qid, question, gold_str, gold in zip(q.question_id, q.question, q.gold_docs, q.gold_list):
            res = C.hybrid_search_standalone(question, TOP_K, model, hnsw.index, hnsw.chunk_ids, bm, loader.chunks)
            doc_gold = [g for g in gold if g in corpus]
            row = {"question_id": qid, "question": question, "gold_docs": gold_str,
                   "doc_gold": json.dumps(doc_gold), "n_gold": len(gold), "n_doc_gold": len(doc_gold),
                   "eval_group": "doc" if doc_gold else "table_only_or_unresolved",
                   "n_returned": len(res)}
            for r in res:
                row[ordinal(r["rank"])] = r["chunk_id"]
            rows.append(row)
            fs.write(json.dumps({"question_id": qid, "ranking": [
                {"rank": r["rank"], "chunk_id": r["chunk_id"], "combined_score": r["combined_score"],
                 "hnsw_score": r["hnsw_score"], "bm25_score": r["bm25_score"]} for r in res]}) + "\n")
    t_query = time.time() - t1
    df = pd.DataFrame(rows)
    df = df[[c for c in df.columns if not c[0].isdigit()] + [ordinal(i) for i in range(1, TOP_K + 1)]]
    df.to_csv(out / "seeds_top100.csv", index=False)
    stop[0] = True

    cfg = {
        "procedure": "agent/ott_qa_api_server.py::hybrid_search (L272-344), standalone transcription (seed_common.py)",
        "fusion": "boost * (0.5 * 1/(1+sqL2) + 0.5 * bm25/max_bm25), boost=1.5 if in HNSW top-2k AND bm25>0",
        "top_k_saved": TOP_K, "hnsw_candidates": 2 * TOP_K,
        "hnsw": {"class": "ott_qa_retrieval_pipeline.HNSWIndex", "M": 32, "efConstruction": hnsw.index.hnsw.efConstruction,
                 "efSearch": hnsw.index.hnsw.efSearch, "vectors": "stored docs_chunks_1 embeddings (384-d)"},
        "bm25": {"class": "ott_qa_retrieval_pipeline.BM25Index (rank_bm25.BM25Okapi k1=1.5 b=0.75 eps=0.25)",
                 "text": "source_info.source_name | content", "tokenizer": "re.findall(r'\\w+', lower)"},
        "query_encoder": f"SentenceTransformer('{C.MODEL_NAME}', device='cpu'), raw question text",
        "corpus": {"documents": len(corpus), "tables": 0, "equals_graph_nodes":
                   corpus == set(pd.read_csv(C.GRAPH_IDS_CSV).chunk_id)},
        "questions_source": str(C.QUESTIONS_CSV), "questions": len(df),
        "gold_source": "bm25_only_results.csv gold_docs (doc_gold = gold IDs present in the 21,742-doc corpus)",
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "PYTHONHASHSEED")},
        "input_sha256": {"bm25_only_results.csv": sha256(C.QUESTIONS_CSV),
                         "graph_indexing_filtered.csv": sha256(C.GRAPH_IDS_CSV)},
        "sage_commit": "405d5f59c5f7ab612deb2fc7a4b94851c2c590ad",
        "runtime_s": {"indices": round(t_index, 1), "queries": round(t_query, 1), "total": round(time.time() - t0, 1)},
        "peak_rss_gb": round(peak[0] / 1e9, 2),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
