# Seed Retrieval Plan: Regenerating SAGE Initial Seeds over the 21,742-Document Graph

**Status:** INVESTIGATION ONLY. No rankings have been regenerated and no code has been written. SAGE is unmodified (`405d5f5`).
**Corpus:** the exact 21,742 document chunks of the accepted graph (`graph_reconstruction/runs/docgraph_k200_q99_20261023`).
Labels: **VERIFIED** means read in code or measured. **PROPOSED** means part of the plan. **UNKNOWN** means not determinable from the repo or artifacts.

## 1. SAGE files and functions involved (VERIFIED)

**Seed consumer** (the format we must produce): `scripts/analysis/graph_enhanced_accuracy_analyzer.py`.
- `get_csv_retrieved_docs` (L485–493) reads `1st, 2nd, 3rd, 4th … 100th` (chunk IDs) from the seed CSV.
- `extract_gold_docs` (L445) and `analyze_question_accuracy` (L1011–1013) read `question_id`, `question` and `gold_docs`.

**The repo has no producer** for `dense_sparse_average_results.csv`, `bm25_only_results.csv` or `dense_sparse_sequential_results.csv`. Repo-wide grep finds them only as inputs. The repo contains exactly two dense+BM25 fusion procedures, both for OTT-QA, and they share the same indices:

| Component | File · symbol | Behavior |
|---|---|---|
| Corpus loader | `experiments/ott_qa/ott_qa_retrieval_pipeline.py` · `ChunkLoader.load_document_chunks` (L55–91) | Uses the **stored** `embedding` per document chunk. BM25 text = `f"{source_info.source_name} \| {content}"` (L79–83) |
| Dense index | same file · `HNSWIndex.build` / `search` (L185–224) | `faiss.IndexHNSWFlat(384, M=32)`, `efConstruction=200`, `efSearch=100`; L2-normalized (cosine-equivalent L2; returns squared L2 distances) |
| Sparse index | same file · `BM25Index.build` / `_tokenize` / `search` (L253–292) | `rank_bm25.BM25Okapi` with defaults (k1=1.5, b=0.75, epsilon=0.25; v0.2.2 installed). Tokenizer: `re.findall(r'\w+', text.lower())`, with no stop-word removal |
| Query encoder | same file · `HybridRetriever._embed_query` (L412–418); `agent/ott_qa_api_server.py` · `embed_query` (L168–171) | `SentenceTransformer('all-MiniLM-L6-v2').encode(query)` on the **raw** question, L2-normalized before search |
| Index builder / cache | same file · `main()` steps 3–4 (L553–584) | Builds and caches `output/ott_qa_pipeline_cache/{hnsw,bm25}` |
| **Fusion A: flat hybrid ranking** | `agent/ott_qa_api_server.py` · `hybrid_search` (L272–344), loading the cache via `load_hnsw_index` / `load_bm25_index` (L100–137) | No graph. Returns a ranked list of up to 200 chunks with scores |
| Fusion B: hybrid + graph expansion | `ott_qa_retrieval_pipeline.py` · `HybridRetriever.retrieve` (L453–523) | Percentile filters, intersection, 1-hop graph expansion, rerank; returns ≤ 20. This is **Variant A** from Phase 1, a full retriever, not a seed ranker |

## 2. How dense and BM25 are combined (VERIFIED)

**Fusion A, `hybrid_search` (L272–344):** a flat, graph-free hybrid ranking of the whole corpus.
1. `k = min(request k, 200)`. HNSW returns the top **2k** results: dense score `h = 1 / (1 + d)`, where `d` is FAISS squared-L2 on normalized vectors (= 2 − 2·cos) (L287–294).
2. BM25 scores **every** chunk (`get_scores`). It keeps scores > 0 and normalizes them by the corpus-wide maximum: `b = score / max` (L297–308).
3. Candidates = union of the two sets. `combined = boost · (0.5·h + 0.5·b)`, with `boost = 1.5` if the chunk is in both sets, else 1.0 (L311–320).
4. Sort descending and return the top k (L325–344), with `combined_score`, `hnsw_score`, `bm25_score` and `rank`.

**Fusion B, `HybridRetriever.retrieve` (L453–523):**
1. HNSW top 100, keeping distances ≤ the 20th percentile. BM25 top 100, keeping scores ≥ the 80th percentile of the nonzero scores.
2. Seeds = the **intersection** of the two filtered sets, with fallbacks at L483–491.
3. Expand the seeds with 1-hop neighbors from the graph plus structural links.
4. Rerank by `boost·(0.5·h + 0.5·b)` (boost 1.5 for the intersection) and keep the top 20.

It can return fewer than k items (seen in Phase 1), and it mixes in graph expansion.

**Assessment:**
- Fusion A is the only in-repo procedure that yields a **ranked list of graph-free seeds of arbitrary depth**, which is what Variant B consumes. It literally *averages* dense and sparse scores (0.5/0.5), matching the artifact name "dense_sparse_average".
- Whether it is the exact producer of the provided CSV is **UNKNOWN**: the producing code is absent, and it could be checked empirically (§6, S6).
- Fusion B cannot serve as a seed source without changing what "seeds" means (it already includes graph neighbors).
- Neither "bm25_only" nor "sequential" has in-repo code. `bm25_search` (`ott_qa_api_server.py` L227–270) is the BM25-only counterpart, if a BM25-only baseline is wanted later.

## 3. Can it run directly on our provided chunks? Yes (VERIFIED feasibility, PROPOSED execution)

- **Dense:** the stored 384-d embeddings feed `HNSWIndex.build` unchanged (the same vectors the graph used; byte-identity verified in V4). No document re-embedding is needed. Only the 2,212 queries are encoded with MiniLM, which takes seconds.
- **BM25:** `"source_name | content"` over the 21,742 chunks. Every chunk has `source_info.source_name`.
  - Note: `\w+` keeps underscores, so source names become a few long tokens such as `2019_fivb_volleyball_men`. That is the released behavior, and it is preserved.
- **Scope:** documents only (your decision G1). The original pipeline also loaded the table chunks (L545–548), so our BM25 IDF statistics and dense candidate pool differ from the authors' by design.
- **How to invoke SAGE's code unmodified:**
  - `ChunkLoader`, `HNSWIndex` and `BM25Index` import cleanly from `ott_qa_retrieval_pipeline.py`; only `main()` touches fixed paths.
  - `hybrid_search` is a Flask route that reads module globals, and **Flask is not installed** in `.venv-sage`. Option S4a (recommended): install `flask`, set the module globals (`hnsw_index`, `chunk_ids_list`, `bm25_index`, `chunks_data`, `embedding_model`), and call the unmodified function inside `app.test_request_context(json={...})`. Option S4b: a line-for-line transcription of L272–344, validated against S4a on a sample.
- **Environment:** `OMP_NUM_THREADS=1` (the faiss/torch libomp crash found earlier). `PYTHONHASHSEED=0`, because candidates are a Python `set` and `list.sort` is stable, so ties would otherwise be ordered by hash randomization. A single-threaded HNSW build for determinism.
- **VERIFIED detail:** with `efSearch=100` and `k·2=200`, FAISS still returns 200 results, but they are not identical to a larger `efSearch`. We keep the code's `efSearch=100` and `efConstruction=200`. This is a separate index from the graph builder's (M=32/1000/500), exactly as in SAGE.
- **Cost estimate:** HNSW build of a few seconds; BM25 build of about 10 s. Per query, `get_scores` over 21,742 documents takes tens of ms, so about 2–5 minutes for 2,212 queries on CPU.

## 4. Query file required (VERIFIED)

SAGE reads questions from `output/dense_sparse_average_results (1).csv`, columns `question_id` and `question` (`ott_qa_retrieval_pipeline.py` L38, L629–632; `agent/ott_qa_batch_processor.py` L29, L441–448).
- The provided CSVs all contain the **same 2,212 questions** (unique `question_id`s and question texts, identical across all six files).
- **PROPOSED:** take `question_id` and `question` from the provided CSV and **ignore its rank columns**.
- The raw OTT-QA dev JSON is not needed and is not in the repo (`sage/datasets/` doesn't exist).

## 5. Gold labels available (VERIFIED)

`gold_docs` (chunk-level IDs) exists in every ranking CSV, in two versions:

| Gold source | Gold IDs | Document gold resolving to our 21,742 | Table gold (out of scope) | Unresolved |
|---|---|---|---|---|
| `dense_sparse_average_results.csv` (and both community files) | 3,110 | 1,626 | 708 | 776 (657 stale hash, 119 absent) |
| **`bm25_only_results.csv`** (= `dense_sparse_sequential_results.csv`) | 3,091 | **2,294 (100% of its document gold)** | 708 | 89 (no matching source document in the corpus; they look like table chunk IDs not shipped in the table file) |

- 586 questions have different gold between the files. In 546 of them the only difference is the hash suffix. For **every** question, the average file's resolvable document gold is a subset of the BM25 file's gold.
- **The BM25-file gold is the same labeling re-keyed to the shipped chunk version** and is the compatible choice.

Per-question coverage with the BM25-file gold:

| | Questions |
|---|---|
| At least one document gold | **1,728** |
| All gold are document chunks (no tables) | 1,678 |
| Gold includes ≥ 1 table chunk | 477 |
| **No document gold at all (table-only)**: cannot be evaluated on a documents-only graph | 484 |

## 6. Missing artifacts and decisions

**Missing (UNKNOWN):**
- the code and parameters that produced `dense_sparse_average_results.csv`;
- whether the authors' index included tables (the in-repo pipeline does);
- the question metadata (A6), which isn't needed for seeds.

No other artifact is missing for seed regeneration.

**Decisions:**
- **S1, procedure:** Fusion A, `agent/ott_qa_api_server.py::hybrid_search` (recommended: flat, graph-free, 0.5/0.5 dense/sparse average with a 1.5 both-sets boost). Fusion B is not a seed source.
- **S2, depth:** request `k=100`, which matches the CSV's 100 rank columns and Variant B's `min(k+1, 101)`. The code then uses HNSW depth 2k=200. `efSearch=100` and `efConstruction=200` stay as coded.
- **S3, gold and evaluation set:** use `bm25_only_results.csv` gold.
  - Recommended primary metric: Recall@k over **document gold only**, on the **1,728** questions with ≥ 1 document gold.
  - Also report the 1,678-question all-document subset, and all 2,212 questions with table gold counted as misses, for transparency.
- **S4, invocation:** install `flask` and call the unmodified `hybrid_search` (S4a, recommended), or use a validated transcription (S4b).
- **S5, determinism:** `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, a single-threaded HNSW build; run twice to confirm identical rankings.
- **S6, fidelity diagnostic (no tuning):** on the provided CSV's seed IDs that resolve to our corpus, report how closely our ranking reproduces the author ranking: overlap@{5,10,20,100} and the fraction of their top-k we reproduce. Expect differences because our corpus excludes tables and intro chunks. This only measures fidelity; nothing would be tuned to it.
- **S7, output format:** a new seed CSV with `question_id, question, gold_docs` (from the BM25 file), `1st…100th` (chunk IDs), plus side files with the per-rank `combined/hnsw/bm25` scores. It would be written outside `sage/` and consumable by Variant B unchanged.
- **Methodological note (preserved, not changed):** queries are embedded raw, while document vectors were built from stop-word-stripped text. This asymmetry is SAGE's released behavior.
