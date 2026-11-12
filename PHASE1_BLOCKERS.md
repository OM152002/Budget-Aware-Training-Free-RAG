# Phase 1 Blockers: Paper-Style SAGE Baseline

**Repository:** `CoRAL-ASU/sage` @ `405d5f5` (unmodified). All paths below are relative to `sage/`.
**Scope:** This document covers only the external artifacts and decisions we need before we can faithfully reproduce the OTT-QA SAGE baseline. It proposes no code changes.

The SAGE code reads every `output/...` path relative to the current working directory. None of these files ship with the repository.

---

## 1. Required external artifacts

| ID | Expected file / path | Consumed by (file · class.function) | Why it is required | Proceed without it? |
|---|---|---|---|---|
| **A1** | `output/full_pipeline/docs_chunks_1/*.json`: one JSON per document chunk with `chunk_id`, `content`, `embedding`, `source_info.source_name`, `metadata.topic`, `metadata.entities` (dict), `metadata.events` (dict) | `scripts/analysis/graph_enhanced_accuracy_analyzer.py` · `GraphEnhancedAccuracyAnalyzer._load_chunk_metadata` (L370); `scripts/analysis/graph_analysis_pipeline.py` · `GraphAnalysisPipeline._load_document_chunks` (L1240); `experiments/ott_qa/ott_qa_retrieval_pipeline.py` · `ChunkLoader.load_document_chunks` (L55) | These files are the retrieval corpus. The LLM-generated topic, entity and event fields feed both edge construction and neighbor reranking. **The code that generates chunk metadata is not in the repository.** | **No.** This is a hard blocker. |
| **A2** | `output/full_pipeline/table_chunks_with_metadata.json`: a list of `{chunk_id, content, metadata.{table_title, table_description, col_desc, entities, events}}` | `GraphEnhancedAccuracyAnalyzer._load_chunk_metadata` (L370); `GraphAnalysisPipeline._load_table_chunks` (L1277); `ChunkLoader.load_table_chunks` (L93) | OTT-QA mixes documents and tables, and the SAGE graph contains doc–table and table–table edges. | **Depends on decision D5.** The analyzer catches a missing file and continues with documents only. A doc-only run would not match the paper, but table retrieval is out of scope in the PRD (§4). |
| **A3** | `output/analysis_cache/knowledge_graph/unique_edges_20250812_164429.json` (an older run references `unique_edges_20250731_173102.json`): a list of `{source_chunk_id, target_chunk_id, weight?}` | `GraphEnhancedAccuracyAnalyzer._load_graph_data` (L337); `ott_qa_retrieval_pipeline.py` · `NeighborGraph.load_similarity_edges` (L319) | This file is the SAGE graph itself. It is produced by `GraphAnalysisPipeline._extract_edges_for_graph` (L2824), which needs A1, A2 and A4 **plus a CUDA GPU** (`_calculate_similarities_for_pairs`, L1587, raises `RuntimeError` without CUDA). | **No.** This is a hard blocker unless we get it from the authors or rebuild it from A1, A2 and A4 on CUDA hardware. |
| **A4** | `output/indexing_filtered.csv` with a `chunk_id` column | `GraphAnalysisPipeline.load_chunks` (L1197; `pd.read_csv` crashes if the file is missing); `GraphEnhancedAccuracyAnalyzer._load_indexing_filtered_data` (L355) and `_load_chunk_metadata` (chunks absent from the CSV are skipped) | It defines the chunk subset that is indexed and placed in the graph. How the subset was chosen is **unknown**. | **No.** It defines the corpus. |
| **A5** | `output/questions_dense_sparse_average_results1.csv` (analyzer default, L266). Variant names elsewhere: `output/dense_sparse_average_results (1).csv` (`ott_qa_retrieval_pipeline.py` L38, `agent/ott_qa_batch_processor.py` L29) and `output/dense_sparse_average_results_filtered.csv`. Columns: `question_id`, `question`, `gold_docs`, `1st`, `2nd`, `3rd`, `4th` … `100th` | `GraphEnhancedAccuracyAnalyzer._load_csv_data` (L331), `get_csv_retrieved_docs` (L485), `extract_gold_docs` (L445) | It supplies the **seed ranking S** (initial hybrid retrieval) and the **gold labels** for every question. **The code that generates it is not in the repository.** | **No.** This is a hard blocker. Variant B has no other source of seeds or gold labels. |
| **A6** | `output/questions_metadata_output.json`: `{question_text: {entities: [...], events: [...]}}` | `GraphEnhancedAccuracyAnalyzer._load_questions_metadata` (L361); used in `_calculate_document_similarities` (L659) and `_calculate_table_similarities` (L702) as entity/event bonuses (+0.1 for an exact match, +0.05 for a substring match) | It provides the question-side entity/event bonuses in neighbor reranking. The closest code, `experiments/ott_qa/question_metadata.py`, calls gpt-4o-mini with hard-coded Windows paths and an empty API key, and never writes this file. | **The code runs without it, but the result is not faithful.** The bonuses become 0. |
| **A7** | `datasets/ott-qa-full-data.json`: records with `id` and `structural_link` | Variant A only: `NeighborGraph.load_structural_links` (L335). **Variant B never loads structural links.** | The paper (§2.1) says parent/structural links are kept as edges. | **Unknown.** We first need to know whether A3 already contains structural edges (Q3). |
| **A8** | Raw OTT-QA input `datasets/ott-qa.json`, the chunk-metadata prompt and model, and a CUDA GPU | `src/pipeline/rag_pipeline.py` · `RAGPipeline.process_from_json_metadata`; `GraphAnalysisPipeline` | We need these only if we **rebuild A1–A4 from scratch**. The chunking path is also broken in the repository: `EmbeddingService.generate_single_embedding` and `compute_similarity` are undefined, and `torch.cuda.Event` crashes on CPU. | Only if the authors cannot provide A1–A5. Rebuilding requires approval and changes how faithful the baseline is. |

**Minimum set for a faithful Variant B run:** A1, A3, A4 and A5 are required; A2 and A6 are required for full fidelity. Once those exist, Variant B runs on the M3 Mac on CPU. Phase 1 verified this on synthetic inputs.

---

## 2. Variant B: the candidate paper-style SAGE baseline

All of Variant B lives in **`scripts/analysis/graph_enhanced_accuracy_analyzer.py`**, class **`GraphEnhancedAccuracyAnalyzer`** (L262). It matches the paper's Listing 1, where the final context is `S ∪ TopK(BM25DenseRank(q, N(S)), k′)`.

| Step | Function (line) | Verified behavior |
|---|---|---|
| Load inputs | `__init__` (L265), `load_all_data` (L319) | Reads A5, A3, A4, A6 and A1/A2. The k-pairs are hard-coded in `self.k_value_tests`. |
| Graph | `_load_graph_data` (L337) | Builds an undirected `networkx.Graph` from A3. The edge `weight` is stored but never used to select neighbors. |
| Seeds S (top-k) | `get_csv_retrieved_docs` (L485) | Takes the first `csv_k` entries of columns `1st`, `2nd`, … from A5. |
| Candidates N(S) | `get_graph_neighbors_for_retrieved_chunks` (L591) | Takes the union of 1-hop neighbors of all seeds, minus the seeds themselves. |
| Neighbor scoring | `_calculate_hybrid_similarities` (L515) | **0.40 × embedding + 0.60 × BM25.** The docstring says 50/50. |
| Embedding part | `_calculate_neighbor_comprehensive_similarities` (L620), `_calculate_document_similarities` (L659), `_calculate_table_similarities` (L702), `GPUEmbeddingService` (L129; `all-MiniLM-L6-v2`, falls back to CPU) | Documents score cos(q, content) + cos(q, topic) + entity/event bonuses. Tables score cos(q, content + description) + cos(q, description) + cos(q, columns) + bonuses. Embeddings are recomputed at query time; the stored chunk embeddings are not used. |
| BM25 part | `_calculate_bm25_similarities` (L564), `BM25` (L83), `preprocess_text_for_bm25` (L66) | BM25 is fit **only on the candidate set**, with unsmoothed IDF. Scores can be negative; Phase 1 observed −0.59. Scores are divided by the maximum score. |
| Select k′ | end of `get_graph_neighbors_for_retrieved_chunks` | Sorts by hybrid score and keeps the top `graph_k`. |
| Evaluate | `analyze_question_accuracy` (L1008), `run_k_value_test` (L1071), `analyze_csv_vs_graph_comparison` (L765), `run_csv_vs_graph_analysis` (L832), `main` (L1179) | Computes **hit** (any gold item in S ∪ selected), not Recall@k. It also compares "CSV k + graph k′" against "CSV k + k′". |

**Graph builder Variant B depends on (A3 producer):** `scripts/analysis/graph_analysis_pipeline.py`, class `GraphAnalysisPipeline`. The call chain is `main` (L3084) → `load_chunks` → `build_hnsw_index` (L1390) → `analyze_chunk_similarities(k_neighbors=200)` (L1443) → `generate_analysis_reports` (L1615), which calls `_analyze_doc_doc_outliers` (L2084), `_analyze_doc_table_outliers` (L2164), `_analyze_table_table_outliers` (L2275) and `_store_outliers_for_graph_building` (L3029) → `build_knowledge_graph` (L2539) → `_extract_edges_for_graph` (L2824).

**For contrast, Variant A (not recommended):** `experiments/ott_qa/ott_qa_retrieval_pipeline.py` · `HybridRetriever.retrieve` (L453). Its seeds are the intersection of the best 20% of HNSW results and the top 20% of BM25 results, so it can return fewer than k seeds. It then expands to *all* 1-hop neighbors (similarity edges plus structural links) and reranks everything to top-k with 1.5 × (0.5·dense + 0.5·BM25).

---

## 3. Decisions required

| ID | Decision | Notes |
|---|---|---|
| D1 | Use Variant B as Baseline C ("Original SAGE")? | **Recommended.** Confirm with the authors (Q1). |
| D2 | Edge rule: the code's rule or the paper's rule? | The code uses the 99th percentile with AND logic, plus `entity_count > 0` for doc–doc edges (L2101–2102); doc–table and table–table edges also use the 99th percentile (L2193–2195, L2303–2305). The paper uses the 95th percentile with OR logic (§2.1, Table 5). This only matters if we rebuild A3. |
| D3 | Metric | The code computes hit@k. The PRD (§25) and the paper use Recall@k, so we need a recall evaluator that does not change Variant B's retrieval. |
| D4 | k and k′ values | The paper's Figure 4 sweeps k from about 2 to 100. The code hard-codes the pairs in `k_value_tests` (L300) and `run_csv_vs_graph_analysis` (L838). Which pairs produced the reported numbers? |
| D5 | Include tables? | OTT-QA and the SAGE graph mix documents and tables, but the PRD excludes table retrieval. Choose a doc-only subset or full OTT-QA. |
| D6 | Source of artifacts | Get A1–A6 from the authors (preferred), or rebuild them (requires A8 and approval). |
| D7 | Gold granularity | Are `gold_docs` in A5 chunk IDs or document IDs? This determines how recall is computed. |

---

## 4. Questions for the SAGE authors

1. Is `GraphEnhancedAccuracyAnalyzer` (Variant B above) the implementation behind the paper's OTT-QA results, rather than `HybridRetriever` in `ott_qa_retrieval_pipeline.py`?
2. Can you share A1–A6, specifically `docs_chunks_1/`, `table_chunks_with_metadata.json`, `unique_edges_20250812_164429.json`, `indexing_filtered.csv`, the seed CSV and `questions_metadata_output.json`? If not, can you share the code and prompts that generate them?
3. Does the edges file A3 already include parent/structural links? Variant B never loads them separately.
4. Which edge rule produced the paper's graph: the 95th percentile with OR (paper) or the 99th percentile with AND plus entity match (code)?
5. How was the seed CSV produced (dense model, BM25 settings, how the scores were combined), and how was `indexing_filtered.csv` selected?
6. Which (k, k′) pairs and which metric (hit vs. recall) produced the Figure 4 and Table 8 numbers?
7. Are the 0.40/0.60 weights (versus the 50/50 docstring) and the BM25 fitted only on candidates intended?
