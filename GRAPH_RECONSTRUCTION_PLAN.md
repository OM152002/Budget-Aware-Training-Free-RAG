# Graph Reconstruction Phase: CPU/MPS Port Plan for SAGE's Graph Builder

**Status:** PLAN ONLY, awaiting approval. SAGE is unmodified (`sage/` @ `405d5f5`, `git status` clean). No port, edge rule or SLM code has been written.
**Target:** rebuild the SAGE graph (document–document edges first) from the author-provided `docs_chunks_1` (+ `table_chunks_with_metadata.json`), using `scripts/analysis/graph_analysis_pipeline.py` with **no algorithm changes**. The 99th-percentile rule stays as released until the authors answer (D2).
**Evidence:** line numbers refer to `sage/scripts/analysis/graph_analysis_pipeline.py` unless another file is named. Probe scripts are in `Project/phase1_repro/artifact_audit/` (`probe_graph.py`, `probe_stopemb.py`); they are read-only and do not modify SAGE.
Labels: **VERIFIED** means read in code or measured on this Mac. **PROPOSED** means part of the plan. **UNKNOWN** means not determinable here.

---

## 1. Existing SAGE graph algorithm (VERIFIED from code)

Entry point `main()` (L3084–3122): `load_chunks` → `build_hnsw_index` → `analyze_chunk_similarities(k_neighbors=200)` → `generate_analysis_reports` → `analyze_connected_components` → `build_knowledge_graph`.

**Step 0: Load chunks** (`load_chunks` L1197)
- Reads `./output/indexing_filtered.csv` unconditionally (L1201). It crashes if the file is missing. If the `chunk_id` column is absent or the file is empty, it falls back to "keep all" (L1203–1207, L1265, L1306).
- If `./output/analysis_cache/processed_chunks.pkl` exists, it returns that cached pickle (L1210–1217), which is a stale-cache trap.
- Documents (`_load_document_chunks` L1240): `content_embedding = data['embedding']` (the **stored** 384-d vector), `topic = metadata.topic`, `topic_embedding=None`, `entities = metadata.entities`, `metadata` = the full metadata dict.
- Tables (`_load_table_chunks` L1277): `content_embedding=[]`, `topic = metadata.table_description`, `column_descriptions = metadata.col_desc`.
- `_process_embeddings` (L1322) embeds with `EmbeddingService` (`all-MiniLM-L6-v2`, `src/core/models.py` L45; `normalize_embeddings=True`). It embeds (a) table `content` (974 texts), (b) **every chunk's topic** (21,742 doc topics + 974 table descriptions = 22,716 texts) and (c) table column descriptions (4,543 texts). Entity embeddings are skipped (L1371).

**Step 1: HNSW index** (`build_hnsw_index` L1390)
- `faiss.IndexHNSWFlat(d, 32)` with `efConstruction=1000` and `efSearch=500` (L1413–1415). The metric is the FAISS default, **L2**, on L2-normalized vectors, so the ranking is identical to cosine.
- The index covers **all chunks with a content embedding (docs and tables)**, so tables occupy some of each document's 200 neighbor slots.
- A GPU branch (L1418–1429) runs only if `faiss.get_num_gpus() > 0`. It has a CPU fallback on any exception (L1430–1433). FAISS has no GPU HNSW, so `index_cpu_to_gpu` on an `IndexHNSWFlat` most likely threw and the authors' index was built on the CPU anyway. This is high-confidence knowledge, **not verified on a GPU**.

**Step 2: Candidate pairs** (`analyze_chunk_similarities` L1443)
- Cache traps: an existing `./output/batch_results/batch_*.csv` (L1448–1459) or `similarity_analysis_k200.pkl` (L1462–1468) short-circuits the computation.
- `_find_all_neighbors_parallel` (L1489) runs a ThreadPool with 128 threads. Each chunk is searched one query at a time with `search_k = k+1 = 201`; self is dropped and the list truncated to 200 (`_find_k_neighbor_ids` L1537–1565).
- `_generate_unique_pairs` (L1571): an undirected pair is stored as `(min_id, max_id)` by **lexicographic string order**, deduplicated through a `set`, and returned as `list(set)` (arbitrary order).

**Step 3: Similarity features** (`_calculate_similarities_for_pairs` L1587 → `GPUAcceleratedSimilarityCalculator`)
1. `precompute_embeddings` (L154): stacks content and topic vectors into tensors and **re-normalizes** them (L180–181). `_precompute_table_metadata_embeddings` (L226) embeds each table title, summary and description **one text per call** (L268/277/286; 3 × 974 calls).
2. `_precompute_all_similarities` (L293): a Python loop over all pairs (`_calculate_all_similarities_for_pair` L336).
   - **Every pair whose chunks both have entities:** `_fuzzy_entity_match_gpu` (L693), with threshold 0.75. Names are normalized (lowercase, collapsed spaces, punctuation stripped; L638). The similarity is `0.7·Jaccard(char sets, spaces removed) + 0.3·(min_len/max_len)` (L646–686). Matching is **greedy in `names1` order**, without reuse of names in `names2`, accepting `max ≥ 0.75` (L715–730). The result is stored for both orientations.
   - From that it computes `entity_count` (the number of matches) and the per-match `relationship_overlap = |rel1 ∩ rel2| / max(|rel1|, |rel2|, 1)`, using `make_hashable` on the relationship lists (L785–806).
   - **Doc–doc pairs only:** `_calculate_event_overlap` (L823) uses the same fuzzy matcher on event names and yields `event_count`.
   - Doc–table and table–table pairs also get the `torch.mm` 1×384 lookups (L343–406), which don't apply to doc–doc pairs.
3. `calculate_similarities_batch` (L453) uses 10,000-pair batches on non-CUDA (L152). `_calculate_batch_similarities_gpu` (L490) computes:
   - `content_similarity = max(0, Σ(eᵢ·eⱼ))` over the normalized stored embeddings (L516–518, L546);
   - `topic_similarity = max(0, Σ(tᵢ·tⱼ))` over the normalized MiniLM topic embeddings (L521–523, L545);
   - `column_similarity = 0` for doc–doc (L560–561);
   - doc–doc extras from `_calculate_doc_doc_features` (L921): `entity_count`, `entity_relationship_overlap` (the mean of per-match overlaps) and `event_count`.
   - Each batch is written to `./output/batch_results/batch_NNNNNN.csv` (L1085), and the run is cached as a pickle (L1484).

**Step 4: Percentiles and edge selection for doc–doc** (`generate_analysis_reports` L1615 → `_generate_outlier_analysis` L2061 → `_analyze_doc_doc_outliers` L2084)
- The thresholds are computed **over doc–doc candidate pairs only**: `t_topic = quantile(topic_similarity, 0.99)` and `t_content = quantile(content_similarity, 0.99)` (pandas default linear interpolation; L2101–2102).
- An edge is kept if `(content > t_content) AND (topic > t_topic)` with strict `>` (L2107; reason `high_content_and_topic`), **or** if `entity_count > 0` regardless of similarity (L2114; reason `entity_match`).
- `_store_outliers_for_graph_building` (L3029) uses `edge_id = f"{source}_{target}"` and the fields `semantic_similarity = content_similarity`, `topic_overlap = topic_similarity`, `topic_similarity`, `content_similarity`, `entity_relationship_overlap`, `event_count` and `entity_count`.
  - **Released-code quirk:** the stored `entity_count` is read from `self.precomputed_similarities`, which `GraphAnalysisPipeline` never defines (L3045). It is **always 0** in the edges file, even though the row's `entity_count` was > 0.
  - Leiden's composite weight (`_calculate_composite_weight` L2784, used at L2711–2714) therefore omits the entity term.
  - A faithful port keeps this quirk; it is flagged for the authors.

**Step 5: Edge extraction** (`build_knowledge_graph` L2539 → `_extract_edges_for_graph` L2824)
- Deduplicates by `edge_id` (first wins, so an AND edge keeps the reason `high_content_and_topic`).
- Filters edges to loaded chunk IDs and writes `analysis_cache/knowledge_graph/unique_edges_<timestamp>.json` (L2858–2861). **This is the A3 file Variant B consumes**; it has no `weight` key, so Variant B uses 1.0.
- Then `_detect_communities_with_leiden` runs (L2682; skipped with a warning if `cdlib` is absent, which is the case in our venv), and `_create_knowledge_graph` / `export_to_json` (L2869, L2570) run after it.

Not part of this pipeline: `datasets/ott-qa-full-data.json` structural links. The released graph builder **never adds the `structural_link` doc→table edges** (VERIFIED). Its only reference, L2921, copies `metadata.get('structural_link')` into a node's `SourceInfo`. In `docs_chunks_1` the field lives under `source_info`, not `metadata`, so even that copy is empty, and no edges are created from it.

---

## 2. CUDA-only operations

| # | Location | What it does | Fails on this Mac? | Truly needs CUDA? |
|---|---|---|---|---|
| C1 | L1591–1592 `if not torch.cuda.is_available(): raise RuntimeError` | Artificial guard | **Yes (hard stop)** | **No.** Everything after it already falls back to `cpu` (L111). |
| C2 | `src/processors/embedding_service_optimized.py` L123–139 `torch.cuda.Event` / `torch.cuda.synchronize()` in `_generate_embeddings_massive_batch` | Timing only | **Yes**: `RuntimeError: Tried to instantiate dummy base class Event` (verified). This path is taken whenever a call has **≤ 8,192 texts** (L74, L89–91): table content (974), column descriptions (4,543), and **each** single-text title/summary/description call (L268/277/286). | **No.** It is a timer around an ordinary `model.encode(...)`. |
| C3 | L194 `torch.cuda.memory_allocated(self.device)` | Logging | No (returns 0 on CPU, verified) | No |
| C4 | L483 `torch.cuda.empty_cache()` | Housekeeping | No (no-op, verified) | No |
| C5 | L148/L430 `_calculate_optimal_batch_size` | Batch sizing | Not reached (guarded by `is_available`); CPU uses 10,000 | No |
| C6 | L1418–1429 FAISS GPU | Index build | Not reached (`get_num_gpus()==0`); CPU branch used | No (and likely never ran on GPU, §1) |
| C7 | `embedding_service_optimized.py` L28 device = `cuda` else `cpu`; `_clear_gpu_cache` guarded | Device choice | No | No |
| C8 | `src/pipeline/parallel_document_processor.py` L935 `torch.amp.autocast('cuda')` | Chunk-time embedding (fp16) | Not used by the graph builder | Not needed (stored embeddings are reused) |

**Conclusion (VERIFIED):** no operation in the graph builder intrinsically requires CUDA. The tensor math is gathers plus row-wise dot products and 1×384 matrix multiplies. The expensive part (fuzzy entity/event matching) is pure-Python string and set work that runs on the CPU even on the original GPU machine (L646–686 builds Python lists and only then wraps them in a tensor). Only C1 and C2 actually block execution.

### Can the stored 384-d embeddings remove GPU embedding work? (VERIFIED)
- **Yes, for document content.** The pipeline already uses `data['embedding']` as is (L1257) and never re-embeds document content. The stored vectors are unit-norm (min/max norm 1.000000).
  - On 300 random chunks, `all-MiniLM-L6-v2(normalize=True)` applied to SAGE's stop-word-stripped content (the logic of `parallel_document_processor.py::_remove_stopwords_from_content` L979–1004, with the `ENGLISH_STOPWORDS` list) reproduces the stored vector with **median cosine 0.999999**. CPU and MPS give identical results.
  - 114 of 300 have cosine < 0.999 (minimum 0.955), likely chunks whose embedding was computed before a final merge. The cause is **UNKNOWN** and irrelevant, because the original builder used the stored vectors too.
  - Raw (unstripped) content does **not** match (median cosine 0.90), so **re-embedding `content` would be a silent substitution and must be avoided.**
- **No, for everything else.** Topic embeddings (22,716), table content (974), column descriptions (4,543) and table title/summary/description (3 × 974) are **not stored** and must be computed with `all-MiniLM-L6-v2` exactly as the code does (raw text, `normalize_embeddings=True`). This is small: measured at 22,716 topics in **about 9 s on MPS and 16 s on CPU**.

---

## 3. CPU/MPS port plan (PROPOSED)

**Principle:** do not edit the SAGE repo. A runner script outside the repo (`Project/graph_reconstruction/run_sage_graph_builder.py`) imports the **unmodified** `GraphAnalysisPipeline` and applies two small, logged monkeypatches that only remove CUDA dependencies (§6). All algorithmic code (HNSW parameters, k=200, fuzzy matcher, thresholds, AND/OR rule, edge dedup) runs from SAGE's own source.

| Stage | Device | Notes |
|---|---|---|
| Load 21,742 doc JSONs + 974 tables | CPU | About 15–20 s measured |
| `_process_embeddings` (MiniLM: topics, table content, columns) | CPU (default after C2 fix); MPS optional | Keep CPU as the canonical run; MPS gave identical results to 1e-6 in probes but adds a device variable |
| HNSW build/search | **FAISS CPU** (`faiss-cpu 1.15.1`) | VERIFIED feasible, §4 |
| Pair features | CPU (`calc.device` resolves to `cpu`, L111) | The hot loop is Python string matching, where MPS gives no benefit. The 10,000-pair dot-product gathers are trivial in NumPy/torch on CPU |
| Percentiles, edge extraction | CPU (pandas) | Unchanged |
| Leiden | Skipped (no `cdlib`) unless approved | Not needed for the edges file; Leiden is also nondeterministic |

**Which features need what (Q4/Q5):**

| Feature | Computation | CPU/MPS efficient? |
|---|---|---|
| `content_similarity` | dot product of stored normalized 384-d vectors | Yes. Vectorized; ~3M pairs × 384 in 10k-row batches takes seconds |
| `topic_similarity` | dot product of MiniLM topic vectors | Yes, same as above |
| `entity_count`, `entity_relationship_overlap` | greedy fuzzy name matching plus relationship set overlap | CPU Python, **52 µs/pair measured** with the unmodified SAGE method |
| `event_count` | greedy fuzzy event-name matching | CPU Python, **24 µs/pair measured** |
| `column_similarity` | 0 for doc–doc | n/a |

**Determinism (PROPOSED):** FAISS HNSW construction with OpenMP is order-nondeterministic. Run the canonical build with `faiss.omp_set_num_threads(1)` (environment configuration, not a code change) so the candidate set is reproducible run to run. Measured CPU HNSW recall@201 against exact search is **0.9981**, so any HNSW build differs from exact k-NN, and from the authors' build, on about 0.2% of neighbors.

**Scope decision (needs approval):** run the **full released pipeline (docs + tables)**, then use the doc–doc edges. A docs-only index would change every document's top-200 list, because tables occupy neighbor slots in the shared index (§1 Step 1). It would no longer be the released candidate-generation algorithm.

---

## 4. Estimated compute (M3 Air, 8 cores, 16 GB)

Measured with real `docs_chunks_1` on this Mac (docs-only index; tables add about 4.5% more nodes):

| Quantity | Value |
|---|---|
| HNSW build (M=32, efC=1000), 21,742 × 384 | **5.4 s** |
| Search, all chunks, k=201, efS=500 | 4.4 s batched; **0.89 ms/query** single-query (SAGE style; about 19 s single-threaded, less with its ThreadPool) |
| HNSW peak RSS | 1.1 GB |
| **Unique undirected doc–doc candidate pairs at k=200** | **3,002,536** (bounds: 2.17M if every neighbor relation is mutual, 4.35M if none is). Expected with tables: about 3.1–3.3M total pairs, about 3.0M of them doc–doc |
| Candidate pairs with `entity_count > 0` (sample of 20k) | **14.6%** → about 440k doc–doc `entity_match` edges, plus AND edges (≤ 1% ≈ ≤ 30k). Order of magnitude consistent with the paper's 751,836 total edges (Table 8, all edge types) |
| Content-similarity quantiles (200k sampled pairs) | q99 ≈ 0.694, q95 ≈ 0.578, median 0.413 |

Projected full run (PROPOSED estimate; Python-bound stages extrapolated from the measured per-pair costs):

| Stage | Time | Peak RAM |
|---|---|---|
| Load + MiniLM embeddings (~28k texts plus ~2.9k single calls) | 1–2 min | ~1.5 GB (model + chunks) |
| HNSW + pairs | < 1 min | ~1.5 GB |
| `_precompute_all_similarities` (~3.2M × ~80 µs) | **~4–6 min** | `entity_fuzzy_matches` dict holds ~6.4M keys (both orientations) → **~2–3 GB** |
| `calculate_similarities_batch` + 320 batch CSVs | ~2–4 min | 3.2M `SimilarityMetrics` objects → **~1.5–2 GB** |
| Pickle cache + DataFrame | ~1–2 min | +1–1.5 GB transient |
| Reports: KDE histograms and 3M-point scatter plots at 300 dpi | ~5–15 min | ~1 GB |
| Outlier JSONs with content (~450k doc–doc rows) | ~3–6 min | **~1–3 GB** list; ~1 GB+ file on disk |
| KnowledgeGraph build/export | UNKNOWN (minutes) | ~1–2 GB |
| **Total** | **~20–40 min** | **peak ~6–10 GB**: feasible on 16 GB with nothing else heavy running; swap is the main risk |

Disk: about 3–5 GB for batch CSVs, the pickle cache, outlier JSONs and graph exports.

---

## 5. Validation plan (PROPOSED)

The original CUDA similarity values **no longer exist** (the authors confirmed the graph artifacts are gone), so the port cannot be diffed against the original run. Validation therefore establishes that (a) the port computes exactly what the released code computes, and (b) device and precision choices don't move any edge decision.

| # | Check | Method | Pass criterion |
|---|---|---|---|
| V1 | Port == unmodified SAGE math | On a fixed random sample of 2,000 candidate pairs, compare the port's `SimilarityMetrics` with an **independent float64 NumPy reimplementation** of L516–523 (dot products) and a direct call of SAGE's own `_fuzzy_entity_match_gpu` / `_calculate_event_overlap` | content/topic `max abs Δ ≤ 1e-6`; `entity_count`, `event_count`, `entity_relationship_overlap` **exactly equal** |
| V2 | CPU vs MPS (vs CUDA if available) | Run the calculator on the same 2,000 pairs with device `cpu` and `mps`. **Optional CUDA reference:** the same script on a CUDA machine (for example ASU Sol or Colab) with the unmodified `_calculate_similarities_for_pairs` | `max abs Δ ≤ 1e-5`; integer features identical |
| V3 | Topic embeddings: device invariance | Encode 1,000 topics on CPU vs MPS (vs CUDA if available) | min cosine ≥ 0.99999 |
| V4 | Stored embeddings used as is | Assert that every doc's `content_embedding` is byte-identical to `docs_chunks_1[*].embedding` after load | 100% |
| V5 | Edge-decision robustness | Count doc–doc pairs with \|content − t_content\| < 1e-5 or \|topic − t_topic\| < 1e-5 (pairs whose edge status a float-precision change could flip) | Report the count; expected to be tiny relative to ~450k edges |
| V6 | Determinism | Two full runs with `omp_set_num_threads(1)` for the HNSW build | Identical pair set, thresholds and edges file (modulo timestamp) |
| V7 | Candidate quality | HNSW top-200 vs exact `IndexFlatIP` on 1,000 queries | Recall ≥ 0.998 (measured 0.9981) |
| V8 | Released-code reproduction of known quirks | Assert that the edges file's `entity_count` is 0 for all edges (the L3045 quirk) and that thresholds use `quantile(0.99)` over doc–doc rows only | Matches §1 |
| V9 | Structural sanity vs paper | Node count, edge counts per type and reason, degree distribution, isolated docs; compare with Table 8 (22,886 nodes / 751,836 edges) | Report the deviation and explain it (node gap 22,716 vs 22,886, known from the artifact audit) |
| V10 | Downstream smoke (later phase) | Variant B on the reconstructed edges with the provided seed CSV | Runs; numbers reported, not tuned |

---

## 6. Required code changes (PROPOSED, all outside `sage/`)

**Zero edits to SAGE source.** Vanilla SAGE stays runnable and diffable. The runner applies:

1. **Patch P1 (removes C2):** override `OptimizedEmbeddingService._generate_embeddings_massive_batch` with an identical body in which `torch.cuda.Event`/`synchronize` are replaced by `time.perf_counter()`. The `model_to_use.encode(texts, convert_to_tensor=True, device=self.device, batch_size=128, normalize_embeddings=True)` call is **unchanged** (`embedding_service_optimized.py` L113–150).
2. **Patch P2 (removes C1):** override `GraphAnalysisPipeline._calculate_similarities_for_pairs` with the identical body minus the two-line CUDA guard (L1590–1592). It still constructs the unmodified `GPUAcceleratedSimilarityCalculator` (batch_size=10000, gpu_id=0), whose device resolves to `cpu` at L111.
3. **Environment and inputs, no code change:**
   - a clean working directory `Project/graph_reconstruction/run_<date>/` whose `output/` contains symlinks `full_pipeline/docs_chunks_1 → sage_hf_artifacts/extracted/docs_chunks_1` and `full_pipeline/table_chunks_with_metadata.json → sage_hf_artifacts/raw/…`, with no stale `batch_results/` or `analysis_cache/` pickles (the caches at L1210, L1448 and L1462 would otherwise short-circuit the run);
   - `output/indexing_filtered.csv`, needed because L1201 crashes without it. **Decision G2:** list all 22,716 provided chunk IDs (equivalent to "no filter"), since the original file is unavailable;
   - the import-order workaround from Phase 1 (`import src.pipeline` before `src.processors`);
   - `faiss.omp_set_num_threads(1)` for the HNSW build (V6), and `os.chdir` into the run directory (all SAGE paths are cwd-relative).
4. **Optional, not applied unless approved:** installing `cdlib` for Leiden; MPS for MiniLM.
5. **Possible compatibility issues** (UNKNOWN until a dry run): Python 3.13 / pandas / pydantic behavior in `_create_knowledge_graph` / `_create_edge_metadata_from_data` (L2869–3027). The edges file is written **before** those steps (L2860), so A3 is produced even if they fail. Any further fix would be reported before it is applied.

**Explicitly not changed:** the 99th-percentile AND + `entity_count > 0` rule (pending the authors' D2 answer), k=200, the HNSW parameters, the fuzzy threshold 0.75, the L3045 entity-count quirk, and edge dedup. Structural links are not added to the graph (the released builder doesn't add them either); whether to add them is a separate, later decision.

### Decisions needed before implementation
- **G1, index scope:** full docs+tables pipeline (recommended, faithful), or docs-only (simpler, but alters candidate generation).
- **G2, `indexing_filtered.csv`:** use all 22,716 provided chunks (recommended; the original is lost).
- **G3, reports:** keep the released reporting and outlier-JSON side effects (faithful; +10–20 min and 1–3 GB RAM). Alternatively skip the plotting and content dumps through the runner; that changes no edge decisions.
- **G4, Leiden:** skip for now (recommended), or install `cdlib`.
- **G5, CUDA reference for V2/V3:** is a CUDA machine available (for example ASU Sol)? If not, validation relies on V1 and V3–V9.
