# Phase 1 Addendum: Audit of the Author-Provided SAGE Artifacts

**Source:** https://huggingface.co/datasets/prasham1515/SAGE (main branch, 10 files; README contains only `license: apache-2.0`).
**Local copy:** `Project/sage_hf_artifacts/raw/` (downloaded unchanged; the SHA-256 of every LFS file matches its Hugging Face OID). `docs_chunks_1.zip` is extracted to `Project/sage_hf_artifacts/extracted/docs_chunks_1/`.
**SAGE repo:** `Project/sage` @ `405d5f5`, unmodified (`git status` clean).
**Audit scripts and logs:** `Project/phase1_repro/artifact_audit/` (read-only inspection; no project code).
Labels: **VERIFIED** means measured directly from the files. **UNKNOWN** means the files cannot answer it.

---

## 1. Artifact inventory

| File | Size | Type | What it is (VERIFIED) | Maps to Phase 1 blocker |
|---|---|---|---|---|
| `docs_chunks_1.zip` | 123.7 MB | zip of 21,742 JSON files | Document chunks with an embedding and LLM metadata | **A1** |
| `table_chunks_with_metadata.json` | 8.6 MB | JSON list, 974 items | Table chunks with LLM metadata (no embedding) | **A2** |
| `dense_sparse_average_results.csv` | 20.9 MB | CSV, 2,212 × 110 | Per-question hybrid ranking, top 100 | **A5 candidate** (see §4) |
| `bm25_only_results.csv` | 4.9 MB | CSV, 2,212 × 20 | Per-question BM25 ranking, top 10 | Baseline output |
| `dense_sparse_sequential_results.csv` | 4.9 MB | CSV, 2,212 × 20 | Per-question "sequential" dense→sparse ranking, top 10 | Baseline output |
| `community_expanded_dense_sparse_results.csv` | 41.5 MB | CSV, 2,212 × 112 | Per-question ranking after community expansion, top 100 | Graph-method output |
| `community_k_plus_k1_dense_sparse_results.csv` | 41.6 MB | CSV, 2,212 × 113 | Same as above plus `retrieval_time_seconds` | Graph-method output |
| `enhanced_results_final_communities_2049.csv` | 1.5 MB | CSV, 2,212 × 48 | Per-question top-N community IDs plus recall/precision scores | Community evaluation |

**Not provided:** A3 (the edges file), A4 (`indexing_filtered.csv`), A6 (`questions_metadata_output.json`), any community-membership file (community → chunk IDs), `ottaqa-intro-*` chunks, and the code that produced any of these CSVs. A search for `bm25_only`, `dense_sparse_sequential`, `community_expanded`, `k_plus_k1`, `enhanced_results_final`, `top2_chunk_ids` and `table_id` across `sage/**/*.py` finds no matches.

All six CSVs have the same 2,212 `question_id`s with identical question text. **UNKNOWN:** whether this is the OTT-QA dev split (OTT-QA dev has about 2.2k questions, which is consistent but not confirmed).

---

## 2. Document chunk schema (`docs_chunks_1/*.json`)

1. **Count:** 21,742 files, all parseable, 21,742 unique `chunk_id`s. They come from 20,955 unique `source_id`s (20,955 are `chunk_0` and 787 are `chunk_1`). All have `chunk_type == "document"`.
2. **Schema** (every key is present in all 21,742 files):

```text
chunk_id               str    e.g. "ottaqa-<TableId>-<DocTitle>_chunk_<i>_<hash12>"
content                str    chunk text (14 to 4,486 chars, median 549)
source_info            dict
  source_id            str    "ottaqa-<TableId>-<DocTitle>"
  source_name          str    (same as source_id)
  source_type          str    "document"
  file_path            str    "OTT-QA/<source_id>.txt"
  structural_link      list[str]  always exactly 1 element: the parent table source_id
  original_source      str    "data/traindev_request_tok\\<TableId>.json"
  additional_information str  ""
  content              str    (duplicate of content)
sentences              list[str]
keywords               list[str]
summary                str
embedding              list[float]  dim 384 (MiniLM size)
merged_sentence_count  int
chunk_type             str    "document"
metadata               dict
  topic                str
  entities             dict[name -> {relationships: list[list], actions: list[str], details: list[str]}]
  events               dict[name -> {date: str, actions: list[str], details: str}]  (None in 3 files)
  timeline             list[str]
```

3. **Chunk ID field:** `chunk_id`. The hash suffix is VERIFIED to be `sha256(content)[:12]` (300/300 files sampled), matching `sage/src/pipeline/parallel_document_processor.py` L407–410. The filename is `<8 hex>_<16 hex>.json`, and its 16-hex part begins with the same 12-hex hash. **Consumers key on `data['chunk_id']`, not on the filename** (`graph_enhanced_accuracy_analyzer.py::_load_chunk_metadata` L381).
4. **Text field:** `content`. `summary`, `sentences` and `source_info.content` hold derived or duplicate text.
5. **Metadata fields:** `topic`, `entities`, `events`, `timeline`.
6. **Topic, entities and events are present.** `topic` is non-empty in 21,742 files, `entities` in 21,741 and `events` in 17,103 (4,639 are empty or None). Entity values have the sub-keys `relationships`, `actions` and `details`.
7. **Structural links are present.** `source_info.structural_link` has length 1 in all 21,742 files. All 21,742 targets resolve to a `source_id` in `table_chunks_with_metadata.json` (788 distinct parent tables), so every link is a document → parent-table link. **Note:** a link targets a table *source_id*, not a table *chunk_id*, so a link has to be resolved to the table's chunk(s); each table source has 1 or more chunks.
8. **Graph neighbors and edge information are not present.** The only key matching `neigh|edge|link|graph|communit|adj|similar` is `structural_link`. There are no similarity edges, no neighbor lists and no community IDs.

The schema is compatible with Variant B's reader (`_load_chunk_metadata` L370 reads `chunk_id`, `content`, `metadata.topic`, `metadata.entities`, `metadata.events`). All of those keys are present.

### Table chunk schema (`table_chunks_with_metadata.json`)
This is a list of 974 dicts with 974 unique `chunk_id`s (`ottaqa-<TableId>_chunk_0_<hash12>`), from 788 distinct `source_id`s. Every item has `chunk_index == 0`, so a table source with several chunks is distinguished by its hash, not by its index. Keys: `chunk_id, source_name, source_id, chunk_type("table"), content, rows_with_headers, row_count, column_headers, column_descriptions, keywords, summary, merged_row_count, content_length, chunk_index, metadata{table_title, table_description, table_summary, col_desc, col_format, entities}`.
- **There is no `embedding`** (Variant B re-embeds at query time, so it does not need one).
- **There is no `metadata.events`.** Variant B's `.get('events', {})` (L436) yields `{}`.
- Table entity values are `{type, category, description}`, a different shape from document entities.

---

## 3. Baseline CSV schemas

All six CSVs share `question` (str) and `question_id` (str, 16-hex, unique, no nulls). All except `enhanced_*` also have `table_id` (str, the OTT-QA table name) and `gold_docs` (str, a Python list literal of chunk IDs; parses in 2,212/2,212 rows; 1–28 per question, median 1). The rank columns `1st, 2nd, 3rd, 4th, …, 21th, 22th, …` hold chunk IDs. The suffixes after 3rd are always "th" (`21th`, `22th`, `23th`), exactly what `get_csv_retrieved_docs` (L489) generates. **No CSV contains a score**; rank is implied only by column order. All columns are read as `str` except where noted.

| CSV | One row = | Rank cols (chunk IDs) | Other list columns | Scores/ranks | Graph/community/expansion info |
|---|---|---|---|---|---|
| `dense_sparse_average_results.csv` | One question's hybrid ranking | `1st`…`100th`, all 100 filled, no duplicates within a row | `top{2,3,5}_results` (table IDs of those chunks), `top{2,3,5}_chunk_ids` (equal to the first N rank columns in 2,212/2,212 rows) | Rank order only | None |
| `bm25_only_results.csv` | One question's BM25 ranking | `1st`…`10th`, all filled | Same `top{2,3,5}_*` columns (consistent with rank columns 2,212/2,212) | Rank order only | None |
| `dense_sparse_sequential_results.csv` | One question's sequential dense→sparse ranking | `1st`…`10th`, all filled | Same `top{2,3,5}_*` columns (consistent) | Rank order only | None |
| `community_expanded_dense_sparse_results.csv` | One question's final ranking after community expansion | `1st`…`100th`; 1,490 rows have 100 filled, the rest have 18–99 (NaN padding) | `top{2,3,5,100}_results`, `top{2,3,5,100}_chunk_ids`; `top100_chunk_ids` holds `None` where rank cells are NaN | Rank order only | **Implicit only:** the output of an expansion, but no column says which items are seeds and which were added, or which community each item came from |
| `community_k_plus_k1_dense_sparse_results.csv` | Same as above ("k + k1" variant) | `1st`…`100th` (same padding pattern) | Same as above | Rank order, plus `retrieval_time_seconds` (float, mean 0.0146 s) | Implicit only (same caveat). Its rank columns equal `community_expanded`'s exactly in only 189/2,212 rows |
| `enhanced_results_final_communities_2049.csv` | One question's community-level retrieval evaluation | None | `top_{1,2,3,5,10}_communities` (pipe-joined strings such as `community_694\|community_2046`; 1,933 distinct community IDs) | 40 float metric columns: `{original_table_only, all_gold, any_gold, any_gold_plus_table}_{recall,precision}_{1,2,3,5,10}` | **Yes, community IDs per question.** There is no membership map from community to chunks. **UNKNOWN:** the metric semantics; `original_table_only_recall_1 + …_precision_1 == 1` in every row, so "precision" is likely not standard precision |

**How the files relate (VERIFIED):**
- `community_*` vs `dense_sparse_average`: the first rank equals the average CSV's first rank in 945/2,212 rows, the first 5 in 14 rows and the first 10 in 0 rows. On average only 22.3% of the community list's IDs appear in the average CSV's top 100. **The community lists are not "average seeds + appended neighbors" in any recoverable way.**
- `dense_sparse_sequential` top 10 equals `dense_sparse_average` top 10 (same order) in 11 rows (same set in 131 rows), so it is a different ranker.
- `gold_docs` is identical between `dense_sparse_average` and both `community_*` files (2,212/2,212), but differs from `bm25_only` and `dense_sparse_sequential` in 586 rows. The differences are the same chunk name with a different hash, for example `…HMS_King_Edward_VII_chunk_0_032a682d80b0` vs `…_chunk_0_15c4e15170eb`.

**Self-consistent ranking quality** (diagnostic only, each CSV scored against its own `gold_docs`; Recall = mean |gold ∩ top-k| / |gold|):

| k | dense_sparse_average R / hit | community_expanded R / hit | community_k_plus_k1 R / hit | bm25_only R | sequential R |
|---|---|---|---|---|---|
| 2 | 0.181 / 0.215 | 0.297 / 0.336 | 0.296 / 0.335 | 0.152 | 0.134 |
| 5 | 0.322 / 0.355 | 0.436 / 0.472 | 0.434 / 0.469 | 0.253 | 0.237 |
| 10 | 0.426 / 0.461 | 0.509 / 0.542 | 0.509 / 0.542 | 0.328 | 0.324 |
| 20 | 0.535 / 0.567 | 0.566 / 0.597 | 0.566 / 0.596 | – | – |
| 100 | 0.724 / 0.753 | 0.635 / 0.662 | 0.635 / 0.662 | – | – |

The community − average difference at k=20 is +3.1 Recall, not the paper's +5.7 (§5 and §6 explain the ID-version issue that likely contributes). **UNKNOWN:** whether either file corresponds to a specific number in the paper.

---

## 4. Chunk-ID cross-compatibility

The resolution target is the 21,742 doc chunk IDs plus the 974 table chunk IDs (22,716 total). Because the hash is a hash of the content, **"same prefix, different hash" means the CSV was produced from a different chunking run whose chunk text differs.**

| CSV | Rank cells resolved | Same name, different hash | `ottaqa-intro-*` (no file anywhere) | Absent entirely | Gold IDs resolved |
|---|---|---|---|---|---|
| `dense_sparse_average_results.csv` | 162,969 / 221,200 (**73.7%**) | 54,248 | 2,284 | 1,699 | 2,334 / 3,110 (**75.0%**) |
| `bm25_only_results.csv` | 17,214 / 22,120 (77.8%) | 4,303 | 496 | 107 | 3,002 / 3,091 (**97.1%**) |
| `dense_sparse_sequential_results.csv` | 17,129 / 22,120 (77.4%) | 4,323 | 585 | 83 | 3,002 / 3,091 (97.1%) |
| `community_expanded_…csv` | 200,144 / 200,144 non-null (**100%**) | 0 | 0 | 0 | 2,334 / 3,110 (75.0%) |
| `community_k_plus_k1_…csv` | 200,144 / 200,144 (100%) | 0 | 0 | 0 | 2,334 / 3,110 (75.0%) |

Questions whose gold set fully resolves to the provided chunks: 1,571/2,212 with the average CSV's gold, and 2,155/2,212 with the BM25/sequential gold. Questions whose gold is entirely document chunks: 1,114/2,212 (average gold).

**Findings:**
- **(VERIFIED) At least two chunk-ID versions exist.** `docs_chunks_1` and the table JSON are version V1. The community files' rank cells are fully in V1, and the BM25/sequential gold is about 97% V1. `dense_sparse_average_results.csv` (both rank cells and gold) is mostly in a different version V0. About a quarter of its IDs carry a hash that does not exist in V1. The community files mix versions: V1 rank IDs with V0 gold.
- **(VERIFIED) 794 distinct `ottaqa-intro-*` chunks (782 tables) are referenced in rankings, but no file contains them.** They appear to be per-table introduction passages. 21,742 + 974 = 22,716 provided nodes, against the paper's 22,886 nodes (Table 8), so this corpus is not exactly the graph the paper describes. **UNKNOWN:** the cause of the 170-node gap.
- **Is `dense_sparse_average_results.csv` the A5 artifact Variant B expects?**
  - **Schema: YES (VERIFIED).** It has the `question_id`, `question` and `gold_docs` columns Variant B reads (`analyze_question_accuracy` L1011–1013, `extract_gold_docs` L445), and `1st`…`100th` named exactly as `get_csv_retrieved_docs` L488–491 generates them. Extra columns (`table_id`, `top*_…`) are ignored.
  - **Filename: different.** Variant B defaults to `output/questions_dense_sparse_average_results1.csv` (L266). `ott_qa_retrieval_pipeline.py` L38 and `agent/ott_qa_batch_processor.py` L29 reference `output/dense_sparse_average_results (1).csv`. This is the same family of names; byte-level identity is **UNKNOWN** (the author would need to confirm it).
  - **Content: NOT compatible with the provided chunks (VERIFIED).** 26.3% of its seed IDs and 25.0% of its gold IDs do not exist in `docs_chunks_1` or the table JSON. In Variant B, a seed absent from the graph or metadata contributes no neighbors and no metadata, and gold under a V0 hash can only be matched by a V0 ranking. The file is therefore not internally consistent with the chunk/graph version shipped here.

---

## 5. Phase 1 blockers now resolved

| ID | Blocker | Status |
|---|---|---|
| A1 | `docs_chunks_1/*.json` | **RESOLVED** (21,742 chunks, full metadata, Variant B-compatible schema). Caveats: 794 intro chunks missing; ID version V1 only. |
| A2 | `table_chunks_with_metadata.json` | **RESOLVED** (974 tables). Caveats: no `events`, no embedding. |
| A5 | Seed ranking plus gold | **PARTIALLY RESOLVED.** The schema matches Variant B exactly, but about 25% of IDs are from another chunking version (V0), so seeds and gold do not fully resolve against A1/A2. |
| A7 | Structural links | **RESOLVED for doc → parent-table links**: they are embedded in A1 as `source_info.structural_link`, so `datasets/ott-qa-full-data.json` is no longer needed for that purpose. |
| D7 | Gold granularity | **RESOLVED: gold is chunk IDs** (`ottaqa-…_chunk_<i>_<hash>`), a mix of document and table chunks. |
| A8 | Rebuild-from-scratch inputs | Less urgent for A1/A2. Still needed if A3 must be rebuilt (CUDA). |

## 6. What is still missing for graph traversal

1. **A3, the similarity edge list (hard blocker).** No provided file contains chunk–chunk similarity edges. Variant B's `_load_graph_data` (L337) needs `unique_edges_*.json` with `source_chunk_id`/`target_chunk_id`. The only graph structure in the provided files is the 21,742 doc → parent-table structural links, a star graph around each table, which is not the SAGE semantic graph (751,836 edges per paper Table 8).
2. **Community membership.** `enhanced_results_*` names communities (`community_694`, …) but there is no community → chunk mapping, and no Leiden output (`leiden_communities_*.json`, written by `graph_analysis_pipeline.py::_detect_communities_with_leiden` L2682–2756). Communities cannot be reconstructed from the CSVs.
3. **A4, `indexing_filtered.csv` (hard blocker for unmodified Variant B).** `_load_chunk_metadata` L383 and L410 filter every chunk by `self.indexing_filtered_data['chunk_id']`, and `GraphAnalysisPipeline.load_chunks` reads it too. It is not provided. Whether it equals "all 22,716 provided chunk IDs" is **UNKNOWN**.
4. **A seed ranking in the same ID version as A1/A2 and A3.** The average CSV is mostly V0. We need either a V1 version of `dense_sparse_average_results.csv` or the V0 chunk set it was made from.
5. **The `ottaqa-intro-*` chunks** (794), which seeds and gold reference.
6. **A6, `questions_metadata_output.json`** (non-blocking; entity/event bonuses become 0 without it).
7. **Provenance:** the code or parameters that produced the five ranking CSVs (dense model, BM25 settings, the "average" vs "sequential" fusion, and what "community_expanded" vs "k_plus_k1" mean), and which file and metric back the paper's tables.

**Answers to the three capability questions**

| | Can the provided files…? | Answer |
|---|---|---|
| **A** | Reproduce the initial seed nodes | **Load: yes; reproduce: no.** The seeds can be *read* from `dense_sparse_average_results.csv`, but about 26% of them do not resolve to provided chunks, and the producing code is absent, so they cannot be *regenerated*. |
| **B** | Reproduce the published baseline retrieval outputs | **No.** The outputs themselves are provided (5 ranking CSVs), but nothing that produced them: no ranker code, no edges, no communities, no scores. They can be *re-evaluated* (Recall@k, §3) but not *reproduced*, and which file corresponds to which published number is UNKNOWN. |
| **C** | Reconstruct or directly load the SAGE graph | **No.** No edges or communities are provided. Only the node set (A1+A2) and doc → table structural links are available. Rebuilding the similarity edges with `GraphAnalysisPipeline` would require A4, CUDA (L1587), and the edge-rule decision D2, which is an unapproved rebuild. |

## 7. Recommended next step (PROPOSED, nothing done)

Send the SAGE author a short follow-up asking for:
1. `unique_edges_20250812_164429.json`, or whichever edge file matches the **V1** chunk IDs in `docs_chunks_1`.
2. `indexing_filtered.csv`.
3. A `dense_sparse_average_results.csv` regenerated on the V1 chunk IDs, or the V0 chunk set it was built from. Include the evidence: 54,248 rank cells and 657 gold IDs have the same chunk name with a different `sha256(content)[:12]` hash, for example `…HMS_King_Edward_VII_chunk_0_032a682d80b0` (CSV) vs `…_chunk_0_15c4e15170eb` (shipped).
4. The `ottaqa-intro-*` chunks.
5. The Leiden community membership file, if the community CSVs are part of the baseline.
6. `questions_metadata_output.json`.
7. Confirmation of which CSV and metric back the paper's OTT-QA Recall@k numbers, and whether Variant B (`GraphEnhancedAccuracyAnalyzer`) or the community pipeline is the reported SAGE method.

Until then, the provided files are enough to start **Baseline A (flat seeds) evaluation on the provided rankings**, but not anything that needs a graph. Choosing to restrict evaluation to the 1,571 questions whose gold fully resolves, or to rebuild edges ourselves, are decisions for you (and possibly the authors). I have not made either one.
