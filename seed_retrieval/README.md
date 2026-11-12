# SAGE seed regeneration (Fusion A / `hybrid_search`), 21,742-document corpus

The plan and decisions are in `../SEED_RETRIEVAL_PLAN.md`; the approved choices were S1–S7. The SAGE repo is unmodified.

| File | Role |
|---|---|
| `seed_common.py` | Index build with SAGE's own `ChunkLoader` / `HNSWIndex` / `BM25Index` (documents only), plus a line-for-line standalone `hybrid_search` |
| `validate_seed.py` | Compares the standalone function with SAGE's **original** `agent/ott_qa_api_server.py::hybrid_search`, run through a validation-only Flask stub in `sys.modules` (nothing installed) |
| `run_seeds.py` | Top-100 rankings for all 2,212 questions, with scores and config |
| `evaluate_seeds.py` | Determinism (run A vs B), Recall/Hit@k over document gold, and the diagnostic overlap with the authors' ranking |

```bash
cd Project/seed_retrieval
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
# use -P -s, NOT -I: -I implies -E, which ignores PYTHONHASHSEED (tie order would become random)
../.venv-sage/bin/python -P -s validate_seed.py runs/<new_dir>
../.venv-sage/bin/python -P -s run_seeds.py     runs/<new_dir_A>
../.venv-sage/bin/python -P -s run_seeds.py     runs/<new_dir_B>
../.venv-sage/bin/python -P -s evaluate_seeds.py runs/<new_dir_A> runs/<new_dir_B>
```

**Canonical output:** `runs/seeds_fusionA_top100_seed0_runA/`
- `seeds_top100.csv`: `question_id, question, gold_docs` (BM25-file gold), `doc_gold`, `n_gold`, `n_doc_gold`, `eval_group`, `n_returned`, then `1st … 100th`. It can be read by Variant B's `get_csv_retrieved_docs`; the number of seeds k is chosen by the consumer.
- `seeds_top100_scores.jsonl`: per question, the rank, `chunk_id`, `combined_score`, `hnsw_score` and `bm25_score`.
- `config.json` and `evaluation.json`.

**Notes:**
- Tie order: SAGE sorts a Python `set` stably, so exact score ties are ordered by string-hash iteration order. `PYTHONHASHSEED=0` fixes that order deterministically without changing any code. With hash randomization, primary Recall@k moved by at most 0.0017.
- Import order matters on this Mac: `faiss` → `torch` → `sentence_transformers`, before any FAISS work. Otherwise both bundled libomp copies initialize and the process aborts.
- `runs/_superseded_hash_randomized/` holds the first, hash-randomized runs, kept as evidence of the tie-order effect.
