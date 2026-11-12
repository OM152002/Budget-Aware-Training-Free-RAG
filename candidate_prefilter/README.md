# Candidate pre-filter: SAGE's neighbor ranker as top-m

The pre-filter is SAGE's own `GraphEnhancedAccuracyAnalyzer._calculate_hybrid_similarities` (`sage/scripts/analysis/graph_enhanced_accuracy_analyzer.py`, L515–562), imported unmodified. The score is 0.40 × [cos(q, content) + cos(q, topic)] + 0.60 × BM25 fitted on the candidate set. Question entity/event bonuses are empty, because the question-metadata artifact (A6) is unavailable. Final order is a stable sort on (−score, chunk_id). Scoring is cache-free.

| File | Role |
|---|---|
| `sage_prefilter.py` | `SagePrefilter(adapter).score(question, candidate_ids)` returns the full ranking; `.top_m(question, ids, m)` returns the top m |
| `verify_prefilter.py` | Cache vs cache-free check, and agreement with SAGE's end-to-end `get_graph_neighbors_for_retrieved_chunks` |
| `evaluate_prefilter.py` | k ∈ {1,3,5}, m ∈ {5,10,20} (plus m=50 as a diagnostic), on the 1,728 document-gold questions |

```bash
cd Project/candidate_prefilter
../.venv-sage/bin/python -P -s verify_prefilter.py   runs/<new_dir>   2> verify.stderr
../.venv-sage/bin/python -P -s evaluate_prefilter.py runs/<new_dir>   2> eval.stderr   # ~70 min, cache-free
```

**Results:**
- `runs/verify_n60/verification.json`: the embedding cache **failed** verification (10 of 180 full rankings changed order; score differences up to 9.5e-8), so the cache is disabled. Agreement with SAGE's own selector: 349 of 360 identical sets, with the remaining 11 differing only by exact ties at the cut-off.
- `runs/eval_cachefree_run{A,B}/`: `evaluation.json` and `rankings.jsonl`. The two runs have byte-identical rankings, and every metric is identical except timing.

SAGE's `encode(show_progress_bar=True)` writes progress bars to stderr, so redirect it.
