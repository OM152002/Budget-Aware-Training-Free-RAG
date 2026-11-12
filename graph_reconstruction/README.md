# SAGE graph reconstruction: documents only, CPU

This rebuilds SAGE's document–document graph from the author-provided `docs_chunks_1` (21,742 chunks) with the **released** `sage/scripts/analysis/graph_analysis_pipeline.py`, imported unmodified. The plan and decisions are in `../GRAPH_RECONSTRUCTION_PLAN.md`; the user's choices were G1 documents only, G2 filter = all documents, G3 skip reports, G4 skip Leiden, G5 CPU-only validation.

| File | Role |
|---|---|
| `sage_cpu_runner.py` | Run-dir setup, the two CUDA-removal patches (P1 embedding timer, P2 CUDA guard), logging |
| `reference_features.py` | Independent reimplementation of the doc–doc features (validation only) |
| `validate_port.py` | Small pre-build validation (V1, V3, V4, V6, V7) |
| `build_graph.py` | Full build: load → HNSW → k=200 pairs → features → 99th-pct AND + entity rule → `unique_edges_*.json` |
| `verify_graph.py` | Post-build independent check of thresholds and edge set from the saved batch CSVs (V5, V8, V9) |

```bash
cd Project/graph_reconstruction
export OMP_NUM_THREADS=1        # REQUIRED: faiss-cpu and torch ship separate libomp copies; multithreaded OpenMP crashes natively
../.venv-sage/bin/python -I validate_port.py runs/<new_validation_dir>
../.venv-sage/bin/python -I build_graph.py   runs/<new_build_dir>
../.venv-sage/bin/python -I verify_graph.py  runs/<new_build_dir>
```

Each run directory must be new (stale SAGE caches would otherwise short-circuit the computation). It contains `output/indexing_filtered.csv` (all 21,742 doc IDs), symlinks to the artifacts, SAGE's own `batch_results/` and caches, the edges file at `output/analysis_cache/knowledge_graph/unique_edges_<timestamp>.json`, `sage_pipeline.log`, `build_summary.json` and `verification_report.json`.

The table file is symlinked at its expected path, but SAGE's own indexing filter (L1306) drops every table because none is listed in `indexing_filtered.csv`. No structural links are used.
