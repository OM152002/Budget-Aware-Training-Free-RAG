# Budget-Aware Training-Free RAG

CSE 576 Project C: a **training-free, budget-aware retriever**. A pretrained small language model (Qwen3-4B-Instruct-2507, 4-bit, MLX) acts as an `EXPLORE(node)` / `STOP` controller for multi-hop traversal over a **SAGE** chunk graph built on OTT-QA document chunks. It works under an explicit budget B, the maximum number of additional graph nodes. The SLM is not fine-tuned.

**Status:** all experiments, statistics, tables and figures are complete and frozen. The interpretation (RQ1–RQ3) is frozen. The final report is not written yet.

## Start here

| Document | What it covers |
|---|---|
| [`Project_C_Updated_Phase_Wise_PRD.md`](Project_C_Updated_Phase_Wise_PRD.md) | Scope, research questions RQ1–RQ3 and working rules |
| [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) | Frozen system, methods (M0–M5, A1–A3), question sets, budgets, experiment matrix |
| [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) | Metrics, statistics and the approved analysis decisions (D1–D8) |
| [`RESULTS_INTERPRETATION.md`](RESULTS_INTERPRETATION.md) | Findings for RQ1–RQ3, with every number traceable to the frozen outputs |
| [`analysis/tables/TABLES.md`](analysis/tables/TABLES.md) | Tables T1–T6 |
| [`analysis/figures/`](analysis/figures/) | Figures F1–F5 (PNG and PDF, plus the plotted data as CSV) |

Background and history:
- [`PHASE1_BLOCKERS.md`](PHASE1_BLOCKERS.md) and [`PHASE1_ARTIFACT_AUDIT.md`](PHASE1_ARTIFACT_AUDIT.md): the SAGE repository and artifact audit.
- [`GRAPH_RECONSTRUCTION_PLAN.md`](GRAPH_RECONSTRUCTION_PLAN.md): how the graph was rebuilt.
- [`SEED_RETRIEVAL_PLAN.md`](SEED_RETRIEVAL_PLAN.md): seed retrieval.

## Pipeline and repository layout

```text
query → seed retrieval (Fusion-A, top-3) → graph neighbours → SAGE pre-filter (top-20)
      → SLM controller: EXPLORE one candidate or STOP → repeat until STOP / budget / no candidates
```

| Directory | Contents |
|---|---|
| `graph_reconstruction/` | CPU runner for SAGE's released graph pipeline (documents only); validation and verification reports |
| `seed_retrieval/` | Fusion-A seed retrieval (a transcription of SAGE's `hybrid_search`) and its evaluation |
| `traversal_adapter/` | `GraphSeedAdapter`: node, neighbour and seed lookup over the reconstructed graph |
| `candidate_prefilter/` | SAGE `_calculate_hybrid_similarities` pre-filter (cache-free) and its evaluation |
| `slm_controller/` | Controller prompt (v2), output parser, MLX backend |
| `adaptive_traversal/` | Traversal loop (budget, visited set, duplicate-text handling); development and held-out pilots |
| `baselines/` | Non-SLM traversal baselines (SAGE-score greedy, random) |
| `experiments/` | Frozen experiment registry, resumable runner, ablations, question sets, `FROZEN.json` (hashes), run manifests |
| `analysis/` | Integrity checks → scoring → statistics → tables → figures → T7 appendix → interpretation check |
| `phase1_repro/artifact_audit/` | Scripts behind the Phase 1 artifact audit |

Every analysis step reads only the frozen outputs of the step before it and verifies their hashes against its manifest.

## Main result (B = 4, TEST-1704, ID Recall; full numbers in T1)

| Method | ID Recall |
|---|---|
| M0 seeds only | 0.2409 |
| M1 flat retrieval | 0.3341 |
| M2 SAGE one-shot expansion | 0.3109 |
| M3 random traversal | 0.2706 |
| M4 SAGE-score greedy traversal | 0.3207 |
| **M5 SLM controller (this work)** | **0.3609** |

See [`RESULTS_INTERPRETATION.md`](RESULTS_INTERPRETATION.md) for the paired statistics, the budget sweep, the ablations and the limitations.

## What is not in this repository

These items are excluded for size. The key excluded artifacts (graph edges, seed rankings, experiment outputs and per-question scores) have SHA-256 hashes recorded in `experiments/FROZEN.json` or the analysis manifests:
- **Upstream SAGE code:** <https://github.com/CoRAL-ASU/sage> at commit `405d5f5`, cloned to `sage/` and unmodified.
- **SAGE artifacts:** Hugging Face dataset `prasham1515/SAGE`, expected in `sage_hf_artifacts/`.
- **Large generated data:**
  - the reconstructed graph edges (275 MB) and caches;
  - seed rankings;
  - pre-filter rankings;
  - per-question experiment outputs (`experiments/runs/*/results.jsonl`) and per-question scores.

  Available on request.
- **Logs and smoke-test outputs.**

Because the per-question files are excluded, the analysis scripts verify their inputs and will stop on a fresh clone. The committed tables, figures, statistics and aggregate scores are the frozen results.

## Environment (frozen; see `experiments/FROZEN.json`)

| Component | Version |
|---|---|
| Python | 3.13.1 |
| mlx | 0.32.4 |
| mlx-lm | 0.32.2 |
| torch | 2.14.1 |
| sentence-transformers | 6.1.0 |
| transformers | 5.19.0 |
| numpy | 2.2.6 |
| pandas | 3.0.6 |

- **Model:** `mlx-community/Qwen3-4B-Instruct-2507-4bit` at revision `50d4277`, greedy decoding.
- **Hardware:** everything ran on an Apple M3 MacBook Air (16 GB), CPU plus Metal.
- **Running the scripts:**
  - Use `python -P -s` with `PYTHONHASHSEED=0`.
  - Set `OMP_NUM_THREADS=1` when faiss and torch share a process.
  - Exact experiment commands are in [`experiments/README.md`](experiments/README.md).
