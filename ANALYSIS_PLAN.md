# Analysis Plan: Project C (frozen v2 system)

**Status: PLAN ONLY.** No gold labels have been read, no recall computed, no statistics run, no figures made. All 11 runs are complete, verified and frozen (`experiments/FROZEN.json`). Nothing will be modified or rerun.

Labels: **VERIFIED** = checked in the files; **PROPOSED** = analysis choice in this plan; **DECISION** = needs approval (§7).

## 0. Fixed decisions (carried over)
- **Primary metric: ID-based Recall** of the final evidence set (seeds ∪ expansions), `|G ∩ R| / |G|`, with G = the question's document gold (`doc_gold`), macro-averaged over questions. Same definition as every earlier run (`baselines/run_baselines_heldout.py::evaluate`, VERIFIED).
- **Text-equivalent Recall** (gold g credited if its normalized text equals that of any retrieved node): **diagnostic only**, never used for a conclusion.
- **DEV-24 is excluded** from all primary numbers (appendix T7 only).
- **TEST-1704** (main B=4 comparison) and **NEW-1654** (TEST-1704 minus PILOT-50, never evaluated before) are reported **separately** where planned.
- **SUB-300** is used for the budget sweep and ablations.
- **No run is modified or rerun.** Every number below comes from logged records, by direct reading or exact prefix truncation.

## 1. Frozen data sources (VERIFIED)

| Analysis label | Source run(s) | Set | Budgets available |
|---|---|---|---|
| M0 seeds, M1 flat top-(3+b), M2 SAGE one-shot (faithful + dedup diagnostic), M4 SAGE-score greedy | `E2_baselines_B8_test1704` | TEST-1704 | 1, 2, 4, 8 (prefix truncation) |
| M3 random ×3 seeds | `E2r_random_B4_test1704` | TEST-1704 | 1, 2, 4 (truncation) |
| M3 random ×3 seeds | `E2r8_random_B8_sub300` | SUB-300 | 8 |
| **M5 SLM, B=4** | `E1_slm_B4_new1654` + `adaptive_traversal/runs/heldout50_k3_m20_B4_v2` (PILOT-50) | TEST-1704 = NEW-1654 ∪ PILOT-50 | 4 |
| M5 SLM, B=1/2/8 | `E3_B1/B2/B8_slm_sub300` | SUB-300 | 1, 2, 8 |
| M5 SLM, B=4 on SUB-300 | E1 (291 q) + PILOT-50 (9 q) | SUB-300 | 4 |
| A3 no adaptive stop | `E4a_B4_nostop_test1704` / `E4a_B8_nostop_sub300` | TEST-1704 / SUB-300 | 4 / 8 |
| A1 no history | `E4b_nohistory_B4_sub300` | SUB-300 | 4 |
| A2 no budget input | `E4c_nobudget_B8_sub300` | SUB-300 | 8; 1, 2, 4 by truncation |

- **PILOT-50 records:** produced by `run_heldout.py`. Its `summary.json` records code hashes for the 5 method files, and all 5 match FROZEN.json (VERIFIED today).
- **Fields to ignore:** PILOT-50 records carry an old gold-derived `eval` field. The analysis **drops it and re-scores** with the common evaluator. No E4a record carries it (VERIFIED, 0/1704).
- **E0 is not used:** it was the reproducibility check only.

**Truncation rules (exact, from EXPERIMENT_PLAN §4):**

| Method | Smaller budget b < logged B |
|---|---|
| M1 | first 3+b of the seed ranking |
| M2 | first b of the one-shot ranking |
| M3, M4 | first b expansions |
| A2 | first b expansions, or the whole trajectory if it stopped earlier |
| M5, A1 | never truncated (the budget or history text changes every prompt) |

The same rules give the matched-expansion baselines. For each question, M1–M4 get exactly the number of expansions M5 used on it (at most B).

## 2. Shared statistics (PROPOSED unless noted)
- **Paired bootstrap 95% CI** for every recall difference vs M5 (or vs the ablation's reference): 10,000 resamples of questions, percentile interval, one fixed seed (D7).
- **Exact two-sided sign test** on per-question wins and losses, ties dropped (EXPERIMENT_PLAN §5).
- Each absolute recall is also given a bootstrap CI.
- **M3:** headline = mean ± SD across the 3 random seeds, with each seed listed. Paired tests use the per-question mean over the 3 seeds (D2).
- **Resolution (EXPERIMENT_PLAN §0):** the paired SE is about 0.009 at n=1,704 and about 0.021 at n=300. SUB-300 differences under about 0.04 are reported as inconclusive, not as "no effect".
- A multiple-comparison policy is still open (D3).

## 3. RQ1: training-free traversal effectiveness
*Can the SLM pick useful neighbors without task-specific training?*

| Item | Plan |
|---|---|
| **Compared** | M5 vs M4 (SAGE-score greedy), M3 (random), M2 (original SAGE one-shot, faithful), M1 (flat), M0 (seeds) |
| **Set / budget** | **TEST-1704, B=4** (primary per EXPERIMENT_PLAN); same table on **NEW-1654** separately (D1) |
| **Primary metric** | ID Recall; paired Δ vs M5 with 95% bootstrap CI |
| **Secondary** | text-equivalent Recall (diagnostic); hit rate (≥1 gold); questions gaining gold over seeds; additional gold found; expansions used; per-question win/tie/loss vs M5 |
| **Two comparisons** | (a) **max-budget**: baselines use all 4 expansions; (b) **matched-expansion**: baselines get M5's per-question expansion count |
| **Statistics** | paired bootstrap CI on each Δ; exact sign test on wins and losses |
| **Behavior diagnostics** | distribution of the SLM's selected pre-filter rank by step (M4 is always rank 1); M2 dedup vs faithful (diagnostic) |
| **Answers** | **T1** (max-budget), **T2** (matched-expansion + win/tie/loss + sign test), **F4** (selected-rank histogram), **F5** (paired outcome M5 vs M4 by category) |

## 4. RQ2: adaptive multi-hop retrieval and stopping
*Can the SLM tell when more exploration is needed and when it has enough?*

| Item | Plan |
|---|---|
| **Compared** | (1) M5 vs **A3** (no adaptive stop) at B=4 on **TEST-1704** and B=8 on **SUB-300**; (2) M5 vs **A1** (no history) at B=4 on **SUB-300**; (3) multi-hop behavior of M5 vs M4 at B=4 on TEST-1704 |
| **Primary metric** | ID Recall; paired Δ (M5 − A3, M5 − A1) with 95% CI |
| **Secondary: stopping** | STOP rate; expansions saved by STOP (A3 − M5); recall lost or gained by stopping; **STOP-when-sufficient** (all gold already in evidence at STOP, post-hoc diagnostic); STOP at step 0; **wasted expansions** (taken after the last gold was found); A3 suppressed-STOP count and extra re-prompt calls |
| **Secondary: multi-hop** | hop depth of each discovered gold (1 vs 2+, D4); recall by post-hoc category (all gold in seeds / reachable in top-20 / reachable but filtered / unreachable in 1 hop, D5) |
| **Statistics** | paired bootstrap CI and sign test for M5 vs A3 and M5 vs A1. Restricted to questions where A3 differs from M5 (the M5-stopped questions), as a descriptive breakdown, not a separate test |
| **Answers** | **T4** (A1, A3 rows), **T5** (stopping + multi-hop), **F3** (gold by hop depth and category, M4 vs M5) |

## 5. RQ3: budget-aware retrieval
*Does the SLM keep effectiveness while cutting unnecessary exploration and cost under a fixed budget?*

| Item | Plan |
|---|---|
| **Compared** | M5 at B ∈ {1, 2, 4, 8} vs M1–M4 at the same B; M5 vs **A2** (no budget input) at B ∈ {1, 2, 4, 8} |
| **Sets** | M5 at B≠4 and all A2 rows on **SUB-300**, compared with baselines restricted to SUB-300. Baselines at B ∈ {1, 2, 4} also on **TEST-1704** (M3 at B=8 is SUB-300 only). M5 at B=4 on both |
| **Primary metric** | ID Recall vs B; paired Δ (M5 − baseline) at each B; paired Δ (M5 − A2) at each B |
| **Secondary: cost** | expansions used and expansions/B; SLM calls (A3: including re-prompts, D6); input/output tokens; SLM, pre-filter and total latency per question; recall gain over seeds per 1k tokens and per second (SLM methods only) |
| **Secondary: behavior** | STOP rate by B (M5 vs A2); STOP timing (step 0 vs later) |
| **Statistics** | paired bootstrap CI per (method, B) difference; sign tests as in RQ1; the SUB-300 resolution caveat applies |
| **Answers** | **T3** (budget sweep, recall + cost), **T4** (A2 rows), **T6** (controller reliability by B: invalid first attempts, retries, CONTROLLER_ERROR), **F1** (recall vs B), **F2** (recall vs cost, D5) |

## 6. Gold-free checks before any scoring (PROPOSED)
Run first. Any failure stops the analysis and is reported.
1. **Fingerprints:** every manifest's fingerprints equal FROZEN.json, and `--verify-frozen` passes.
2. **Question order:** each run's question IDs equal its set in its frozen order. NEW-1654 ∪ PILOT-50 = TEST-1704.
3. **Prefix property:**
   - M4 and M3 truncations are prefixes of the logged runs.
   - M5's step-1 `top_m` equals M4's step-1 `top_m` on every TEST-1704 question (same seeds and pre-filter).
   - The E2r8 first 4 picks equal E2r's picks (already VERIFIED, 300/300).
4. **A2 truncation:** with `include_budget=False`, `build_prompt` produces identical text for different remaining budgets. This is a code-level check with no SLM call.
5. **Evaluation code:** the evaluator (`evaluate()` imported unchanged from `baselines/run_baselines_heldout.py`) and the analysis scripts are hashed and recorded before gold is read.

Outputs go to `analysis/`: per-question CSV, table CSV + Markdown, figures PNG + PDF, and a manifest with the code hashes.

## 7. Decisions needing your approval

| # | Ambiguity | Recommendation |
|---|---|---|
| **D1** | **Headline set for RQ1.** PILOT-50's results (and the baselines on it) were seen before freezing, without tuning. NEW-1654 has never been scored. | Keep **TEST-1704 as primary** (as planned). Give the **NEW-1654** column in T1/T2 equal prominence. Note in the text if any conclusion differs between them. |
| **D2** | **M3 in paired tests.** Average the 3 random seeds per question, or test each seed? | Use the **per-question mean over seeds** for Δ, CI and sign test. List the per-seed recall too. |
| **D3** | **Multiple comparisons.** The plan has no correction; T1/T2 make 5 comparisons, T4 makes 3+. | **Holm correction on sign-test p-values within each table family** (T1/T2, T3 per B, T4). Leave CIs unadjusted and say so. Alternative: no correction, CIs only. |
| **D4** | **"Hop depth" of a discovered gold.** | **Shortest graph distance from the seed set** in the frozen graph: 1 = neighbor of a seed, 2+ = otherwise. It's a gold-free graph property, applied post hoc. |
| **D5** | **Post-hoc categories and the cost axis for F2.** Per-step latency isn't logged for M3/M4 (only per-run totals), so latency at a truncated budget isn't exact. Baselines also use 0 SLM tokens. | **Categories:** reuse the held-out definitions exactly (step-1 raw neighbors + step-1 top-20 including duplicate members), computed from E2's logged step-1 ranking. **F2:** two panels. (a) x = expansions used, exact for every method and B. (b) x = wall latency, only where measured exactly: M5 all B, M4 B=8, M3 B=4 (TEST) and B=8 (SUB), M2 faithful. No prorating. |
| **D6** | **A3 cost accounting.** | Report the **actual SLM calls including re-prompts** (`slm_calls_actual`) as the cost, with the recorded calls shown alongside. |
| **D7** | **Bootstrap seed.** | One seed, **20261102**, for all resamples. Fixed now, before any scoring. |
| **D8** | **Downstream QA (PRD §26).** It was never in EXPERIMENT_PLAN and has no runs. | **Exclude**; the analysis is retrieval-only. |

**APPROVED 2026-11-01: D1–D8 as recommended**, with two clarifications:
- **D1:** TEST-1704 is the primary RQ1 population. NEW-1654 is reported separately and prominently as the robustness set that had never been scored before.
- **D4:** seed evidence = **hop 0**; direct graph discovery (neighbor of a seed) = **hop 1**; genuine multi-hop discovery = **hop 2+**.

## 8. Execution order (after approval)
1. Run the gold-free checks (§6).
2. Hash the evaluation code.
3. Compute the per-question scores once for all methods and budgets.
4. T1 → T2 → T3 → T4 → T5 → T6, then F1–F5, then the T7 appendix.

The analysis stops for review after the tables, before writing any interpretation.
