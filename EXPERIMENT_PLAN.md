# Final Experimental Plan: Project C (frozen v2 system)

**Status: PLAN ONLY.** Nothing new has been run. All methods and settings are frozen: graph `docgraph_k200_q99_20261023`, Fusion-A seeds (top-3), SAGE pre-filter (cache-free, top-20), v2 controller and prompt, text dedup, Qwen3-4B-Instruct-2507 4-bit MLX @ `50d4277`, greedy decoding, 1 retry. The frozen code hashes are recorded in `adaptive_traversal/runs/heldout50_k3_m20_B4_v2/summary.json`.

## 0. Evaluation sets
| Set | Definition | Use |
|---|---|---|
| **DEV-24** | the 24 questions used in both pilots (the v2 fixes were made while looking at them) | excluded from all final numbers; reported once as a development set |
| **TEST-1704** | all 1,728 document-gold questions minus DEV-24 | every non-SLM method at every budget; the main SLM configuration (B=4) |
| **SUB-300** | fixed random 300 from TEST-1704 (`random.Random(20261101)`, sampled before any result is seen) | expensive SLM budget sweeps and ablations |
| HELDOUT-50 | already run, ⊂ TEST-1704 | reused: its B=4 SLM results are reused rather than recomputed |

Primary metric: **ID-based document-gold Recall** of the final evidence set (seeds + expansions), the same metric as before.

Statistical resolution. Measured on HELDOUT-50, the per-question SLM − SAGE-score recall difference has SD 0.365. That gives a paired standard error of about **0.009 at n=1,704** and about **0.021 at n=300**. So TEST-1704 can resolve the SLM-vs-baseline gap seen so far (0.025), while SUB-300 ablations can only detect effects of roughly 0.04 or more. This is the reason for the full-vs-subset split.

## 1. Methods (all share graph, seeds, candidate generation, dedup and pre-filter unless noted)
| ID | Method | Decision rule | Qwen? |
|---|---|---|---|
| M0 | Seeds-only | top-3 Fusion-A seeds, no expansion | no |
| M1 | Flat equal-node (PRD Baseline A) | Fusion-A top-(3+b) from the existing seed ranking; graph not used | no |
| M2 | Original SAGE one-shot (PRD Baseline C) | seeds ∪ top-b neighbors of S by the SAGE hybrid score, one hop, not iterative. Faithful version: no text dedup (Variant B behavior); the dedup version is a diagnostic | no |
| M3 | Random traversal (PRD Baseline B) | uniform over eligible top-20 each step; **3 fixed seeds {0,1,2}, report mean ± SD and each seed** | no |
| M4 | SAGE-score greedy traversal (PRD Baseline B) | top-1 eligible candidate each step | no |
| M5 | **Frozen SLM traversal (full)** | Qwen: EXPLORE/STOP with (q, E, N, H, B) | yes |
| A1 | SLM, no history | M5 with `include_history=False` | yes |
| A2 | SLM, no budget input | M5 with `include_budget=False` | yes |
| A3 | SLM, no adaptive stopping | M5, except that when the controller outputs STOP, one fixed re-prompt removes STOP ("STOP is not available; choose one of A1..Am") and traversal continues until B is exhausted | yes (only at STOP points) |

"Matched-expansion" runs give M1–M4 exactly the number of expansions M5 used on each question. They cost nothing extra (see §4).

## 2. Experiment matrix
| Exp | RQ | Methods | Set | Budgets B |
|---|---|---|---|---|
| **E1 main** | RQ1, RQ3 | M5 | TEST-1704 (1,654 new + HELDOUT-50 reused) | 4 |
| E2 baselines | RQ1, RQ3 | M0, M1, M2, M4 | TEST-1704 | 1, 2, 4, 8 |
| E2r random | RQ1, RQ3 | M3 × 3 seeds | TEST-1704 for B ≤ 4; SUB-300 for B=8 | 1, 2, 4 (+8 on SUB-300) |
| E3 budget sweep | RQ3 | M5 | SUB-300 | 1, 2, 8 (B=4 reused from E1) |
| E4a stopping | RQ2 | A3 | TEST-1704 at B=4; SUB-300 at B=8 | 4, 8 |
| E4b history | RQ2 | A1 | SUB-300 | 4 |
| E4c budget input | RQ3 | A2 | SUB-300 | 8 (B=1, 2, 4 derived by truncation; §4) |
| E0 reproducibility | – | M5 | 10 HELDOUT-50 questions | 4 (must reproduce byte-identically before E1 starts) |

**RQ mapping:**
- **RQ1 (training-free effectiveness):** M5 vs M4, M3, M2, M1 and M0 at B=4. Report both max-budget and matched-expansion, with paired 95% bootstrap CIs on TEST-1704. Also report where the SLM's choices differ from the SAGE score (distribution of the selected pre-filter rank).
- **RQ2 (multi-hop and stopping):**
  - M5 vs A3 (does STOP save expansions without losing recall?);
  - M5 vs A1 (does history matter?);
  - hop depth of discovered gold (1 hop vs 2+ hops from the seeds);
  - outcomes by post-hoc category (all gold in seeds, reachable in top 20, reachable but filtered, unreachable in one hop).
- **RQ3 (budget awareness):** recall-vs-budget and recall-vs-cost curves for M5 (B ∈ {1,2,4,8}) against M1–M4 at the same B; M5 vs A2 (does seeing the budget change behavior?); expansions used / B; tokens and latency per unit of recall gained.

## 3. Proposed budgets
B ∈ **{1, 2, 4, 8}** (doubling). B=4 is the main point, already validated. B=1 is a single SLM choice, B=2 a short chain, and B=8 tests whether gains continue at larger depth. These values are fixed now, before any results, and will not be tuned. m=20 and k=3 stay fixed.

## 4. Reuse and exact cost-saving derivations (no approximation)
- **M0 and M1** come straight from `seed_retrieval/runs/seeds_fusionA_top100_seed0_runA/seeds_top100.csv` (top-11 is enough for B=8). Zero compute.
- **M3 and M4 are prefix-nested.** They're deterministic, never stop early, and ignore B, so one run at B_max gives every smaller B by truncation (verified: matched-expansion runs were exact prefixes of the B=4 runs, 50/50). The matched-expansion comparisons also come from truncation, at no cost.
- **M2 dedup diagnostic:** take the top-b of M4's logged step-1 ranking. Free. The faithful no-dedup M2 needs one pre-filter call per question.
- **A2 is prefix-nested.** With no budget in the prompt, the B=8 trajectory truncated at b equals a B=b run, so one B=8 run covers B ∈ {1,2,4,8}.
- **A3 shares every trajectory with M5 up to the first STOP.** Only questions where M5 stopped need new calls (about 6% of questions, measured).
- **M5 at B=4 on HELDOUT-50** is reused (frozen code, matching hashes). E0 re-checks determinism on 10 of them.
- Graph, adapter caches, seeds and pre-filter code are reused unchanged. The DEV-24 pilot results are kept for the development-set appendix.
- **Not reusable:** M5 at B ≠ 4, A1, and M5 at B=4 on new questions. The budget or history text changes the prompt at every step.

## 5. Metrics
- **Effectiveness:**
  - ID recall (primary);
  - text-equivalent recall (diagnostic);
  - hit rate (≥1 gold);
  - number of questions gaining gold over seeds;
  - additional gold found;
  - paired per-question wins and losses against M5.
- **Efficiency:**
  - expansions used and expansions / B;
  - SLM calls, input and output tokens;
  - SLM latency, pre-filter latency, total latency per question;
  - recall gain per 1k tokens and per second.
- **Controller behavior:**
  - STOP rate;
  - STOP when all gold is already in the evidence (post-hoc diagnostic only);
  - expansions after the last gold is found ("wasted");
  - selected pre-filter rank distribution;
  - invalid first attempts, retries, CONTROLLER_ERROR rate.
- **Analysis:**
  - hop depth of each discovered gold;
  - recall by post-hoc category.
- **Statistics:**
  - paired bootstrap 95% CIs (10,000 resamples, fixed seed) for every recall difference against M5;
  - an exact sign test on per-question wins and losses.

## 6. Compute estimate (from HELDOUT-50 measurements)
**Measured per SLM step** (calls / input tokens / latency):

| Step | Input tokens | Latency | Pre-filter time |
|---|---|---|---|
| 1 | 1,847 | 7.7 s | 0.77 s |
| 2 | 1,973 | 8.2 s | 0.96 s |
| 3 | 2,108 | 8.9 s | 1.15 s |
| 4 | 2,237 | 9.3 s | 1.42 s |

Output is about 14 tokens per call. Steps 5–8 are linearly extrapolated (+130 tokens, +0.54 s and +0.21 s pre-filter per step).

**Per-question estimates:**

| B | SLM calls | Input tokens | SLM time per question | Pre-filter-only baseline per question |
|---|---|---|---|---|
| 1 | ~1.0 | ~1.9k | ~8.5 s | ~0.8 s |
| 2 | ~1.95 | ~3.8k | ~17.6 s | ~1.7 s |
| 4 | 3.88 (measured) | 7.9k (measured) | 37.2 s (measured) | 3.5–3.7 s (measured) |
| 8 | ~7.7 | ~18k | ~86 s | ~12 s |

**Qwen experiments:**

| Experiment | Questions | SLM calls | Input tokens | Wall time |
|---|---|---|---|---|
| E0 reproducibility | 10 | ~40 | 0.08M | 0.1 h |
| E1 main, B=4 | 1,654 | ~6,400 | 13.1M | **17.1 h** |
| E4a no-stop, B=4 (STOP points only) | ~100 | ~300 | 0.6M | 0.8 h |
| E3 B=1 / B=2 / B=8 | 300 each | ~300 / ~590 / ~2,300 | 0.55M / 1.1M / 5.4M | 0.7 h / 1.5 h / 7.2 h |
| E4a no-stop, B=8 (STOP points only) | ~20 | ~100 | 0.2M | 0.3 h |
| E4b no history, B=4 | 300 | ~1,160 | 2.3M | 3.1 h |
| E4c no budget, B=8 | 300 | ~2,300 | 5.4M | 7.2 h |
| **Qwen total** | | **~13,500** | **~28.8M** (+~0.19M output) | **~38 h** |

**Non-SLM experiments (CPU, pre-filter-bound):**

| Experiment | Wall time |
|---|---|
| M2 faithful, 1,704 × 1 call | 0.4 h |
| M4, B=8 on TEST-1704 | 5.7 h |
| M3, 3 seeds × B=4 on TEST-1704 | 5.3 h |
| M3, 3 seeds × B=8 on SUB-300 | 3.0 h |
| **Non-SLM total** | **~14.4 h** |

Peak memory stays around 3.2 GB MLX plus about 1.3 GB process.

Runs should be **resumable shards** (for example, E1 in 4 shards of about 4.3 h) with per-question append and skip-if-done. That is a runner change only: no method changes, and outputs are identical. Running non-SLM jobs at the same time as Qwen jobs is possible (Qwen uses the Metal GPU, the pre-filter uses the CPU), but the contention hasn't been measured, so the estimates above assume sequential runs.

**If time is short (in order of what to drop):**
1. A1 no-history: −3.1 h.
2. E3 B=1: −0.7 h.
3. Shrink SUB-300 to 200: −~6 h.

E1 and E2 shouldn't be cut. They're what RQ1 rests on.

## 7. Required tables and figures
| ID | Content | Set |
|---|---|---|
| **T1** | Main results at B=4: M0, M1, M2 (faithful), M3 (mean ± SD of 3 seeds), M4, M5. Columns: ID recall [95% CI], Δ vs M5 [CI], text recall, hit, questions gaining, new gold, expansions, SLM calls, tokens, latency | TEST-1704 |
| **T2** | Matched-expansion version of T1 (M1–M4 get M5's per-question expansions) + per-question win/tie/loss vs M5 + sign-test p | TEST-1704 |
| **T3** | Budget sweep: recall and cost for every method at B ∈ {1,2,4,8} (M5 rows for B≠4 on SUB-300, with the matching SUB-300 baseline rows) | TEST-1704 / SUB-300 |
| **T4** | Ablations: M5 vs A1, A2, A3. Recall, expansions used, STOP rate, tokens, latency, paired Δ [CI] | SUB-300 (A3 also TEST-1704) |
| **T5** | Stopping and multi-hop behavior: STOP rate, STOP-when-sufficient, wasted expansions, gold by hop depth, recall by post-hoc category | TEST-1704 |
| **T6** | Controller reliability: invalid first attempts, retries, CONTROLLER_ERROR, by B | all SLM runs |
| T7 | Development set (DEV-24): v1 → v2 changes (appendix) | DEV-24 |
| **F1** | Recall vs B (1, 2, 4, 8) for M1–M5; M3 as a mean with a ±SD band | SUB-300 + TEST-1704 |
| **F2** | Recall vs cost: x = SLM tokens or latency per question, one point per (method, B); Pareto frontier | SUB-300 |
| **F3** | Stacked bars of discovered gold by hop depth and category, M4 vs M5 | TEST-1704 |
| F4 | Histogram of the SLM's selected pre-filter rank by step (M4 is always rank 1) | TEST-1704 |
| F5 | Per-question paired outcome, M5 vs M4 (win/tie/loss), by category | TEST-1704 |

## 8. Execution order (after approval)
E0 → E2 / E2r (non-SLM, cheap, no Qwen) → E1 (sharded) → E4a (B=4) → SUB-300: E3, E4c, E4b, E4a (B=8) → analysis.

Every result file records the frozen code hashes; any hash mismatch aborts the run.
