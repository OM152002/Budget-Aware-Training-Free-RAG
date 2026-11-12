# Results Interpretation: Project C (frozen v2 system)

**Status: FROZEN (2026-11-09).** Interpretation only. Nothing was run or recomputed. Every number comes from the validated, frozen outputs:
- tables T1–T6: `analysis/tables/`;
- figures F1–F5: `analysis/figures/`;
- statistics: `analysis/stats/comparisons.json`.

The DEV-24 appendix (T7) isn't used here.

**How to read this document**
- **Primary metric:** ID Recall of the final evidence set (seeds plus expansions), macro-averaged over questions. Text-equivalent Recall is mentioned only as a diagnostic.
- **Paired comparisons:** Δ = SLM (M5) − comparator, with a 95% paired-bootstrap CI and win/tie/loss counts (W = M5 higher). p is the Holm-adjusted exact sign-test p within the preregistered family.
- **No single significance criterion.** Per the frozen reporting decision, the paired-bootstrap CI and the Holm-adjusted sign-test p are reported side by side. Where they point the same way, both are stated. Where they disagree, the disagreement is stated explicitly and no verdict is assigned.
- **Descriptive:** everything not covered by a preregistered test, including all cost comparisons, cross-budget changes and category or hop breakdowns.
- **Populations:**
  - TEST-1704 (n=1,704) is the primary population.
  - NEW-1654 (n=1,654) is the robustness set that had never been scored before.
  - SUB-300 (n=300) is used for the budget sweep and ablations. It resolves paired differences of roughly ≥ 0.04 (EXPERIMENT_PLAN §0), so smaller SUB-300 differences are inconclusive, not evidence of no effect.
- **Methods:**
  - M0: seeds only.
  - M1: flat retrieval, top-(3+b) of the seed ranking, no graph.
  - M2: original SAGE one-shot expansion (faithful).
  - M3: random traversal (mean of 3 seeds).
  - M4: SAGE-score greedy traversal.
  - M5: the frozen SLM controller.
  - A1: SLM without history. A2: SLM without the budget value. A3: SLM without adaptive stopping.
- **Shared components:** every graph method uses the same reconstructed document graph, the same Fusion-A top-3 seeds, the same SAGE pre-filter (top-20) and the same duplicate-text handling. The first-step candidate list is identical for M3, M4, M5 and the ablations (verified on every question).

---

## RQ1: Training-free traversal effectiveness

**Question.** Can a pretrained SLM (Qwen3-4B-Instruct, 4-bit, no task-specific training) select relevant neighboring chunks well enough to beat non-learned traversal and retrieval baselines?

**Evidence used:**
- **T1:** main results, B=4, TEST-1704 and NEW-1654.
- **T2:** matched-expansion comparison.
- **T5c:** recall by post-hoc category.
- **F4:** pre-filter rank of the SLM's selections.
- **F5:** per-question outcome, M5 vs M4.

**Main quantitative findings (B=4)**

| SLM vs | Δ, TEST-1704 [95% CI] | W/T/L | Holm p | Δ, NEW-1654 [95% CI] | Holm p |
|---|---|---|---|---|---|
| M0 seeds only | +0.1200 [+0.1051, +0.1350] | 235/1469/0 | <0.0001 | +0.1201 [+0.1052, +0.1353] | <0.0001 |
| M1 flat retrieval | +0.0268 [+0.0102, +0.0429] | 153/1446/105 | 0.0034 | +0.0262 [+0.0100, +0.0430] | 0.0050 |
| M2 SAGE one-shot | +0.0500 [+0.0368, +0.0631] | 130/1537/37 | <0.0001 | +0.0498 [+0.0366, +0.0632] | <0.0001 |
| M3 random | +0.0902 [+0.0763, +0.1039] | 234/1404/66 | <0.0001 | +0.0904 [+0.0765, +0.1043] | <0.0001 |
| M4 SAGE-score greedy | +0.0402 [+0.0275, +0.0528] | 116/1543/45 | <0.0001 | +0.0407 [+0.0282, +0.0534] | <0.0001 |

- **Absolute ID Recall on TEST-1704:**
  - M5 0.3609 [0.3388, 0.3829];
  - M1 0.3341, M4 0.3207, M2 0.3109, M3 0.2706, M0 0.2409.

  NEW-1654 matches to within 0.0006 for every method.
- **Matched expansions (T2):** when each baseline gets exactly the SLM's per-question number of expansions, the gaps widen slightly: vs M1 +0.0322, M2 +0.0533, M3 +0.0911, M4 +0.0445 on TEST-1704. All four have Holm p ≤ 0.0002 on both populations.
- **Diagnostic:** text-equivalent Recall moves the same way (M5 0.3819 vs M4 0.3349).

**Statistical evidence vs descriptive results**
- **Tested comparisons:** M5 has higher recall than every baseline at B=4 on TEST-1704 and on NEW-1654, at full budget and with matched expansions. In all 18 comparisons, the two tests agree: every CI excludes 0 and every Holm-adjusted p is < 0.01.
- **Descriptive** (no preregistered test):
  - **Gains by category (T5c, F5):**

    | Category | M5 recall | M4 recall | W/T/L vs M4 |
    |---|---|---|---|
    | Gold reachable in top-20 (n=304) | 0.5313 | 0.3838 | 76/200/28 |
    | Not reachable in 1 hop | | | 32/859/16 |

  - **Rank choices (F4):** the SLM picks the pre-filter's rank-1 candidate in only 36–42% of explorations, depending on the step.

**What this means for the proposed method**
- A training-free SLM, choosing from the same candidate list the SAGE score ranks, picks neighbors that yield higher ID Recall than SAGE-score greedy, SAGE's original one-shot expansion, random traversal and flat retrieval.
- Because candidates and seeds are identical, the gain over M4 comes from the selection decisions themselves.
- The SLM regularly departs from the pre-filter's top choice, and on net those departures help: more wins than losses against M4 overall (116 vs 45) and in the main category (76 vs 28).
- The effect is the same on the set that had never been scored before, so it isn't an artifact of the development or pilot questions.

**Most important limitation**
- **Small absolute gains, mostly ties:** the improvement is consistent but modest, and against the strongest baseline it is small. M5 beats flat retrieval by only +0.027. Most questions are ties: 1,543 of 1,704 against M4.
- **Overall recall is low (0.36), and graph reachability is the main reason** (T5c):
  - 907 of 1,704 TEST questions have no remaining gold within one hop of the seeds;
  - in that category every method's recall is ≤ 0.10.
- **Narrow evidence base:** one dataset (OTT-QA documents only), a reconstructed graph, one SLM and one prompt.

---

## RQ2: Adaptive multi-hop retrieval and stopping

**Question.** Can the SLM tell when more graph exploration is needed and when it already has enough evidence, and does it make genuine multi-hop discoveries?

**Evidence used:**
- **T4:** M5 vs A3 (no adaptive stopping) and M5 vs A1 (no history), plus the stopped-questions breakdown.
- **T5a / T5b:** stopping behavior, and gold by hop depth with wasted expansions.
- **T6:** controller reliability.
- **F3:** gold by hop depth, M4 vs M5.

**Main quantitative findings**

*Effect of adaptive stopping (M5 − A3):*

| Population | B | Δ [95% CI] | W/T/L | p |
|---|---|---|---|---|
| TEST-1704 | 4 | −0.0024 [−0.0044, −0.0008] | 0/1695/9 | Holm 0.0273 |
| SUB-300 | 8 | −0.0100 [−0.0204, −0.0021] | 0/294/6 | Holm 0.1875 (raw 0.0312) |
| NEW-1654 (supplementary) | 4 | −0.0024 [−0.0046, −0.0008] | 0/1645/9 | raw 0.0039 |
| SUB-300 (supplementary) | 4 | −0.0069 [−0.0158, −0.0008] | 0/296/4 | raw 0.1250 |

- **Stopping can only cost recall here, never add it.** A3 is identical to M5 except on the questions where M5 stopped.
- **Restricted to the stopped questions:** Δ = −0.0214 on 187 questions (TEST-1704, B=4) and −0.0501 on 60 questions (SUB-300, B=8). On 178 of the 187 TEST-1704 stops, continuing to the full budget found no additional gold.
- **What stopping saves (TEST-1704, B=4, descriptive), M5 vs A3:**

  | | M5 | A3 |
  |---|---|---|
  | Expansions per question | 3.71 | 4.00 |
  | SLM calls | 3.82 | 4.20 |
  | Tokens | 7,976 | 8,790 |
  | Latency | 36.5 s | 39.6 s |

- **Stop quality (T5a, TEST-1704):** the SLM stopped on 187 of 1,704 questions (11.0%).
  - 83 of those stops came at step 0, before any expansion.
  - On 74 of the 187 stops, all gold was already in the evidence (checked post hoc with gold).
- **Stop rate by budget (SUB-300, descriptive):** 8.3% at B=1, 6.3% at B=2, 15.3% at B=4, 20.0% at B=8.

*Effect of traversal history (M5 − A1, SUB-300, B=4):* Δ = +0.0011 [−0.0133, +0.0167], W/T/L 3/293/4, Holm p = 1.0000. A1 stopped less often (9.0% vs 15.3%, descriptive).

*Multi-hop discovery (TEST-1704, B=4, descriptive; T5b, F3):*

| Method | Gold found at hop 1 | Gold found at hop 2+ |
|---|---|---|
| M5 SLM | 190 | 55 |
| M4 SAGE-score greedy | 135 | 40 |
| M3 random (mean) | 62.3 | 7.7 |

- Hop 0 (gold already in the seeds) is 478 for every method.
- **Wasted expansions** (taken after the last gold found): M5 3.42, M4 3.77, A3 3.70.

*Reliability (T6):* 0 invalid first attempts, 0 retries and 0 controller errors in every SLM run.

**Statistical evidence vs descriptive results**
- **Stopping at B=4 (TEST-1704):** the two tests agree. Disabling STOP (A3) gives slightly *higher* recall than M5: Δ = −0.0024, CI [−0.0044, −0.0008] excludes 0, Holm p = 0.0273. Both indicate that adaptive stopping costs a small amount of recall at the primary setting.
- **Stopping at B=8 (SUB-300):** the two tests disagree. The CI excludes 0 (−0.0100 [−0.0204, −0.0021]), but the Holm-adjusted sign-test p is 0.1875 (raw 0.0312). The direction matches B=4; the evidence at this budget is not consistent across the two tests.
- **History (A1):** the two tests agree in finding no difference: the CI includes 0 and Holm p = 1.0000. With SUB-300's resolution of ~0.04, this is inconclusive, not evidence that history is useless.
- **Descriptive only:**
  - the cost savings from stopping;
  - the stop-quality counts;
  - the stop-rate pattern across budgets;
  - all hop-depth and wasted-expansion comparisons.

**What this means for the proposed method**
- **Stopping is conservative and cheap, but its benefit is efficiency, not accuracy.** The SLM stops on about 11% of questions at B=4. In 178 of 187 cases, exploring to the full budget would have found nothing more, and stopping saves about 7% of expansions and about 9% of tokens. But it gives up recall on 9 questions and never gains any, so it is not free.
- **It isn't a precise "evidence is sufficient" signal.** Only 74 of 187 stops (40%) happened when all gold was already present. Of the other 113 stops, 104 were on questions where continuing to the full budget found no additional gold. Stopping there was harmless, but not because the evidence was complete.
- **Multi-hop discovery happens.** The SLM retrieves more gold at hop 2+ than SAGE-score greedy (55 vs 40), so it does follow chains beyond the seeds' direct neighbors. This is descriptive and isn't a tested effect.
- **No difference from removing history was detected** at this sample size.
- **The interface is robust.** No invalid outputs in thousands of calls means the controller is practical to run.

**Most important limitation**
- **Weak test of stopping:**
  - Stopping is assessed only against a forced-continuation ablation, and only about 11–20% of questions stop, so most of each comparison is ties.
  - "All gold already in evidence" is a post-hoc gold-based diagnostic, and the step-0 stops weren't cross-tabulated with it (not planned).
  - The B=8 stopping result and the history ablation rest on SUB-300, which can't resolve effects under about 0.04.
- **Hop-depth caveats:**
  - The hop-depth comparisons are counts without tests.
  - For flat retrieval (M1), "hop 2+" also includes passages with no graph connection, so M1's hop numbers aren't comparable to the traversal methods'.

---

## RQ3: Budget-aware retrieval

**Question.** Under a fixed graph-expansion budget, does SLM-controlled traversal maintain effectiveness while limiting unnecessary exploration and inference cost?

**Evidence used:**
- **T3a:** SUB-300, all methods at B = 1, 2, 4, 8, with paired tests per budget.
- **T3b:** TEST-1704 baselines across budgets, descriptive.
- **T3c:** the SLM across budgets, descriptive.
- **T4:** M5 vs A2 (no budget input).
- **F1:** recall vs B. **F2:** recall vs cost, with Pareto frontiers.
- **T1** for B=4 on TEST-1704.

**Main quantitative findings**

*M5 vs baselines at each budget, SUB-300 (Δ [95% CI], Holm p):*

| B | vs M0 seeds | vs M1 flat | vs M2 SAGE one-shot | vs M3 random | vs M4 SAGE-score greedy |
|---|---|---|---|---|---|
| 1 | +0.0557 [+0.0323, +0.0813], <0.0001 | +0.0187 [−0.0108, +0.0481], 0.1686 | +0.0307 [+0.0113, +0.0523], 0.0137 | +0.0465 [+0.0242, +0.0707], 0.0137 | +0.0257 [+0.0073, +0.0467], 0.0234 |
| 2 | +0.0707 [+0.0433, +0.0997], <0.0001 | +0.0117 [−0.0210, +0.0456], 0.5966 | +0.0350 [+0.0133, +0.0583], 0.0137 | +0.0532 [+0.0281, +0.0807], 0.0729 | +0.0247 [+0.0033, +0.0478], 0.1846 |
| 4 | +0.1173 [+0.0833, +0.1540], <0.0001 | +0.0223 [−0.0128, +0.0585], 0.3368 | +0.0531 [+0.0217, +0.0859], 0.0178 | +0.0885 [+0.0565, +0.1230], 0.0025 | +0.0476 [+0.0183, +0.0783], 0.0178 |
| 8 | +0.1507 [+0.1123, +0.1913], <0.0001 | +0.0006 [−0.0393, +0.0404], 0.8804 | +0.0518 [+0.0143, +0.0907], 0.0826 | +0.0919 [+0.0602, +0.1253], 0.0002 | +0.0329 [+0.0003, +0.0674], 0.2204 |

*The SLM across budgets (SUB-300, T3c, descriptive):*

| B | ID Recall | Expansions per question | Expansions / B | Tokens | Latency |
|---|---|---|---|---|---|
| 1 | 0.2820 | 0.92 | 0.92 | 1,918 | 7.7 s |
| 2 | 0.2970 | 1.88 | 0.94 | 3,850 | 15.7 s |
| 4 | 0.3437 | 3.61 | 0.90 | 7,875 | 36.2 s |
| 8 | 0.3770 | 7.00 | 0.88 | 16,508 | 77.1 s |

- Recall rises at every step (+0.0150, +0.0467, +0.0333), while tokens and latency roughly double with each doubling of B.

*Budget input (M5 − A2, SUB-300):*

| B | Δ [95% CI] | W/T/L | Holm p |
|---|---|---|---|
| 1 | +0.0033 [+0.0000, +0.0100] | 1/299/0 | 1.0000 |
| 2 | −0.0033 [−0.0100, +0.0000] | 0/299/1 | 1.0000 |
| 4 | −0.0022 [−0.0156, +0.0111] | 2/294/4 | 1.0000 |
| 8 | +0.0083 [−0.0083, +0.0267] | 5/292/3 | 1.0000 |

- Descriptively, A2 stops more often at step 0 (32 questions at every B, vs 17–25 for M5) and less often overall at B=8 (15.3% vs 20.0%).

*Cost frontiers (F2, SUB-300, descriptive):*
- **Cost = expansions:** the frontier runs through M0 and then M5 at B = 1, 2, 4 and 8. M1 at B=8 (0.3764) sits just below M5 at B=8 (0.3770).
- **Cost = measured latency:**
  - The frontier runs M2 at B=8 (0.3252, 0.8 s) → M4 at B=8 (0.3441, 10.4 s) → M5 at B=4 (0.3437, 36.2 s; level with M4 at B=8, 0.0004 lower) → M5 at B=8 (0.3770, 77.1 s).
  - M4 at B=8 reaches the same recall as M5 at B=4 with about a third of the latency.

**Statistical evidence vs descriptive results**
- **Where the CI and the Holm-adjusted p agree on a positive difference (SUB-300):** every CI excludes 0 and every Holm p is ≤ 0.0234 for:
  - M5 vs M0 at every B;
  - M5 vs M2 at B = 1, 2 and 4;
  - M5 vs M3 at B = 1, 4 and 8;
  - M5 vs M4 at B = 1 and 4.
- **On TEST-1704 at B=4:** the two tests agree for all five baselines (T1; see RQ1).
- **Where the two tests disagree (SUB-300):** the CI excludes 0, but the Holm-adjusted sign-test p is considerably larger (0.0729–0.2204):
  - M5 vs M2 at B=8 (Holm p 0.0826);
  - M5 vs M3 at B=2 (0.0729);
  - M5 vs M4 at B=2 (0.1846);
  - M5 vs M4 at B=8 (0.2204; CI lower bound +0.0003).

  These are reported as disagreements, with no verdict.
- **Where both find no difference:** the CI includes 0 and the Holm p is ≥ 0.1686 for:
  - M5 vs M1 at every budget on SUB-300 (B=8 Δ +0.0006);
  - M5 vs A2 (hidden budget value) at every budget (Holm p = 1.0000).
- **Descriptive only:**
  - all changes across budgets;
  - expansions/B, tokens and latency;
  - the Pareto frontiers;
  - A2's stopping differences.

  No cross-budget tests were preregistered (frozen decision 3).

**What this means for the proposed method**
- **Recall versus expansions:** the SLM lies on the recall-versus-expansion Pareto frontier at every tested budget (B = 1, 2, 4, 8; F2a, descriptive).
- **Recall keeps rising with budget:** it never plateaus between B=1 and B=8, and the controller uses about 88–94% of the allowed budget. So it doesn't systematically under-explore.
- **Budget-awareness:**
  - Showing the remaining budget in the prompt made no difference to recall that either test detected (CI includes 0 and Holm p = 1.0000 at every budget).
  - Whatever budget sensitivity the controller has doesn't depend on seeing the budget number. The v2 system prompt still describes the budget as a maximum without giving its value.
  - Descriptively, the budget value changes *when* the controller stops (fewer step-0 stops when it's shown), but not how much it retrieves.
- **Wall-clock latency:** its inference cost makes it substantially less competitive here. The SLM takes about 36 s per question at B=4, and a non-SLM traversal with twice the budget (M4 at B=8) matches that recall in about a third of the time (F2b, descriptive).
- **Flat retrieval on SUB-300:** at every budget the CI includes 0 and the Holm p is ≥ 0.1686, so the difference is not resolved at this sample size. At B=8 the two are level (+0.0006).

**Most important limitation**
- **Thin SLM data off B=4:** the budget sweep for the SLM exists only on SUB-300 (n=300). Most per-budget differences are near or below its ~0.04 resolution, which is why the CI and the sign test disagree in several comparisons.
- **Cost comparisons are descriptive and partial:**
  - latency was measured on one Apple M3 laptop with 4-bit MLX inference;
  - several baseline latencies weren't measured exactly and are omitted (D5).
- **The budget counts graph expansions only**, so it doesn't capture the SLM's own inference cost, which dominates in practice.
