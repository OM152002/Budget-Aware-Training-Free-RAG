# Results Interpretation

This document explains what our experiments found, organised around the project's three research questions. It is written so that someone new to the repository can follow it. Every number comes from the frozen results in [`analysis/`](analysis/): tables T1–T6, figures F1–F5 and the statistics in `analysis/stats/comparisons.json`. Each number has been checked against those files automatically (`analysis/verify_interpretation.py`).

**Status:** frozen interpretation (2026-11-09; restructured for readability 2026-11-12 with the same findings and numbers). No new analysis was run for this document.

---

## At a glance

| Research question | Short answer |
|---|---|
| **RQ1:** Can an untrained SLM pick useful passages to explore? | **Yes, modestly.** With a budget of 4, it beats every baseline on both test sets. The gain over the strongest baseline, flat search, is small (+0.027 recall). |
| **RQ2:** Does it know when to stop, and does it find multi-hop evidence? | **Partly.** Stopping saves about 7% of expansions and 9% of tokens, at a small recall cost (−0.0024). It isn't a precise "I have enough" signal. It does find more multi-hop gold than the score-based rule. |
| **RQ3:** Does it work well under a fixed budget? | **Per passage yes, per second no.** It sits on the recall-versus-expansion Pareto frontier at every budget tested, but its inference cost makes it much slower in wall-clock time. Hiding the budget value from the model made no measurable difference. |

---

## Before you read: terms used below

**Methods.** All of them share the same graph, the same three seed passages and the same candidate lists.

| Code | Method | In one line |
|---|---|---|
| **M5** | **SLM controller (ours)** | Qwen3-4B chooses which candidate to explore next, or stops |
| M4 | SAGE-score greedy | Always explores the top-scored candidate |
| M3 | Random | Explores a random candidate from the top 20 (averaged over 3 random seeds) |
| M2 | Original SAGE expansion | Adds the best-scoring neighbours of the seeds in one shot |
| M1 | Flat retrieval | Ignores the graph; takes more results from the initial search |
| M0 | Seeds only | No expansion |
| A1 / A2 / A3 | Ablations of M5 | No history / no budget value in the prompt / no ability to stop |

**Question sets**

| Set | Size | Role |
|---|---|---|
| **TEST-1704** | 1,704 questions | Main evaluation set |
| **NEW-1654** | 1,654 questions | The part of TEST-1704 that had never been scored before; a robustness check |
| **SUB-300** | 300 questions | A fixed random subset, used for the budget sweep and the ablations |

On SUB-300, differences smaller than about 0.04 can't be resolved: they're inconclusive, not evidence of no effect.

**Metrics and statistics**
- **ID Recall** (the main metric) is the fraction of a question's gold passages that end up in the retrieved evidence, averaged over questions. *Text Recall* also counts exact-text duplicates and is used only as a cross-check.
- **Budget (B)** is the maximum number of passages the method may add beyond the 3 seeds.
- **Δ** is SLM recall minus the other method's recall, computed on the same questions. A positive Δ means the SLM did better.
- **95% CI** is the bootstrap confidence interval for Δ. If it doesn't include 0, the difference is consistently in one direction.
- **W/T/L** counts the questions where the SLM did better, the same, or worse.
- **Holm p** is the sign-test p-value, corrected for multiple comparisons.
- **No single test decides.** We report the CI and Holm p side by side and don't use a single significance cut-off. When they agree, we say so; when they disagree, we say that too.
- **"Descriptive"** marks a result with no formal test, such as costs, changes across budgets, and category or hop breakdowns.

---

## RQ1: Can a training-free SLM pick useful passages?

> Can a pretrained SLM (Qwen3-4B-Instruct, 4-bit, no task-specific training) select relevant neighbouring passages well enough to beat non-learned traversal and retrieval baselines?

*Sources: T1 (main results), T2 (matched expansions), T5c (recall by category), F4 (rank of chosen candidate), F5 (per-question outcome vs M4).*

### What we found

With a budget of 4 on TEST-1704, the SLM had the highest recall of any method:

| Method | ID Recall |
|---|---|
| **M5 SLM** | **0.3609** |
| M1 flat retrieval | 0.3341 |
| M4 SAGE-score greedy | 0.3207 |
| M2 SAGE one-shot | 0.3109 |
| M3 random | 0.2706 |
| M0 seeds only | 0.2409 |

The results on NEW-1654 match these to within 0.0006 for every method.

Paired comparisons (SLM minus each baseline):

| SLM vs | Δ on TEST-1704 [95% CI] | W/T/L | Holm p | Δ on NEW-1654 [95% CI] | Holm p |
|---|---|---|---|---|---|
| M0 seeds only | +0.1200 [+0.1051, +0.1350] | 235/1469/0 | <0.0001 | +0.1201 [+0.1052, +0.1353] | <0.0001 |
| M1 flat retrieval | +0.0268 [+0.0102, +0.0429] | 153/1446/105 | 0.0034 | +0.0262 [+0.0100, +0.0430] | 0.0050 |
| M2 SAGE one-shot | +0.0500 [+0.0368, +0.0631] | 130/1537/37 | <0.0001 | +0.0498 [+0.0366, +0.0632] | <0.0001 |
| M3 random | +0.0902 [+0.0763, +0.1039] | 234/1404/66 | <0.0001 | +0.0904 [+0.0765, +0.1043] | <0.0001 |
| M4 SAGE-score greedy | +0.0402 [+0.0275, +0.0528] | 116/1543/45 | <0.0001 | +0.0407 [+0.0282, +0.0534] | <0.0001 |

- **Equal number of expansions (T2).** Each baseline was also given exactly as many expansions as the SLM used on each question. The gaps then widen slightly on TEST-1704: +0.0322 vs M1, +0.0533 vs M2, +0.0911 vs M3, +0.0445 vs M4. Every Holm p is ≤ 0.0002 on both sets.
- **Cross-check:** Text Recall moves the same way (0.3819 for M5 vs 0.3349 for M4).

### What the statistics say

The two tests **agree in all 18 comparisons**: both sets, full budget and matched expansions. Every CI excludes 0 and every Holm p is < 0.01.

Two descriptive breakdowns (no formal test):

| Where the gold is | M5 recall | M4 recall | W/T/L vs M4 |
|---|---|---|---|
| Gold reachable in the top-20 candidates (304 questions) | 0.5313 | 0.3838 | 76/200/28 |
| Gold not reachable within one hop | | | 32/859/16 |

The SLM picks the pre-filter's rank-1 candidate in only 36–42% of explorations, depending on the step (F4).

### What it means

Given exactly the candidates the SAGE score ranks, the SLM makes better choices than taking the top-ranked one. Seeds and candidates are identical across methods, so the gain over M4 comes from the selection decisions themselves.

The SLM regularly departs from the pre-filter's top choice, and on balance those departures help:
- more wins than losses against M4 overall (116 vs 45);
- more wins than losses in the category where gold is within reach (76 vs 28).

The result holds on the set that had never been scored before, so it isn't an artifact of the development questions.

### Caveats
- **Modest gains, mostly ties.** The improvement is consistent but small: only +0.027 over flat retrieval, and 1,543 of 1,704 questions tie against M4.
- **Recall is low overall (0.36), mainly because of graph reachability.** In 907 of the 1,704 questions, none of the remaining gold is within one hop of the seeds, and in that category every method's recall is ≤ 0.10.
- **Narrow evidence base:** one dataset (OTT-QA documents), a reconstructed graph, one SLM and one prompt.

---

## RQ2: Does the SLM know when to stop, and does it go multi-hop?

> Can the SLM tell when more exploration is needed and when it already has enough evidence? Does it make genuine multi-hop discoveries?

*Sources: T4 (ablations), T5a/T5b (stopping and hop depth), T6 (reliability), F3 (gold by hop depth).*

### What we found

**Stopping.** We compared the SLM with A3, an identical controller that isn't allowed to stop. The two differ only on the questions where the SLM chose to stop, so stopping can lose recall but never gain it.

| Set | Budget | Δ (SLM − no-stop) [95% CI] | W/T/L | p |
|---|---|---|---|---|
| TEST-1704 | 4 | −0.0024 [−0.0044, −0.0008] | 0/1695/9 | Holm 0.0273 |
| SUB-300 | 8 | −0.0100 [−0.0204, −0.0021] | 0/294/6 | Holm 0.1875 (raw 0.0312) |
| NEW-1654 (supplementary) | 4 | −0.0024 [−0.0046, −0.0008] | 0/1645/9 | raw 0.0039 |
| SUB-300 (supplementary) | 4 | −0.0069 [−0.0158, −0.0008] | 0/296/4 | raw 0.1250 |

What stopping bought (TEST-1704, budget 4, descriptive):

| | SLM (can stop) | No-stop (A3) |
|---|---|---|
| Expansions per question | 3.71 | 4.00 |
| SLM calls | 3.82 | 4.20 |
| Tokens | 7,976 | 8,790 |
| Latency | 36.5 s | 39.6 s |

How the stops played out (TEST-1704, budget 4):
- **How often:** the SLM stopped on 187 of 1,704 questions (11.0%); 83 of those stops came before any expansion.
- **What continuing would have found:** on 178 of the 187 stops, continuing to the full budget found no more gold. Restricted to the stopped questions, Δ = −0.0214 (and −0.0501 on the 60 stopped questions at budget 8 on SUB-300).
- **Whether the evidence was complete:** on 74 of the 187 stops, all gold was already in the evidence (checked afterwards against the gold labels).
- **By budget (SUB-300, descriptive):** the stop rate is 8.3% at B=1, 6.3% at B=2, 15.3% at B=4 and 20.0% at B=8, so it is highest at the largest budget.

**History.** Removing the traversal history (A1; SUB-300, budget 4) gave Δ = +0.0011 [−0.0133, +0.0167], W/T/L 3/293/4, Holm p = 1.0000. Without history, the model stopped less often: 9.0% vs 15.3% (descriptive).

**Multi-hop discovery** (TEST-1704, budget 4, descriptive):

| Method | Gold found 1 hop from the seeds | Gold found 2+ hops away |
|---|---|---|
| M5 SLM | 190 | 55 |
| M4 SAGE-score greedy | 135 | 40 |
| M3 random (mean) | 62.3 | 7.7 |

- Gold already in the seeds (hop 0) is 478 for every method.
- **Wasted expansions** (taken after the last gold was found): M5 3.42, M4 3.77, A3 3.70.

**Reliability (T6).** Every SLM run had 0 invalid first attempts, 0 retries and 0 controller errors.

### What the statistics say
- **Stopping at budget 4 (TEST-1704): the tests agree.**
  - The CI [−0.0044, −0.0008] excludes 0 and Holm p = 0.0273.
  - Both indicate that adaptive stopping costs a small amount of recall at the main setting.
- **Stopping at budget 8 (SUB-300): the tests disagree.**
  - The CI excludes 0 (−0.0100 [−0.0204, −0.0021]), but the Holm-adjusted sign-test p is 0.1875 (raw 0.0312).
  - The direction matches budget 4, but the evidence at this budget isn't consistent across the two tests.
- **History: the tests agree in finding no difference.**
  - The CI includes 0 and Holm p = 1.0000.
  - With SUB-300's resolution of about 0.04, this is inconclusive, not evidence that history is useless.
- **Descriptive only:** the cost savings, the stop-quality counts, the stop rate by budget, and all hop-depth and wasted-expansion comparisons.

### What it means
- **Stopping is conservative and cheap, but it buys efficiency, not accuracy.**
  - The SLM stops on about 11% of questions, saving about 7% of expansions and about 9% of tokens.
  - In 178 of 187 cases, continuing would have found nothing more.
  - It still gives up recall on 9 questions and never gains any, so it isn't free.
- **It isn't a precise "evidence is sufficient" signal.**
  - Only 74 of 187 stops (40%) happened with all gold already present.
  - Of the other 113 stops, 104 were on questions where continuing to the full budget found no more gold. Stopping there was harmless, but not because the evidence was complete.
- **Multi-hop discovery happens.** The SLM retrieves more gold two or more hops away than SAGE-score greedy (55 vs 40), so it does follow chains beyond the seeds' direct neighbours. This is descriptive, not a tested effect.
- **No difference from removing history was detected** at this sample size.
- **The interface is robust.** No invalid outputs across thousands of calls makes the controller practical to run.

### Caveats
- **Stopping is weakly tested:**
  - It's compared only against a forced-continuation ablation, and just 11–20% of questions stop, so most of each comparison is ties.
  - "All gold already in evidence" is an after-the-fact diagnostic, and the stops before any expansion weren't cross-tabulated with it (not planned).
  - The budget-8 stopping result and the history ablation rest on SUB-300, which can't resolve effects under about 0.04.
- **The hop-depth comparisons are counts without tests.** For flat retrieval (M1), "2+ hops" also includes passages with no graph connection, so its hop numbers aren't comparable to the traversal methods'.

---

## RQ3: How does the SLM do under a fixed budget?

> Under a fixed budget of graph expansions, does SLM-controlled traversal keep its effectiveness while limiting unnecessary exploration and inference cost?

*Sources: T3a/T3b/T3c (budget sweep), T4 (no-budget-input ablation), F1 (recall vs budget), F2 (recall vs cost), T1 (budget 4 on TEST-1704).*

### What we found

**Recall keeps rising with budget** (SUB-300, descriptive):

| Budget B | SLM ID Recall | Expansions per question | Share of budget used | Tokens | Latency |
|---|---|---|---|---|---|
| 1 | 0.2820 | 0.92 | 0.92 | 1,918 | 7.7 s |
| 2 | 0.2970 | 1.88 | 0.94 | 3,850 | 15.7 s |
| 4 | 0.3437 | 3.61 | 0.90 | 7,875 | 36.2 s |
| 8 | 0.3770 | 7.00 | 0.88 | 16,508 | 77.1 s |

Recall goes up at every step (+0.0150, +0.0467, +0.0333), while tokens and latency roughly double with each doubling of the budget.

**SLM vs each baseline at the same budget** (SUB-300), summarised:

| Compared with | What the two tests show |
|---|---|
| M0 seeds only | Agree at every budget: CI excludes 0, Holm p < 0.0001 |
| M2 SAGE one-shot | Agree at B = 1, 2, 4. Disagree at B = 8 (CI excludes 0; Holm p 0.0826) |
| M3 random | Agree at B = 1, 4, 8. Disagree at B = 2 (CI excludes 0; Holm p 0.0729) |
| M4 SAGE-score greedy | Agree at B = 1, 4. Disagree at B = 2 (Holm p 0.1846) and B = 8 (Holm p 0.2204; CI lower bound +0.0003) |
| M1 flat retrieval | Both find no difference at any budget: CI includes 0, Holm p ≥ 0.1686; level at B = 8 (+0.0006) |

Where both tests agree on a positive difference, every CI excludes 0 and every Holm p is ≤ 0.0234. On TEST-1704 at budget 4, the two tests agree for all five baselines (see RQ1).

<details>
<summary>Full per-budget comparison table (Δ [95% CI], Holm p)</summary>

| B | vs M0 seeds | vs M1 flat | vs M2 SAGE one-shot | vs M3 random | vs M4 SAGE-score greedy |
|---|---|---|---|---|---|
| 1 | +0.0557 [+0.0323, +0.0813], <0.0001 | +0.0187 [−0.0108, +0.0481], 0.1686 | +0.0307 [+0.0113, +0.0523], 0.0137 | +0.0465 [+0.0242, +0.0707], 0.0137 | +0.0257 [+0.0073, +0.0467], 0.0234 |
| 2 | +0.0707 [+0.0433, +0.0997], <0.0001 | +0.0117 [−0.0210, +0.0456], 0.5966 | +0.0350 [+0.0133, +0.0583], 0.0137 | +0.0532 [+0.0281, +0.0807], 0.0729 | +0.0247 [+0.0033, +0.0478], 0.1846 |
| 4 | +0.1173 [+0.0833, +0.1540], <0.0001 | +0.0223 [−0.0128, +0.0585], 0.3368 | +0.0531 [+0.0217, +0.0859], 0.0178 | +0.0885 [+0.0565, +0.1230], 0.0025 | +0.0476 [+0.0183, +0.0783], 0.0178 |
| 8 | +0.1507 [+0.1123, +0.1913], <0.0001 | +0.0006 [−0.0393, +0.0404], 0.8804 | +0.0518 [+0.0143, +0.0907], 0.0826 | +0.0919 [+0.0602, +0.1253], 0.0002 | +0.0329 [+0.0003, +0.0674], 0.2204 |

</details>

**Hiding the budget value from the model (A2, SUB-300)**

| B | Δ (SLM − A2) [95% CI] | W/T/L | Holm p |
|---|---|---|---|
| 1 | +0.0033 [+0.0000, +0.0100] | 1/299/0 | 1.0000 |
| 2 | −0.0033 [−0.0100, +0.0000] | 0/299/1 | 1.0000 |
| 4 | −0.0022 [−0.0156, +0.0111] | 2/294/4 | 1.0000 |
| 8 | +0.0083 [−0.0083, +0.0267] | 5/292/3 | 1.0000 |

Descriptively, A2 stops more often before any expansion (32 questions at every budget, against 17–25 for M5), and less often overall at B = 8 (15.3% vs 20.0%).

**Cost trade-offs (F2, SUB-300, descriptive)**
- **Cost = passages added:** the frontier runs from M0 through the SLM at B = 1, 2, 4 and 8. Flat retrieval at B = 8 (0.3764) sits just below the SLM at B = 8 (0.3770).
- **Cost = measured latency:**
  - The frontier runs M2 at B = 8 (0.3252, 0.8 s) → M4 at B = 8 (0.3441, 10.4 s) → SLM at B = 4 (0.3437, 36.2 s; 0.0004 below M4 at B = 8) → SLM at B = 8 (0.3770, 77.1 s).
  - M4 at B = 8 matches the SLM's B = 4 recall in about a third of the time.

### What the statistics say
- **Agree on a positive difference** (CI excludes 0, Holm p ≤ 0.0234):
  - SLM vs M0 at every budget;
  - vs M2 at B = 1, 2, 4;
  - vs M3 at B = 1, 4, 8;
  - vs M4 at B = 1, 4.
- **Disagree** (CI excludes 0, but Holm p is considerably larger, 0.0729–0.2204): vs M2 at B = 8, vs M3 at B = 2, and vs M4 at B = 2 and 8. These are reported as disagreements, with no verdict.
- **Agree on no difference:**
  - vs flat retrieval (M1) at every budget (CI includes 0, Holm p ≥ 0.1686);
  - vs the no-budget-input version (A2) at every budget (Holm p = 1.0000).
- **Descriptive only:** changes across budgets, the share of budget used, tokens and latency, the cost frontiers, and A2's stopping differences. No cross-budget tests were planned.

### What it means
- **Recall versus expansions:** the SLM lies on the recall-versus-expansion Pareto frontier at every tested budget (B = 1, 2, 4, 8; F2a, descriptive).
- **It doesn't under-explore.** Recall rises from B = 1 to B = 8 without levelling off, and the controller uses about 88–94% of the allowed budget.
- **The budget value itself barely matters.**
  - Showing the remaining budget in the prompt made no difference to recall that either test detected.
  - Whatever budget sensitivity the controller has doesn't depend on seeing the number. (The v2 system prompt still describes the budget as a maximum, without its value.)
  - Descriptively, the value changes *when* the controller stops (fewer stops before any expansion when it's shown), not how much it retrieves.
- **Wall-clock latency:** its inference cost makes it substantially less competitive. The SLM takes about 36 s per question at B = 4, while a non-SLM traversal with twice the budget (M4 at B = 8) matches that recall in about a third of the time (F2b, descriptive).
- **Flat retrieval stays close on SUB-300.** At every budget the CI includes 0 and Holm p is ≥ 0.1686, so the difference is not resolved at this sample size. At B = 8 the two are level (+0.0006).

### Caveats
- **Thin data away from budget 4.** The SLM's budget sweep exists only on SUB-300 (300 questions). Most per-budget differences are near or below its ~0.04 resolution, which is why the CI and the sign test disagree in several comparisons.
- **Cost comparisons are descriptive and partial.**
  - Latency was measured on one Apple M3 laptop with 4-bit MLX inference.
  - Several baseline latencies weren't measured exactly and are left out.
- **The budget counts graph expansions only**, so it doesn't capture the SLM's own inference cost, which dominates in practice.
