# Tables T1–T6 (canonical scores and statistics; no interpretation)

## T1

### T1 (TEST-1704, PRIMARY): main results, budget B = 4, maximum budget (n = 1704)

| Method | ID Recall [95% CI] | Δ = M5 − method [95% CI] | W/T/L | Holm p (sign) | Text Recall (diag.) | Hit rate | Questions gaining | New gold | Expansions | SLM calls | Tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 seeds only | 0.2409 [0.2215, 0.2610] | +0.1200 [+0.1051, +0.1350] | 235/1469/0 | <0.0001 | 0.2442 | 0.2635 | 0 | 0 | 0.00 | 0 | 0 | — |
| M1 flat top-(3+b) | 0.3341 [0.3130, 0.3558] | +0.0268 [+0.0102, +0.0429] | 153/1446/105 | 0.0034 | 0.3359 | 0.3592 | 187 | 207 | 4.00 | 0 | 0 | — |
| M2 SAGE one-shot (faithful) | 0.3109 [0.2899, 0.3323] | +0.0500 [+0.0368, +0.0631] | 130/1537/37 | <0.0001 | 0.3159 | 0.3374 | 141 | 152 | 3.98 | 0 | 0 | 0.7 |
| M3 random (mean of 3 seeds) | 0.2706 ± 0.0046 (seed SD) [0.2514, 0.2901] | +0.0902 [+0.0763, +0.1039] | 234/1404/66 | <0.0001 | 0.2771 | 0.2969 | 67 | 70 | 4.00 | 0 | 0 | 3.8 |
| M4 SAGE-score greedy | 0.3207 [0.2993, 0.3421] | +0.0402 [+0.0275, +0.0528] | 116/1543/45 | <0.0001 | 0.3349 | 0.3474 | 161 | 175 | 4.00 | 0 | 0 | — |
| **M5 SLM (frozen, full)** | 0.3609 [0.3388, 0.3829] | — | — | — | 0.3819 | 0.3891 | 235 | 245 | 3.71 | 3.82 | 7976 | 36.5 |

- ID Recall is primary; Text Recall is diagnostic only. Δ, CI and W/T/L are paired per question; W = M5 higher.
- Holm correction within this table's family (m = 5, TEST-1704). CIs are unadjusted; no significance label is assigned.
- Baselines always use all 4 expansions (except when candidates run out); M5 may STOP earlier.
- M3: per-question mean over random seeds {0,1,2}; ± is the SD of the three seed-level recalls.
- Latency is wall time per question and only where measured exactly (D5): M4 at B=4 is a truncation of the B=8 run, so its latency is not reported. M0/M1 add no retrieval cost beyond the shared seed retrieval.
- M2 (dedup, diagnostic) is reported in the supplementary statistics, not here.

### T1 (NEW-1654, never-previously-scored robustness set): main results, budget B = 4, maximum budget (n = 1654)

| Method | ID Recall [95% CI] | Δ = M5 − method [95% CI] | W/T/L | Holm p (sign) | Text Recall (diag.) | Hit rate | Questions gaining | New gold | Expansions | SLM calls | Tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 seeds only | 0.2406 [0.2210, 0.2611] | +0.1201 [+0.1052, +0.1353] | 228/1426/0 | <0.0001 | 0.2440 | 0.2636 | 0 | 0 | 0.00 | 0 | 0 | — |
| M1 flat top-(3+b) | 0.3345 [0.3123, 0.3564] | +0.0262 [+0.0100, +0.0430] | 146/1407/101 | 0.0050 | 0.3363 | 0.3603 | 183 | 203 | 4.00 | 0 | 0 | — |
| M2 SAGE one-shot (faithful) | 0.3109 [0.2890, 0.3329] | +0.0498 [+0.0366, +0.0632] | 125/1494/35 | <0.0001 | 0.3154 | 0.3374 | 137 | 148 | 3.98 | 0 | 0 | 0.7 |
| M3 random (mean of 3 seeds) | 0.2704 ± 0.0047 (seed SD) [0.2505, 0.2905] | +0.0904 [+0.0765, +0.1043] | 227/1364/63 | <0.0001 | 0.2770 | 0.2969 | 65 | 68 | 4.00 | 0 | 0 | 3.8 |
| M4 SAGE-score greedy | 0.3201 [0.2981, 0.3421] | +0.0407 [+0.0282, +0.0534] | 111/1502/41 | <0.0001 | 0.3342 | 0.3470 | 155 | 169 | 4.00 | 0 | 0 | — |
| **M5 SLM (frozen, full)** | 0.3607 [0.3384, 0.3836] | — | — | — | 0.3818 | 0.3888 | 228 | 238 | 3.70 | 3.82 | 7976 | 36.5 |

- ID Recall is primary; Text Recall is diagnostic only. Δ, CI and W/T/L are paired per question; W = M5 higher.
- Holm correction within this table's family (m = 5, NEW-1654). CIs are unadjusted; no significance label is assigned.
- Baselines always use all 4 expansions (except when candidates run out); M5 may STOP earlier.
- M3: per-question mean over random seeds {0,1,2}; ± is the SD of the three seed-level recalls.
- Latency is wall time per question and only where measured exactly (D5): M4 at B=4 is a truncation of the B=8 run, so its latency is not reported. M0/M1 add no retrieval cost beyond the shared seed retrieval.
- M2 (dedup, diagnostic) is reported in the supplementary statistics, not here.

## T2

### T2 (TEST-1704): matched-expansion comparison, B = 4 (n = 1704)

| Method (matched to M5's per-question expansions) | ID Recall [95% CI] | Δ = M5 − method [95% CI] | W/T/L | Holm p (sign) | Text Recall (diag.) | Expansions |
|---|---|---|---|---|---|---|
| **M5 SLM (frozen, full)** | 0.3609 [0.3388, 0.3829] | — | — | — | 0.3819 | 3.71 |
| M1 flat top-(3+b) | 0.3287 [0.3077, 0.3500] | +0.0322 [+0.0160, +0.0481] | 155/1456/93 | <0.0001 | 0.3319 | 3.71 |
| M2 SAGE one-shot (faithful) | 0.3076 [0.2867, 0.3289] | +0.0533 [+0.0399, +0.0664] | 133/1540/31 | <0.0001 | 0.3128 | 3.69 |
| M3 random (mean of 3 seeds) | 0.2698 [0.2505, 0.2893] | +0.0911 [+0.0772, +0.1047] | 234/1409/61 | <0.0001 | 0.2762 | 3.71 |
| M4 SAGE-score greedy | 0.3163 [0.2952, 0.3376] | +0.0445 [+0.0319, +0.0571] | 118/1550/36 | <0.0001 | 0.3297 | 3.71 |

- Each baseline gets exactly the number of expansions M5 used on that question (fewer only where the baseline ran out of candidates). Exact prefix truncation of the logged runs.
- Holm correction within this table's family (m = 4, TEST-1704); exact two-sided sign test, ties dropped.
- ID Recall primary; Text Recall diagnostic only.

### T2 (NEW-1654): matched-expansion comparison, B = 4 (n = 1654)

| Method (matched to M5's per-question expansions) | ID Recall [95% CI] | Δ = M5 − method [95% CI] | W/T/L | Holm p (sign) | Text Recall (diag.) | Expansions |
|---|---|---|---|---|---|---|
| **M5 SLM (frozen, full)** | 0.3607 [0.3384, 0.3836] | — | — | — | 0.3818 | 3.70 |
| M1 flat top-(3+b) | 0.3295 [0.3076, 0.3514] | +0.0312 [+0.0148, +0.0479] | 148/1416/90 | 0.0002 | 0.3329 | 3.70 |
| M2 SAGE one-shot (faithful) | 0.3075 [0.2858, 0.3295] | +0.0533 [+0.0401, +0.0664] | 128/1497/29 | <0.0001 | 0.3123 | 3.69 |
| M3 random (mean of 3 seeds) | 0.2695 [0.2497, 0.2896] | +0.0913 [+0.0775, +0.1052] | 227/1369/58 | <0.0001 | 0.2760 | 3.70 |
| M4 SAGE-score greedy | 0.3156 [0.2940, 0.3376] | +0.0451 [+0.0326, +0.0577] | 113/1509/32 | <0.0001 | 0.3288 | 3.70 |

- Each baseline gets exactly the number of expansions M5 used on that question (fewer only where the baseline ran out of candidates). Exact prefix truncation of the logged runs.
- Holm correction within this table's family (m = 4, NEW-1654); exact two-sided sign test, ties dropped.
- ID Recall primary; Text Recall diagnostic only.

## T3

### T3a (SUB-300): budget sweep, every method at B ∈ {1, 2, 4, 8} (n = 300)

| Method | B | ID Recall [95% CI] | Δ = M5 − method [95% CI] | W/T/L | Holm p (sign) | Text Recall (diag.) | Expansions | Expansions / B | Pre-filter calls | SLM calls | Tokens | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 seeds only | 1 | 0.2263 [0.1819, 0.2733] | +0.0557 [+0.0323, +0.0813] | 20/280/0 | <0.0001 | 0.2263 | 0.00 | — | 0.00 | 0 | 0 | — |
| M1 flat top-(3+b) | 1 | 0.2632 [0.2164, 0.3126] | +0.0187 [-0.0108, +0.0481] | 17/274/9 | 0.1686 | 0.2682 | 1.00 | 1.00 | 0.00 | 0 | 0 | — |
| M2 SAGE one-shot (faithful) | 1 | 0.2513 [0.2052, 0.2993] | +0.0307 [+0.0113, +0.0523] | 12/287/1 | 0.0137 | 0.2513 | 1.00 | 1.00 | 1.00 | 0 | 0 | 0.8 |
| M3 random (mean of 3 seeds) | 1 | 0.2355 ± 0.0019 [0.1916, 0.2816] | +0.0465 [+0.0242, +0.0707] | 20/275/5 | 0.0137 | 0.2366 | 1.00 | 1.00 | 1.00 | 0 | 0 | — |
| M4 SAGE-score greedy | 1 | 0.2563 [0.2099, 0.3052] | +0.0257 [+0.0073, +0.0467] | 10/289/1 | 0.0234 | 0.2597 | 1.00 | 1.00 | 1.00 | 0 | 0 | — |
| **M5 SLM (frozen, full)** | 1 | 0.2820 [0.2345, 0.3322] | — | — | — | 0.2853 | 0.92 | 0.92 | 1.00 | 1.00 | 1918 | 7.7 |
| M0 seeds only | 2 | 0.2263 [0.1819, 0.2733] | +0.0707 [+0.0433, +0.0997] | 24/276/0 | <0.0001 | 0.2263 | 0.00 | — | 0.00 | 0 | 0 | — |
| M1 flat top-(3+b) | 2 | 0.2853 [0.2374, 0.3358] | +0.0117 [-0.0210, +0.0456] | 18/268/14 | 0.5966 | 0.2914 | 2.00 | 1.00 | 0.00 | 0 | 0 | — |
| M2 SAGE one-shot (faithful) | 2 | 0.2620 [0.2152, 0.3113] | +0.0350 [+0.0133, +0.0583] | 12/287/1 | 0.0137 | 0.2703 | 2.00 | 1.00 | 1.00 | 0 | 0 | 0.8 |
| M3 random (mean of 3 seeds) | 2 | 0.2438 ± 0.0027 [0.1996, 0.2898] | +0.0532 [+0.0281, +0.0807] | 24/266/10 | 0.0729 | 0.2449 | 2.00 | 1.00 | 2.00 | 0 | 0 | — |
| M4 SAGE-score greedy | 2 | 0.2722 [0.2247, 0.3217] | +0.0247 [+0.0033, +0.0478] | 10/287/3 | 0.1846 | 0.2772 | 2.00 | 1.00 | 2.00 | 0 | 0 | — |
| **M5 SLM (frozen, full)** | 2 | 0.2970 [0.2488, 0.3480] | — | — | — | 0.3064 | 1.88 | 0.94 | 1.94 | 1.94 | 3850 | 15.7 |
| M0 seeds only | 4 | 0.2263 [0.1819, 0.2733] | +0.1173 [+0.0833, +0.1540] | 38/262/0 | <0.0001 | 0.2263 | 0.00 | — | 0.00 | 0 | 0 | — |
| M1 flat top-(3+b) | 4 | 0.3214 [0.2720, 0.3737] | +0.0223 [-0.0128, +0.0585] | 23/261/16 | 0.3368 | 0.3247 | 4.00 | 1.00 | 0.00 | 0 | 0 | — |
| M2 SAGE one-shot (faithful) | 4 | 0.2906 [0.2424, 0.3414] | +0.0531 [+0.0217, +0.0859] | 21/273/6 | 0.0178 | 0.2939 | 3.98 | 1.00 | 1.00 | 0 | 0 | 0.8 |
| M3 random (mean of 3 seeds) | 4 | 0.2552 ± 0.0076 [0.2114, 0.3010] | +0.0885 [+0.0565, +0.1230] | 38/249/13 | 0.0025 | 0.2591 | 4.00 | 1.00 | 4.00 | 0 | 0 | 4.0 |
| M4 SAGE-score greedy | 4 | 0.2961 [0.2467, 0.3463] | +0.0476 [+0.0183, +0.0783] | 19/276/5 | 0.0178 | 0.3078 | 4.00 | 1.00 | 4.00 | 0 | 0 | — |
| **M5 SLM (frozen, full)** | 4 | 0.3437 [0.2925, 0.3968] | — | — | — | 0.3620 | 3.61 | 0.90 | 3.77 | 3.77 | 7875 | 36.2 |
| M0 seeds only | 8 | 0.2263 [0.1819, 0.2733] | +0.1507 [+0.1123, +0.1913] | 49/251/0 | <0.0001 | 0.2263 | 0.00 | — | 0.00 | 0 | 0 | — |
| M1 flat top-(3+b) | 8 | 0.3764 [0.3226, 0.4305] | +0.0006 [-0.0393, +0.0404] | 21/256/23 | 0.8804 | 0.3764 | 8.00 | 1.00 | 0.00 | 0 | 0 | — |
| M2 SAGE one-shot (faithful) | 8 | 0.3252 [0.2752, 0.3770] | +0.0518 [+0.0143, +0.0907] | 28/259/13 | 0.0826 | 0.3252 | 7.93 | 0.99 | 1.00 | 0 | 0 | 0.8 |
| M3 random (mean of 3 seeds) | 8 | 0.2851 ± 0.0039 [0.2406, 0.3314] | +0.0919 [+0.0602, +0.1253] | 49/235/16 | 0.0002 | 0.2910 | 8.00 | 1.00 | 8.00 | 0 | 0 | 10.8 |
| M4 SAGE-score greedy | 8 | 0.3441 [0.2932, 0.3970] | +0.0329 [+0.0003, +0.0674] | 21/268/11 | 0.2204 | 0.3657 | 8.00 | 1.00 | 8.00 | 0 | 0 | 10.4 |
| **M5 SLM (frozen, full)** | 8 | 0.3770 [0.3242, 0.4312] | — | — | — | 0.3987 | 7.00 | 0.88 | 7.20 | 7.20 | 16508 | 77.1 |

- M5 at B = 1/2/8 exists only on SUB-300; M5 at B = 4 on SUB-300 = 291 questions from E1 + 9 from PILOT-50.
- Holm correction within each budget's family (m = 5 per B, SUB-300). CIs unadjusted; no significance label.
- M0 (no expansion) is repeated in each block as the per-B family reference; its Δ differs because M5 differs.
- SUB-300 resolves paired differences of roughly ≥ 0.04 (EXPERIMENT_PLAN §0); smaller Δ are inconclusive at this n.
- Latency only where measured exactly (D5): M3 at B = 4 (E2r) and B = 8 (E2r8), M4 at B = 8, M2 faithful, all SLM rows.
- ID Recall primary; Text Recall diagnostic only.

### T3b (TEST-1704): baselines across budgets (n = 1,704), descriptive

| Method | B | ID Recall | Text Recall (diag.) | Expansions | Pre-filter calls | Latency (s) |
|---|---|---|---|---|---|---|
| M0 seeds only | 0 | 0.2409 | 0.2442 | 0.00 | 0.00 | — |
| M1 flat top-(3+b) | 1 | 0.2712 | 0.2746 | 1.00 | 0.00 | — |
| M2 SAGE one-shot (faithful) | 1 | 0.2653 | 0.2694 | 1.00 | 1.00 | 0.7 |
| M4 SAGE-score greedy | 1 | 0.2653 | 0.2718 | 1.00 | 1.00 | — |
| M3 random (mean of 3 seeds) | 1 | 0.2494 ± 0.0021 | 0.2533 | 1.00 | 1.00 | — |
| M1 flat top-(3+b) | 2 | 0.2957 | 0.2997 | 2.00 | 0.00 | — |
| M2 SAGE one-shot (faithful) | 2 | 0.2828 | 0.2889 | 1.99 | 1.00 | 0.7 |
| M4 SAGE-score greedy | 2 | 0.2866 | 0.2963 | 2.00 | 2.00 | — |
| M3 random (mean of 3 seeds) | 2 | 0.2575 ± 0.0027 | 0.2621 | 2.00 | 2.00 | — |
| M1 flat top-(3+b) | 4 | 0.3341 | 0.3359 | 4.00 | 0.00 | — |
| M2 SAGE one-shot (faithful) | 4 | 0.3109 | 0.3159 | 3.98 | 1.00 | 0.7 |
| M4 SAGE-score greedy | 4 | 0.3207 | 0.3349 | 4.00 | 4.00 | — |
| M3 random (mean of 3 seeds) | 4 | 0.2706 ± 0.0046 | 0.2771 | 4.00 | 4.00 | 3.8 |
| **M5 SLM (frozen, full)** | 4 | 0.3609 | 0.3819 | 3.71 | 3.82 | 36.5 |
| M1 flat top-(3+b) | 8 | 0.3872 | 0.3893 | 8.00 | 0.00 | — |
| M2 SAGE one-shot (faithful) | 8 | 0.3467 | 0.3485 | 7.91 | 1.00 | 0.7 |
| M4 SAGE-score greedy | 8 | 0.3558 | 0.3760 | 8.00 | 8.00 | 10.1 |

- M5 exists on TEST-1704 only at B = 4 (paired tests in T1). M3 at B = 8 exists only on SUB-300 (T3a).
- No paired tests in this panel (none preregistered for TEST-1704 at B ≠ 4).
- ID Recall primary; Text Recall diagnostic only.

### T3c (SUB-300): M5 across budgets, descriptive only (n = 300)

| B | ID Recall | Change vs previous B | Expansions | STOP rate | SLM calls | Tokens | Token change | Latency (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2820 | — | 0.92 | 0.083 | 1.00 | 1918 | — | 7.7 |
| 2 | 0.2970 | +0.0150 | 1.88 | 0.063 | 1.94 | 3850 | +1932 | 15.7 |
| 4 | 0.3437 | +0.0467 | 3.61 | 0.153 | 3.77 | 7875 | +4025 | 36.2 |
| 8 | 0.3770 | +0.0333 | 7.00 | 0.200 | 7.20 | 16508 | +8633 | 77.1 |

- Changes between budgets are differences of canonical means; no cross-budget significance test (frozen decision 3).

## T4

### T4: ablations, M5 (full controller) vs A1 no history, A2 no budget input, A3 no adaptive stopping

| Comparison | Population | B | M5 ID Recall | Ablation ID Recall [95% CI] | Δ = M5 − ablation [95% CI] | W/T/L | p (sign) | Text Recall abl. (diag.) | Expansions M5 / abl. | STOP rate M5 / abl. | SLM calls M5 / abl. | Tokens M5 / abl. | Latency (s) M5 / abl. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 no history | SUB-300 | 4 | 0.3437 | 0.3425 [0.2923, 0.3956] | +0.0011 [-0.0133, +0.0167] | 3/293/4 | 1.0000 (Holm) | 0.3542 | 3.61 / 3.72 | 0.153 / 0.090 | 3.77 / 3.81 | 7875 / 7813 | 36.2 / 33.4 |
| A2 no budget input | SUB-300 | 1 | 0.2820 | 0.2787 [0.2314, 0.3289] | +0.0033 [+0.0000, +0.0100] | 1/299/0 | 1.0000 (Holm) | 0.2820 | 0.92 / 0.89 | 0.083 / 0.107 | 1.00 / 1.00 | 1918 / 1907 | 7.7 / 8.4 |
| A2 no budget input | SUB-300 | 2 | 0.2970 | 0.3003 [0.2522, 0.3509] | -0.0033 [-0.0100, +0.0000] | 0/299/1 | 1.0000 (Holm) | 0.3070 | 1.88 / 1.79 | 0.063 / 0.107 | 1.94 / 1.89 | 3850 / 3725 | 15.7 / 16.5 |
| A2 no budget input | SUB-300 | 4 | 0.3437 | 0.3459 [0.2953, 0.3988] | -0.0022 [-0.0156, +0.0111] | 2/294/4 | 1.0000 (Holm) | 0.3642 | 3.61 / 3.53 | 0.153 / 0.133 | 3.77 / 3.67 | 7875 / 7622 | 36.2 / 34.1 |
| A2 no budget input | SUB-300 | 8 | 0.3770 | 0.3687 [0.3167, 0.4228] | +0.0083 [-0.0083, +0.0267] | 5/292/3 | 1.0000 (Holm) | 0.3903 | 7.00 / 6.95 | 0.200 / 0.153 | 7.20 / 7.10 | 16508 / 16226 | 77.1 / 74.0 |
| A3 no adaptive stop | TEST-1704 | 4 | 0.3609 | 0.3632 [0.3410, 0.3852] | -0.0024 [-0.0044, -0.0008] | 0/1695/9 | 0.0273 (Holm) | 0.3845 | 3.71 / 4.00 | 0.110 / 0.000 | 3.82 / 4.20 | 7976 / 8790 | 36.5 / 39.6 |
| A3 no adaptive stop | SUB-300 | 8 | 0.3770 | 0.3870 [0.3341, 0.4417] | -0.0100 [-0.0204, -0.0021] | 0/294/6 | 0.1875 (Holm) | 0.4104 | 7.00 / 8.00 | 0.200 / 0.000 | 7.20 / 8.62 | 16508 / 19939 | 77.1 / 91.7 |
| A3 no adaptive stop (supplementary) | NEW-1654 | 4 | 0.3607 | 0.3632 [0.3405, 0.3862] | -0.0024 [-0.0046, -0.0008] | 0/1645/9 | 0.0039 (raw) | 0.3845 | 3.70 / 4.00 | 0.111 / 0.000 | 3.82 / 4.20 | 7976 / 8796 | 36.5 / 39.6 |
| A3 no adaptive stop (supplementary) | SUB-300 | 4 | 0.3437 | 0.3505 [0.2993, 0.4038] | -0.0069 [-0.0158, -0.0008] | 0/296/4 | 0.1250 (raw) | 0.3689 | 3.61 / 4.00 | 0.153 / 0.000 | 3.77 / 4.28 | 7875 / 8992 | 36.2 / 40.4 |

- Holm correction within the ablation family (m = 7: A1 B4; A2 B1/2/4/8; A3 B4 TEST-1704; A3 B8 SUB-300). Rows marked supplementary are outside the family and show the raw sign-test p.
- A2 removes the budget VALUE from the prompt; the fixed v2 system prompt still says the remaining budget is a maximum. A2 at B = 1/2/4 is an exact truncation of the B = 8 run (verified gold-free).
- A3 SLM calls include the STOP-suppression re-prompts (D6); A3 tokens and latency include them too.
- A2 at B = 1 and 2 differs from M5 on a single question each; reported as computed (frozen decision 4).
- SUB-300 resolves paired differences of roughly ≥ 0.04; smaller Δ are inconclusive at this n.
- ID Recall primary; Text Recall diagnostic only.

### T4 (continued): M5 vs A3 restricted to the questions where M5 stopped (descriptive, no test)

| Population | B | Questions where M5 stopped | Δ = M5 − A3 | W/T/L |
|---|---|---|---|---|
| TEST-1704 | 4 | 187 | -0.0214 | 0/178/9 |
| NEW-1654 | 4 | 184 | -0.0218 | 0/175/9 |
| SUB-300 | 8 | 60 | -0.0501 | 0/54/6 |

- On every other question A3 is identical to M5 (verified), so the full-population Δ comes only from these questions.

## T5

### T5a (TEST-1704): M5 stopping behavior, B = 4 (n = 1704)

| Measure | Value |
|---|---|
| STOP rate | 0.110 (187 of 1704) |
| STOP at step 0 (no expansion) | 83 |
| STOP when all gold already in evidence (post-hoc) | 74 of 187 |
| Expansions saved by STOP, per question (A3 mean − M5 mean) | 0.29 |
| Terminations | BUDGET_EXHAUSTED 1516, NO_CANDIDATES 1, SLM_STOP 187 |

- 'All gold already in evidence' uses gold labels after the fact; the controller never sees gold.

### T5b (TEST-1704): discovered gold by hop depth and wasted expansions, B = 4 maximum budget

| Method | Gold found at hop 1 | Gold found at hop 2+ | Total new gold | Wasted expansions (mean) | Expansions (mean) |
|---|---|---|---|---|---|
| M1 flat top-(3+b) | 107 | 100 | 207 | 3.74 | 4.00 |
| M2 SAGE one-shot (faithful) | 152 | 0 | 152 | 3.79 | 3.98 |
| M3 random (mean of 3 seeds) | 62.3 | 7.7 | 70 | 3.90 | 4.00 |
| M4 SAGE-score greedy | 135 | 40 | 175 | 3.77 | 4.00 |
| **M5 SLM (frozen, full)** | 190 | 55 | 245 | 3.42 | 3.71 |
| A3 no adaptive stop | 204 | 56 | 260 | 3.70 | 4.00 |

- Hop 0 = gold already in the seeds: 478 of 2242 gold passages (identical for every method).
- Hop 1 = discovered gold that is a direct graph neighbor of a seed; hop 2+ = discovered gold that is not (genuine multi-hop discovery for the traversal methods M3/M4/M5/A3).
- M1 does not use the graph: its 'hop 2+' means 'not a 1-hop neighbor of the seeds', including passages with no graph connection. M2 is one hop by construction.
- Wasted expansions = expansions taken after the last gold passage was found (all expansions if none was found).
- M3 counts are means over the three random seeds.

### T5c (TEST-1704): ID Recall by post-hoc category, B = 4 maximum budget

| Method | all_gold_in_seeds (n = 382) | reachable_top20 (n = 304) | reachable_filtered (n = 111) | unreachable_1hop (n = 907) |
|---|---|---|---|---|
| M0 seeds only | 1.0000 | 0.0474 | 0.0345 | 0.0113 |
| M1 flat top-(3+b) | 1.0000 | 0.2806 | 0.1096 | 0.0991 |
| M2 SAGE one-shot (faithful) | 1.0000 | 0.4282 | 0.0661 | 0.0113 |
| M3 random (mean of 3 seeds) | 1.0000 | 0.1852 | 0.0460 | 0.0196 |
| M4 SAGE-score greedy | 1.0000 | 0.3838 | 0.0390 | 0.0478 |
| **M5 SLM (frozen, full)** | 1.0000 | 0.5313 | 0.1006 | 0.0664 |
| A3 no adaptive stop | 1.0000 | 0.5445 | 0.1006 | 0.0664 |

- Categories (held-out definitions, D5), from the seeds' raw 1-hop neighbors and the step-1 top-20 pre-filter list including text-duplicate members: all gold in seeds / some remaining gold in the top-20 / reachable in one hop but filtered out of the top-20 / not reachable in one hop.
- Category membership is defined post hoc with gold and is identical for every method.

### T5a (NEW-1654): M5 stopping behavior, B = 4 (n = 1654)

| Measure | Value |
|---|---|
| STOP rate | 0.111 (184 of 1654) |
| STOP at step 0 (no expansion) | 81 |
| STOP when all gold already in evidence (post-hoc) | 73 of 184 |
| Expansions saved by STOP, per question (A3 mean − M5 mean) | 0.29 |
| Terminations | BUDGET_EXHAUSTED 1469, NO_CANDIDATES 1, SLM_STOP 184 |

- 'All gold already in evidence' uses gold labels after the fact; the controller never sees gold.

### T5b (NEW-1654): discovered gold by hop depth and wasted expansions, B = 4 maximum budget

| Method | Gold found at hop 1 | Gold found at hop 2+ | Total new gold | Wasted expansions (mean) | Expansions (mean) |
|---|---|---|---|---|---|
| M1 flat top-(3+b) | 107 | 96 | 203 | 3.74 | 4.00 |
| M2 SAGE one-shot (faithful) | 148 | 0 | 148 | 3.79 | 3.98 |
| M3 random (mean of 3 seeds) | 60.7 | 7.3 | 68 | 3.90 | 4.00 |
| M4 SAGE-score greedy | 132 | 37 | 169 | 3.77 | 4.00 |
| **M5 SLM (frozen, full)** | 186 | 52 | 238 | 3.42 | 3.70 |
| A3 no adaptive stop | 200 | 53 | 253 | 3.70 | 4.00 |

- Hop 0 = gold already in the seeds: 463 of 2181 gold passages (identical for every method).
- Hop 1 = discovered gold that is a direct graph neighbor of a seed; hop 2+ = discovered gold that is not (genuine multi-hop discovery for the traversal methods M3/M4/M5/A3).
- M1 does not use the graph: its 'hop 2+' means 'not a 1-hop neighbor of the seeds', including passages with no graph connection. M2 is one hop by construction.
- Wasted expansions = expansions taken after the last gold passage was found (all expansions if none was found).
- M3 counts are means over the three random seeds.

### T5c (NEW-1654): ID Recall by post-hoc category, B = 4 maximum budget

| Method | all_gold_in_seeds (n = 370) | reachable_top20 (n = 296) | reachable_filtered (n = 109) | unreachable_1hop (n = 879) |
|---|---|---|---|---|
| M0 seeds only | 1.0000 | 0.0486 | 0.0352 | 0.0111 |
| M1 flat top-(3+b) | 1.0000 | 0.2882 | 0.1116 | 0.0977 |
| M2 SAGE one-shot (faithful) | 1.0000 | 0.4296 | 0.0673 | 0.0111 |
| M3 random (mean of 3 seeds) | 1.0000 | 0.1862 | 0.0469 | 0.0193 |
| M4 SAGE-score greedy | 1.0000 | 0.3874 | 0.0398 | 0.0459 |
| **M5 SLM (frozen, full)** | 1.0000 | 0.5397 | 0.0933 | 0.0646 |
| A3 no adaptive stop | 1.0000 | 0.5533 | 0.0933 | 0.0646 |

- Categories (held-out definitions, D5), from the seeds' raw 1-hop neighbors and the step-1 top-20 pre-filter list including text-duplicate members: all gold in seeds / some remaining gold in the top-20 / reachable in one hop but filtered out of the top-20 / not reachable in one hop.
- Category membership is defined post hoc with gold and is identical for every method.

## T6

### T6: controller reliability, all SLM runs, by budget

| Run | Population | B | Questions | SLM calls | Invalid first attempts | Retries | CONTROLLER_ERROR | Error rate | Terminations |
|---|---|---|---|---|---|---|---|---|---|
| **M5 SLM (frozen, full)** | SUB-300 | 1 | 300 | 300 | 0 | 0 | 0 | 0.0000 | BUDGET_EXHAUSTED 275, SLM_STOP 25 |
| **M5 SLM (frozen, full)** | SUB-300 | 2 | 300 | 583 | 0 | 0 | 0 | 0.0000 | BUDGET_EXHAUSTED 281, SLM_STOP 19 |
| **M5 SLM (frozen, full)** | TEST-1704 | 4 | 1704 | 6505 | 0 | 0 | 0 | 0.0000 | BUDGET_EXHAUSTED 1516, NO_CANDIDATES 1, SLM_STOP 187 |
| **M5 SLM (frozen, full)** | NEW-1654 | 4 | 1654 | 6311 | 0 | 0 | 0 | 0.0000 | BUDGET_EXHAUSTED 1469, NO_CANDIDATES 1, SLM_STOP 184 |
| **M5 SLM (frozen, full)** | SUB-300 | 4 | 300 | 1130 | 0 | 0 | 0 | 0.0000 | BUDGET_EXHAUSTED 254, SLM_STOP 46 |
| **M5 SLM (frozen, full)** | SUB-300 | 8 | 300 | 2161 | 0 | 0 | 0 | 0.0000 | BUDGET_EXHAUSTED 240, SLM_STOP 60 |
| A1 no history | SUB-300 | 4 | 300 | 1143 | 0 | 0 | 0 | 0.0000 | BUDGET_EXHAUSTED 273, SLM_STOP 27 |
| A2 no budget input | SUB-300 | 8 | 300 | 2130 | 0 | 0 | 0 | 0.0000 | BUDGET_EXHAUSTED 254, SLM_STOP 46 |
| A3 no adaptive stop | TEST-1704 | 4 | 1704 | 7158 | 0 | 0 | 0 | 0.0000 | BUDGET_EXHAUSTED 1703, NO_CANDIDATES 1 |
| A3 no adaptive stop | SUB-300 | 8 | 300 | 2587 | 0 | 0 | 0 | 0.0000 | BUDGET_EXHAUSTED 300 |

- SLM calls include A3's STOP-suppression re-prompts (D6).
- A2 is shown for the B = 8 run only; its B = 1/2/4 rows are truncations of that same run, not separate runs.
- M5 B = 4 rows overlap (SUB-300 and NEW-1654 are subsets of TEST-1704); they are not additive.
- Invalid first attempt = the first reply at a step failed the parser; a retry is the single deterministic re-prompt; CONTROLLER_ERROR = the retry also failed.
