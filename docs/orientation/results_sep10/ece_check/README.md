# (a) Is Table 1's ECE "exact on the four discrete levels without binning"?

**Script:** `ece_levels.py`. **Output:** `ece_levels.out`, `ece_levels.json`. **Inputs:** `data/prep/predictions_7b/{tsct,sft,base}_test.jsonl` (tracked in git).

## Result

**The ECE numbers are correct, but the caption misdescribes the code.** Table 1 was produced by `compute_ece` in `src/evaluation/eval_pipeline.py:34`, which uses 10 equal-frequency bins over sorted confidences. It does not group by level. On this data the binned and per-level values are identical to six decimals for every Table 1 row:

| policy (correctness from) | binned (pipeline) | by level | Sep 2 Table 1 |
|---|---|---|---|
| oracle (tsct) | 0.4504 | 0.4504 | 0.4504 |
| trained model (tsct) | 0.4506 | 0.4506 | 0.4506 |
| cross-entropy baseline (sft) | 0.4552 | 0.4552 | 0.4552 |
| constant [TEMPORAL_HEDGE] (tsct) | 0.4227 | 0.4227 | 0.4227 |
| constant [UNKNOWN] (tsct) | 0.0727 | 0.0727 | 0.0727 |

**Why they agree.** Every one of the 10 bins has accuracy at or below its mean confidence, and every level does too (checked, not assumed; column `all_bins_overconfident`). Under that condition both estimators equal mean(confidence) − mean(accuracy). This is a property of this data, not of the code: on the untuned base model's correctness, constant [UNKNOWN] gives 0.0629 binned against 0.0608 by level.

**Table 2 is also unaffected.** The repaired set could not be rebuilt here, because it needs the train split, which isn't on disk. The bound argument still holds: at most 9 of 1,953 items are correct (accuracy ≤ 0.46%), and a bin holds 195 items. So no bin can exceed 9/195 = 0.046, which is below the lowest level (0.10), and binned equals per-level.

**Does Table 1 change?** No ECE value changes. The caption should say either "computed with 10 equal-frequency bins; identical to the per-level value on this data because every bin is overconfident", or switch the code to per-level grouping, which gives the same numbers.

## The Brier and resolution columns do not reproduce

The Sep 2 Table 1 adds Brier and resolution columns. No Brier or Murphy code exists in the repo, its git history, or the upstream snapshot. Recomputed exactly (grouped by level, correctness as scored):

| policy | Brier (recomputed) | Brier (Sep 2) | resolution (recomputed) | resolution (Sep 2) |
|---|---|---|---|---|
| oracle | 0.2360 | 0.218 | 0.000031 | 0.002 |
| trained model | 0.2363 | 0.219 | 0.000031 | 0.001 |
| cross-entropy baseline | 0.2368 | 0.224 | 0.000000 | 0.001 |
| constant [TEMPORAL_HEDGE] | 0.2052 | **0.187** | 0 | 0.000 |
| constant [UNKNOWN] | 0.0319 | 0.031 | 0 | 0.000 |

The printed 0.187 is impossible for binary outcomes. A constant forecast of 0.45 has Brier 0.2025 + 0.1·accuracy, which is at least 0.2025. Using token F1 instead of exact match does not reproduce it either: 0.1967 and 0.2261. The qualitative claim survives, since both constant policies have zero resolution and the trained model's resolution is about 3e-5. **The column's values need replacing** with the recomputed ones, or Jason needs to supply the script that made them.
