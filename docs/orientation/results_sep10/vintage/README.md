# Item 6: a third vintage point (Qwen3.8-27B)

**Scripts:**
- `vintage_ladder.py` (analysis)
- `../run_local_scoring.sh` (scoring; wall-clock times in `../local_scoring.log`)

**Outputs:** `vintage_table_matched.md`, `vintage_matched.{json,png,pdf}`, `vintage_matched.out`.

**Input:** `data/prep/predictions_7b/matched_qwen38_27b.jsonl`. It was scored 2026-09-28 in 153 s wall, through the same harness as the Aug 13 rows, on the original claim set. No BOS is involved for Qwen.

**Audit.** Qwen3.8-27B-4bit is clean (`../MODELS.md`): NLL 2.07, known-answer 0.94 / 0.92, and it loads through mlx_lm. Its truth AUROC is 0.844, above the 0.70 bar, so it would be admitted.

## Result

| model | truth AUROC | regime AUROC | staleness AUROC [95% CI] | staleness sign p |
|---|---|---|---|---|
| Qwen3.5-27B | 0.845 | 0.521 | 0.415 [0.362, 0.467] | 0.015 |
| Qwen3.6-27B | 0.855 | 0.510 | 0.442 [0.388, 0.493] | 0.015 |
| Qwen3.8-27B | 0.844 | 0.555 | 0.447 [0.397, 0.499] | 0.63 |

All three differences are paired over the same 106 entities (4,000 bootstrap resamples):

| staleness difference | Δ AUROC | 95% CI | p |
|---|---|---|---|
| 3.6 − 3.5 | +0.027 | [−0.002, +0.056] | 0.06 |
| 3.8 − 3.6 | +0.005 | [−0.038, +0.051] | 0.81 |
| 3.8 − 3.5 | +0.032 | [−0.009, +0.075] | 0.13 |

## Reading

- **The staleness point estimate stays below chance for the third vintage (0.447).** Adding Qwen3.8 does not break "every admitted model below chance".
- **Its per-model sign test is not significant, though (p = 0.63).** The CI upper bound is 0.499.
- **No vintage step is distinguishable from zero.** The ladder is flat within noise (0.415 → 0.442 → 0.447), so it cannot localize the effect to pretraining versus post-training in either direction. The Sep 2 sentence "the small change after re-alignment is consistent with a pretraining rather than post-training locus" is not supported by the three-point ladder beyond "no detectable change".
- **Truth discrimination is flat** (0.845 / 0.855 / 0.844), so the vintages differ in release date but not in the manipulation check.
- **Regime AUROC rises slightly for 3.8** (+0.046 over 3.6, p = 0.07). That is also inside noise.
- **Caveat:** converter versions differ (mlx-vlm 0.3.12 / 0.4.4 / 0.6.8), though the recipe (4-bit affine g64) is the same.
- **The gemma-3-4B / gemma-4-E4B pair is withdrawn** with the Gemma-4 drop.

The disjoint-pool (A6) version of this table is `vintage_table_disjoint.md` once A6 finishes.
