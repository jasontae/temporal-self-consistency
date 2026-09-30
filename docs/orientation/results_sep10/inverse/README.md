# (d) Item 7: the inverse truth-discrimination result

**Script:** `inverse_truth.py [--prefix matched_]`. **Outputs:** `inverse_matched.png`/`.pdf` (figure), `inverse_table_matched.md` (table, all numbers), `inverse_matched.json`, `inverse_matched.out`; `--with-e4b` and `--with-qwen38` variants alongside. **Inputs:** `data/prep/predictions_7b/matched_*.jsonl` (12 models: 9 admitted, 3 Gemma-4 excluded; 106 entities × 4 arms, `pmi`).

Rerun on a fixed-quantization ladder with `--prefix matched4b_` once item 4 has run.

## Result: the inverse relationship survives every control applied

**Primary set, per Edward 2026-09-28: 9 admitted models, with all Gemma-4 rows dropped** (`../MODELS.md` finding 1). Across the nine, regime AUROC falls as truth AUROC rises:

| statistic | 9 admitted (primary) | + gemma-4-E4B (appendix sensitivity; the Sep 2 set) | + Qwen3.8-27B (item 6) |
|---|---|---|---|
| OLS slope | **−0.961** | −0.889 | −0.903 |
| permutation p (exact) | **0.0082** | 0.0056 | 0.0052 |
| Pearson r | −0.836 (p = 0.005) | −0.823 (p = 0.003) | −0.825 (p = 0.003) |
| Spearman ρ | −0.717 (p = 0.030) | −0.673 (p = 0.033) | −0.794 (p = 0.006) |
| slope, 95% entity-bootstrap CI | [−1.57, −0.33] | [−1.45, −0.26] | [−1.44, −0.29] |
| **split-half control** (2,000 splits), share negative | **100%**, median −0.92 | 100%, median −0.85 | 100%, median −0.87 |
| **crossed random-effects model**, paired-win outcome: b1 (SE), p | **−1.20 (0.32), p = 0.007** (7 df) | −1.17 (0.28), p = 0.003 (8 df) | −1.14 (0.29), p = 0.005 (8 df) |
| same, PMI difference z-scored within model | −3.32 (0.88), p = 0.007 | −3.10 (0.79), p = 0.0045 | −3.21 (0.79), p = 0.0035 |
| family-clustered CR1 (3 clusters), p (t, 2 df), not relied on | 0.051 | 0.064 | 0.044 |
| staleness mean [95% entity-bootstrap CI] | **0.439 [0.399, 0.476]** | 0.445 [0.407, 0.480] | 0.439 [0.401, 0.476] |
| omnibus sign test, all admitted below 0.5 | 9/9, p = 0.0039 | 10/10, p = 0.0020 | 10/10, p = 0.0020 |

Files: `inverse_table_matched.md` / `inverse_matched.{png,pdf,json}` (primary), `*_with_e4b.*`, `*_with_qwen38.*`.

- **Random effects.** The model and family variance components go to about 0 once truth AUROC is in, so truth discrimination absorbs the between-model differences in regime separation. Entity variance is large, so the entity random effect requested in review matters for the SEs.
- **Split-half control.** It addresses a design-specific confound. Truth (immutable_true vs immutable_false) and regime (immutable_true vs volatile_current) share the immutable_true arm. Computing them on disjoint entities removes the shared items, and the slope keeps its sign and size.
- **Banding.** Dropping E4B breaks the Sep 2 banding sentence "all four admitted models with the lowest truth-discrimination scores show a significant regime effect". Among the nine, the four lowest are gemma-3-4B, Qwen3-4B-Thinking, the trained model and Qwen3-4B-Instruct, and the last is not significant (p = 0.77). The regression, permutation test and LMM replace the banding, as requested in review (A4).

## What changes relative to the Sep 2 text

| Sep 2 statement | Recomputed | Recommendation |
|---|---|---|
| β = −0.890, family-clustered SE = 0.196, p = 0.020 | slope −0.889 reproduces on the Sep 2 set of ten; −0.961 on the nine admitted now. CR1 family-clustered SE = 0.236, p = 0.064 (t, 2 df). The printed p equals a t-test with SE 0.196 on **3** df; 3 clusters give 2 df, which yields p = 0.045. | Drop the 3-cluster SE. Report the exact permutation p and the crossed random-effects model, as requested in review. |
| Spearman ρ = −0.673, p = 0.033 | reproduces | keep |
| staleness mean 0.444, 95% CI [0.427, 0.463] | mean 0.445. A t-interval across the 10 model values gives [0.424, 0.465], close to the printed one, but it ignores that all models share the same 106 entities. The entity-bootstrap CI is [0.407, 0.480] on the Sep 2 ten and **[0.399, 0.476] (mean 0.439) on the nine admitted now**. | Use the entity-bootstrap CI. It still excludes 0.5. |
| Figure 3 axis: "regime AUROC (… paired within entity)" | The plotted AUROC is the unpaired Mann–Whitney AUROC over pooled scores (`analyze_matched_regime.py`). Only the sign test is paired. The paired win rate is in the table and can differ (Qwen3-4B-Instruct: AUROC 0.568, paired 0.519). | Fix the axis label, or plot the paired win rate. |
| "four lowest significant, four highest not" banding | holds on the Sep 2 ten; **fails on the nine** (Qwen3-4B-Instruct, fourth lowest, p = 0.77) | Review asked to drop the banding (A4). The regression, permutation and LMM make it unnecessary. |

## Caveats

- **Mixed quantization.** The nine rows are 4-bit, 8-bit, 6.5-bit and nvfp4. Qwen dominates (7 of 9), so leave-one-family-out without Qwen leaves 3 points and is uninformative. Item 4 (fixed 4-bit ladder) is the missing control.
- **Derangement build.** Every false founder here is another entity's true founder (the derangement build; see `../../SEP10_PLAN.md`). The disjoint-pool rescore (A6) is in `../a6_disjoint/`.
