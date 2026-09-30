# A6: false founders from a disjoint pool

## Why

In the Aug 13 build every false founder was a derangement of the true founders. Each founder name therefore appeared twice (once true, once false), and on the regime contrast person-name frequency separated the arms perfectly (1.000). The Sep 2 Appendix C describes a disjoint-pool resample, but no such build was ever scored.

## Build (`build_disjoint.py` → `data/stress_tests/matched_regime_claims_disjoint.jsonl`, report in `build_report.json`)

- The volatile_current, volatile_stale and immutable_true arms are kept exactly.
- immutable_false is replaced by a human founder (Wikidata P112, P31 = Q5) of some *other* organization. Pool: `data/prep/wikidata_a6/human_founders.json`, a SPARQL fetch on 2026-09-28 (19,951 names after filters).
- Each false founder is matched to its entity's true founder on log(1 + sitelinks), greedy without replacement in a seeded order.
- **Checks:**
  - 424 claims, 106 entities
  - no person appears twice
  - no false founder is a true founder elsewhere
  - median sitelinks 10 vs 10 (KS p = 0.13)
  - mean |Δ log(1 + sitelinks)| = 0.14
- The first build fell back to a nominal prominence for two founders (López Obrador, Navalny) whose labels had not resolved. After they were resolved it was rebuilt, and the first partial scoring was discarded and rerun (`../local_scoring.log`).

## Scoring (`../run_local_scoring.sh`, `../local_scoring.log`)

- Same harness as Aug 13. gpt-oss uses `--bos off`, which reproduces its Aug 13 scores (`../MODELS.md`).
- Wall time 33–283 s per model.
- **Harness check** (`check_unchanged_arms.py` → `unchanged_arms.json`): the three unchanged arms reproduce the Aug 13 scores exactly (max |Δ| = 0) for 10 of 11 models. Qwen3.6-35B-A3B (MoE, nvfp4) drifts by up to 0.033 nats per claim, which moves its AUROCs by at most 0.001.
- **Qwen2.5-7B and the trained model** were rescored once the base checkpoint was restored (FETCH_ALL step 1; `../item4/item4.log`, 49 s and 58 s). Both audit clean (`../model_audit/runs/qwen25_7b.json`, `tsct.json`), and both reproduce their unchanged arms exactly.

## Results

**Truth AUROC before and after** (`truth_before_after.py` → `truth_before_after.md`):

| model | derangement (Aug 13) | disjoint pool | change [95% CI] | ≥ 0.70 |
|---|---|---|---|---|
| gemma-3-4B | 0.719 | 0.770 | +0.052 [−0.005, +0.109] | yes |
| Qwen3-4B-Thinking | 0.746 | 0.777 | +0.030 [−0.011, +0.071] | yes |
| Qwen3-4B-Instruct | 0.771 | 0.797 | +0.026 [−0.016, +0.069] | yes |
| gpt-oss-20B | 0.806 | 0.799 | −0.007 [−0.050, +0.037] | yes |
| Qwen3.6-35B-A3B | 0.844 | 0.847 | +0.002 | yes |
| Qwen3.5-27B | 0.845 | 0.852 | +0.007 | yes |
| Qwen3.6-27B | 0.855 | 0.856 | +0.001 | yes |
| Qwen3.8-27B (item 6) | 0.844 | 0.853 | +0.009 | yes |
| Qwen2.5-7B | 0.794 | 0.818 | +0.024 [−0.025, +0.069] | yes |
| trained model | 0.768 | 0.794 | +0.026 [−0.016, +0.071] | yes |
| gemma-4-E4B (dropped) | 0.738 | 0.750 | +0.012 | yes |
| gemma-4-26B-A4B (dropped) | 0.584 | 0.530 | −0.054 | no |
| gemma-4-31B (dropped) | 0.669 | 0.590 | −0.079 | no |

- The weakest models gain the most (+0.03 to +0.05), while the strong models are unchanged, although no single change excludes zero.
- Founder-name reuse was, if anything, *deflating* truth discrimination for weak models, not inflating the x-axis of the inverse result.
- Admission is unchanged: every non-Gemma-4 model still clears 0.70, and 26B and 31B still fail.

**The inverse result (item 7) on the disjoint build** (`../inverse/inverse_*disjoint*.out`):

| | derangement, same 7 models | disjoint, 7 models | disjoint + Qwen3.8 (8) |
|---|---|---|---|
| OLS slope | −0.884 | **−1.307** | −1.134 |
| Spearman ρ | −0.821 | −0.821 | −0.786 |
| permutation p | 0.0036 (exact) | 0.0032 (exact) | 0.0016 (exact) |
| split-half, share of splits negative | 100% | 100% | 99.9% |
| crossed random-effects model, paired win: b1, p | −1.15, p = 0.011 (5 df) | −1.60, p = 0.021 (5 df) | −1.42, p = 0.012 (6 df) |

The relationship survives the rebuild and steepens, because truth scores compress upward for the weak models.

**All nine admitted models** (`../inverse/inverse_disjoint.out`), derangement build → disjoint build:

| statistic | derangement | disjoint |
|---|---|---|
| slope | −0.961 | **−1.367** |
| exact permutation p | 0.0082 | **0.013** |
| Spearman ρ | −0.717 (p = 0.030) | −0.667 (p = 0.050) |
| split-half, share negative | 100% | 99.9% |
| crossed random-effects model, paired win: b1, p | −1.20, p = 0.007 | −1.63, p = 0.007 |
| staleness mean [95% CI] | 0.439 [0.399, 0.476] | unchanged (same arms) |

The rank correlation weakens slightly (p = 0.050), while the slope, the permutation test and the LMM hold. The disjoint build should be the primary one.

**Appendix C screen on the disjoint build** (`src/evaluation/screen_matched_set.py --src …disjoint_qwen35_27b.jsonl` → `screen_disjoint_qwen35.out`; same model on the Aug build in `screen_matched_qwen35.out`):
- person-name frequency: 1.000 on regime → **0.500 on every contrast**
- every surface feature on the truth contrast sits in 0.505–0.508
- name length on regime is still 0.574 [0.52, 0.63]

The set passes.

**Sep 2 Appendix C inconsistency.** The text gives name length on regime as 0.574 [0.52, 0.63], but Table 5 prints 0.524. The data gives 0.574. Table 5's "model log-probability" row should also name its model: it is Qwen2.5-7B (0.634 / 0.794 / 0.426).

**Staleness is unaffected by A6:** the arms involved are identical in both builds.
