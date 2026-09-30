# Validation-estimated constant and confidence mappings (directive 3, worker B)

**Script:** `validation_constant.py`. **Outputs:** `validation_constant.{md,json}`. **Inputs:** `data/prep/predictions_7b/{tsct,tsctS1,sftS1,base}_test_lp.jsonl` (tracked).

**Question** (fused critique (b)12, reviewer attack 9): is "the constant beats the oracle" an artifact of setting the constant to the test accuracy, or of the fixed 0.95/0.75/0.45/0.10 mapping?

There is no validation split on disk, since train and val are in FETCH_ALL step 2. The deployable constant is therefore estimated two ways, neither of which sees a scored record's label:
- **cross-fitted:** 5-fold within the test set;
- **cross-run:** the other seed's realized accuracy.

## Result: no, on either count

**TSCT seed 0** (n = 3,622, accuracy 0.0273). The other three runs agree; see `validation_constant.md`.

| policy | ECE | Brier | resolution | AUROC |
|---|---|---|---|---|
| fixed mapping, emitted hedge (TSCT) | 0.4506 | 0.2363 | 0.00003 | 0.530 |
| volatility-label oracle, fixed mapping | 0.4504 | 0.2360 | 0.00003 | 0.530 |
| validation-fit mapping, emitted hedge | 0.0087 | 0.0266 | 0.00000 | 0.513 |
| oracle, validation-fit mapping | 0.0087 | 0.0266 | 0.00000 | 0.513 |
| **constant, cross-fitted base rate** | 0.0107 | 0.0266 | 0 | 0.50 |
| **constant, cross-run base rate** (seed 1's 0.0210) | 0.0160 | 0.0266 | 0 | 0.50 |
| constant [UNKNOWN] = 0.10 (fixed level, Table 1) | 0.0727 | 0.0319 | 0 | 0.50 |
| constant at test accuracy (oracle reference) | 0.0173 | 0.0266 | 0 | 0.50 |
| continuous: log-prob, isotonic, cross-fitted | 0.0086 | 0.0253 | 0.00132 | 0.824 |

1. **Deployable constants beat the fixed-mapping oracle by far more than Table 1's constant does.** The cross-fitted constant reaches 0.0107 against 0.4504, about 42× lower. The cross-run constant (0.0160) is estimated from a different run entirely. Table 1's [UNKNOWN] = 0.10 is itself a fixed vocabulary level, chosen without test labels, so it is a legitimate baseline too. **The 6.2× is conservative, not an oracle artifact.**
2. **Re-fitting the hedge mapping on held-out folds turns both TSCT and the oracle into the constant.** Resolution drops to 0.0000, and the Brier difference from the cross-fitted constant has a 95% CI containing 0 in all four runs: TSCT seed 0 −0.00002 [−0.00009, +0.00005]. With the levels re-fitted, the hedge channel carries no correctness information beyond the base rate, and that holds for the volatility-label oracle as well.
3. **Continuous confidence is the only informative score.** AUROC is 0.81–0.87, and Brier is lower than the cross-fitted constant with a CI that excludes 0 in all four runs:
   - TSCT seed 0: −0.0013 [−0.0020, −0.0006]
   - TSCT seed 1: −0.0006 [−0.0011, −0.0001]
   - SFT seed 1: −0.0012 [−0.0019, −0.0005]
   - base: −0.0059 [−0.0087, −0.0033]

   Its ECE (0.0086) is indistinguishable from the constants'.
4. **Estimator note.** The perfectly calibrated test-accuracy constant reads ECE 0.0173, not 0. The pipeline's equal-frequency binning cuts tied confidences in file order, so a constant's ECE measures accuracy variation across file-order chunks. ECE differences below about 0.01 between these policies are therefore noise. That is one more reason the paper's comparisons should rest on Brier and resolution.
5. **The base model's fixed mapping** (ECE 0.91) is not a hedge signal at all. The untrained base model emits no hedge token on any of the 3,622 records, so all 3,622 get the fallback `[CONFIDENT]` = 0.95 (`hedge_from_fallback` true on every record). It is listed for completeness only.

**For the paper (§5.1):** one sentence saying that the constant is not tuned to test accuracy, plus the cross-fitted row in Table 1 or its caption. The seed-1 run was the one where Brier had touched zero in the vocabulary sweep; here, with cross-fitting and 2,000 resamples, its CI excludes zero narrowly ([−0.0011, −0.0001]).
