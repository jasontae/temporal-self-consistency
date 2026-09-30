# Why repairing the answer key makes every ECE worse (directive 3, worker B)

**Script:** `repair_check.py` → `repair_check.json`. **Inputs:** `data/prep/predictions_7b/{tsct,sft}_test.jsonl` (tracked) and the printed Sep 2 Table 2. Rebuilding the repaired set needs FETCH_ALL step 2.

## Mechanism

When every bin is overconfident, which holds on both keys (`../ece_check/`), the ECE of *any* policy reduces to

> **ECE = m − a**, where m is the policy's mean asserted confidence and a is accuracy.

Which items are correct drops out entirely. Three things follow:

1. **The repair lowers a**, from 0.0273 to 0.0026 for the trained arm, because the models mostly do not know the current values. That adds the accuracy drop to every policy's ECE, so every arm "gets worse" by construction. This is Sep 2's L179–185 ("pushes the base rate further from every confidence our vocabulary can express") made exact.
2. **The oracle/constant ratio** is (m_oracle − a) / (0.10 − a). As a falls, the numerator and denominator grow by the same amount, so the ratio falls toward its a → 0 limit, m_oracle / 0.10. That is why the ratio moves from 6.2× to 4.9×: the constant advantage does not weaken, and both numbers approach accuracy-free constants.
3. **"The trained model attains the oracle outcome"** (L151–152, L176–177) means only that TSCT's mean asserted confidence (0.4780) equals the oracle's (0.4777). It says nothing about which items TSCT hedges.

## Checks

**Released key: the identity holds exactly for every policy.**

| policy | accuracy | mean confidence m | m − a | ECE (pipeline) |
|---|---|---|---|---|
| volatility-label oracle | 0.0273 | 0.4777 | 0.4504 | 0.4504 |
| TSCT seed 0 | 0.0273 | 0.4780 | 0.4506 | 0.4506 |
| SFT seed 0 | 0.0226 | 0.4778 | 0.4552 | 0.4552 |
| constant [TEMPORAL_HEDGE] | 0.0273 | 0.4500 | 0.4227 | 0.4227 |
| constant [UNKNOWN] | 0.0273 | 0.1000 | 0.0727 | 0.0727 |

**Repaired key (printed Sep 2 Table 2):**
- **Constant [UNKNOWN]:** 0.10 − 0.0026 = 0.0974, which matches the printed 0.0974 exactly.
- **Oracle:** 0.4788 implies a mean oracle confidence of 0.4814 on the 1,953 items. That corresponds to a 10.5% [COND_CONFIDENT] share, against 9.25% on the full test set, which is plausible after dropping the train-overlapping questions. It will be checked exactly when the set is rebuilt (Table 2, FETCH_ALL step 2).
- **Ratio:** printed 0.4788 / 0.0974 = 4.92. The limit m_oracle / 0.10 on the released mix is 4.78, and the released-key value (0.4777 − 0.0273) / (0.10 − 0.0273) is 6.20.

**Direction of the fused critique's candidate mechanism.** The V/R candidate, "higher volatile accuracy penalizes the constant", has the direction backwards for this case. The repair *lowers* accuracy. The constant is penalized because accuracy moves further *below* the 0.10 floor, and every other policy is penalized by the same amount.

## For the paper (§5.2)

Replace L182–185 with two sentences: "When every confidence bin is overconfident, ECE equals mean asserted confidence minus accuracy for any policy, regardless of which items are correct. The repair lowers accuracy from 2.7% to 0.26%, which adds the same amount to every policy's ECE and drives the oracle-to-constant ratio toward its accuracy-free limit (4.8)."

## Update: Table 2 rebuilt by worker A (`altj-sep10-revisions` @ `f764b5b`, `results_sep10/table2/table2_full.md`)

A rebuilt the repaired set from the train split. It has **1,849 items**; the Sep 2 count of 1,953 does not reproduce. The identity holds on the rebuilt numbers:
- **Constant [UNKNOWN]:** 0.10 − 0.0027 = 0.0973, matching A's 0.0973 exactly.
- **Oracle:** 0.4713 with accuracy 0.0027 implies a mean asserted confidence of 0.4740, i.e. an 8.0% [COND_CONFIDENT] share (against 9.25% on the full test set). A's binned and by-level ECE are identical (0.4713), as the identity requires.
- **Ratio:** 0.4713 / 0.0973 = 4.84, which is A's 4.84× [4.75, 4.96]. The limit m / 0.10 on this set is 4.74.

The earlier inference from the Sep 2 numbers (a 10.5% [COND_CONFIDENT] share on 1,953 items) is superseded.
