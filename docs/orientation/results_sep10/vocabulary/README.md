# (c) Item 3: does the failure disappear with a 0.01 level or continuous confidence?

**Script:** `vocab_sweep.py`. **Outputs:** `vocab_table.md` (all numbers), `vocab.png`/`.pdf`, `vocab.json`, `vocab.out`. **Inputs:** `data/prep/predictions_7b/{tsct,tsctS1,sftS1,base}_test_lp.jsonl` (per-record `correct` and the answer's `mean_logprob`).

"The failure" means a policy with no item-level information gets better ECE than the volatility-label oracle.

## Answer: no. A finer vocabulary makes it worse, and continuous confidence makes ECE blind to it.

**1. A 0.01 level (exact, floor sweep).** The oracle's ECE does not involve the floor. The best constant's ECE is |accuracy − floor|.

| lowest level | constant ECE, released key | oracle / constant | constant ECE, repaired key | oracle / constant |
|---|---|---|---|---|
| 0.10 (current) | 0.0727 | 6.2x | 0.0974 | 4.9x |
| 0.05 | 0.0227 | 19.9x | 0.0474 | 10.1x |
| 0.01 | 0.0173 | 26.0x | 0.0074 | 64.7x |

- With a 0.01 option the constant wins by 26x instead of 6.2x.
- The repaired key's "confidence-support mismatch" (accuracy 0.26% below the 0.10 floor) shrinks from 0.097 to 0.007, but the constant's advantage grows.
- The repaired-key row uses the ledger's accuracy and oracle ECE (`specs/CLAIMS-LEDGER.md`, F-1) because the repaired set is not rebuilt here.

**2. A free vocabulary (each gold class asserts its own realized accuracy, 5-fold cross-fitted).** Both the refit oracle and a constant at the base rate reach an ECE of about 0.01 (TSCT seed 0: 0.0087 vs 0.0107). Both have resolution 0.0000 and AUROC about 0.5. The oracle has no item-level information on this key, so freeing its levels only lets it tie the constant.

**3. Continuous confidence (the answer's own mean log-probability, recalibrated by isotonic or Platt maps, 5-fold cross-fitted).** This score carries real information: correctness AUROC is 0.81–0.88 across all four runs, against 0.53 for the emitted hedge. Yet:

| TSCT seed 0 | ECE (pipeline) | Brier | resolution | AUROC |
|---|---|---|---|---|
| log-prob, isotonic | 0.0086 | 0.0253 | 0.00132 | 0.824 |
| log-prob, Platt | 0.0064 | 0.0254 | 0.00104 | 0.840 |
| constant at base rate | 0.0107 | 0.0266 | 0 | 0.50 |
| raw exp(mean log-prob) | 0.5345 | 0.3354 | 0.00143 | 0.841 |

- **ECE cannot separate the informative score from the constant.** The two sit within about 0.004 of each other, which is inside the estimator's own binning noise: the base-rate constant, perfectly calibrated by construction, still reads 0.0107.
- **Resolution and Brier do separate them.** Brier(isotonic) − Brier(constant) has a 95% CI that excludes zero:
  - TSCT seed 0: [−0.0020, −0.0006]
  - SFT seed 1: [−0.0019, −0.0005]
  - base model: [−0.0086, −0.0034]
  - TSCT seed 1 touches zero: [−0.0011, −0.0000]
- The raw, uncalibrated log-probability has the worst ECE of any policy (0.53) and the best AUROC. ECE would rank it last.

## Implication for the paper

The confidence vocabulary does not cause the failure. It is a property of ECE at a low base rate: any vocabulary that contains a level near the base rate lets a constant win, and a continuous one lets a constant tie the best informative score. So the protocol's requirement to report resolution (or AUROC) and the constant control next to every ECE is necessary, not cosmetic. This answers the review question directly, and it also supports the paper's Murphy decomposition once that column is corrected (see `../ece_check/`).
