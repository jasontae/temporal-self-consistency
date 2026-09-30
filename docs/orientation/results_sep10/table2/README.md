# Table 2: the repaired benchmark, rebuilt

**Script:** `rebuild_repaired.py`. **Outputs:** `table2_full.{md,json,out}`, plus the earlier dry run `table2_c5_only.*`.

**Inputs:**
- `data/prep/temporal_delta/temporal_delta_train.jsonl` (FETCH_ALL step 2; 5,356,184 bytes, matching the HF file)
- `gold_currency_audit.json`
- `{tsct,sft}_test.jsonl`

The accidental HF-cache copy is not used.

**Repair.** Gold becomes the live Wikidata current value (2,610 audit pairs have one), and every test question that also appears in train is dropped. 927 of 3,177 unique test questions are in train, which matches the ledger exactly. Existing predictions are rescored with the pipeline's `exact_match`; nothing is regenerated.

## Result

| policy (correctness from) | items | accuracy | ECE | Brier | reliability | resolution | AUROC | Sep 2 (acc, ECE, Brier, res) |
|---|---|---|---|---|---|---|---|---|
| cross-entropy baseline (SFT) | 1,849 | 0.49% | 0.4693 | 0.2305 | 0.2262 | 0.00056 | 0.687 | 0.46%, 0.4768, 0.231, 0.001 |
| trained model (TSCT) | 1,849 | 0.27% | 0.4715 | 0.2315 | 0.2288 | 0.000001 | 0.560 | 0.26%, 0.4789, 0.233, 0.001 |
| volatility-label oracle (TSCT correctness) | 1,849 | | 0.4713 | 0.2313 | 0.2286 | 0.000001 | 0.560 | —, 0.4788, 0.232, 0.001 |
| constant [UNKNOWN] | 1,849 | | 0.0973 | 0.0122 | 0.0095 | 0 | 0.500 | —, 0.0974, 0.012, 0.000 |

ECE binned equals ECE by level on every row, as the `../ece_check/` bound predicts.

**Oracle / constant ratio:**
- TSCT correctness: **4.84× [4.75, 4.96]**
- SFT correctness: 4.93× [4.81, 5.08]
- Sep 2 prints 4.9×, which lies inside both intervals.

## What reproduces and what does not

- **Reproduces:**
  - the accuracies (0.27% / 0.49%, against 0.26% / 0.46%)
  - the constant-[UNKNOWN] ECE (0.0973 against 0.0974)
  - the Brier column (0.2305 / 0.2315 / 0.2313 / 0.0122 against 0.231 / 0.233 / 0.232 / 0.012). Unlike Table 1's, it matches to rounding.
  - the ratio
  - the qualitative claims: the trained model sits on the oracle, the constant wins, and ECE worsens after repair
- **Does not reproduce: the item count.** Every filter variant tried gives 1,849 items (1,785 unique questions), not 1,953:

  | label set | train filter (question exact / normalized / question+answer / none) |
  |---|---|
  | any current label | 1,849 / 1,849 / 2,665 / 2,693 |
  | decided verdicts | 1,849 / 1,849 / 2,665 / 2,693 |
  | stale only | 1,821 / 1,821 / 2,628 / 2,652 |

  Counting unique (question, answer) pairs gives 1,795. The ledger's F-1 section gives 1,953 without the script, and the oracle ECE differs (0.4713 against 0.4788). That is consistent with a slightly different class mix.
  **Recommendation:** report 1,849 with this script and the values above, which differ from Sep 2 by at most 0.0075 ECE.
- **Resolution** is effectively zero for every policy here (SFT 0.0006; the rest ≤ 1e-6). The Sep 2 "0.001" is a rounded upper bound. The SFT arm's AUROC of 0.687 comes from its few correct answers carrying higher hedge levels.
