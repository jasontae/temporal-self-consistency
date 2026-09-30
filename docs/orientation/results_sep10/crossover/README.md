# (b) Item 2: where constant [UNKNOWN] starts beating the oracle

**Script:** `crossover_sweep.py`. **Outputs:** `crossover.png`/`.pdf`, `crossover_table.md`, `crossover.json`, `crossover.out`. **Inputs:** `data/prep/predictions_7b/tsct_test.jsonl`, `data/prep/gold_currency_audit.json`.

## Why this is exact, not simulated

Neither the oracle nor a constant policy consults which items are correct. Their ECE is therefore a function of per-class accuracy alone:

- ECE(oracle) = Σ_k w_k |a_k − c_k|
- ECE(constant c) = |ā − c|

The test set has two gold classes:

| class | n | share | level | observed accuracy |
|---|---|---|---|---|
| volatile ([TEMPORAL_HEDGE]) | 3,287 | w_v = 0.9075 | 0.45 | a_v = 0.0256 |
| slow ([COND_CONFIDENT]) | 335 | 0.0925 | 0.75 | a_s = 0.0448 |

## Results

1. **The crossover is at volatile accuracy a_v* = 0.314**, holding the slow class at its observed 0.045.
   - Closed form: a_v* = (0.10 + 0.45 w_v + 0.75 w_s − 2 w_s a_s) / (2 w_v). A numerical grid check gives 0.3138.
   - Below 31.4% volatile accuracy, constant [UNKNOWN] has lower ECE than the volatility-label oracle. The released test set sits at 2.6%, about twelve times below the crossover.
   - The threshold depends weakly on the slow class:

     | slow-class accuracy | crossover a_v* |
     |---|---|
     | 0 | 0.318 |
     | 0.25 | 0.293 |
     | 0.50 | 0.267 |
     | 0.75 | 0.242 |

2. **The oracle never beats the *best* constant in the vocabulary on this slice, at any volatile accuracy.** It beats every constant only in a small region: a_v between 0.435 and 0.60 and a_s at least 0.60. That region is 6.2% of the accuracy plane (middle panel), i.e. only when realized class accuracies sit close to the asserted levels 0.45 and 0.75. This is the precise form of the paper's "the metric rewards level-matching, not resolution".

3. **The 6.2x ratio has a 95% bootstrap CI of [5.85, 6.62]** (10,000 record resamples). The ECE gap has a CI of [0.375, 0.381].

4. **Age axis (right panel).** An answerer that knows the world at date Y is correct iff the key's value was valid at Y (Wikidata `t_start` ≤ Y < `t_end`; 3,312 of 3,622 records matched to the audit). Every key value in this test set ends in 2023 or 2024.
   - An ideal answerer (knowledge rate k = 1) makes the oracle win from 2012 through 2023. Constant [UNKNOWN] wins from 2024 onward, the year the last key values expire. So the crossover is the 2023→2024 expiry.
   - With k = 0.5 the oracle's window narrows to 2019–2022.
   - With k ≤ 0.2 constant [UNKNOWN] wins at every date.
   - The real models score 2.6% against this key, which is far below the curves for any k plotted. That is consistent with the benchmark being in the constant-wins regime for these models regardless of age. However, the models' training cutoffs are not pinned here, so this is not measured per model.
   - k is an illustrative knowledge rate, not a fitted parameter.

## Answer to the review question

Constant [UNKNOWN] beats the oracle whenever volatile-class accuracy falls below 0.31 (0.24–0.32 over any slow-class accuracy). The best constant beats the oracle everywhere except within about ±0.1 of the asserted levels. On this key, aging alone moves an ideal answerer across the boundary in 2023–2024, when the recorded values expire.
