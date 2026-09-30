# Item 1: the protocol on a second aged benchmark (FreshQA), evaluation only

**Why FreshQA:** see `../../SEP10_PLAN.md`, item 1. It is Apache-2.0, carries a never/slow/fast volatility label, and publishes dated answer keys.

**Data:** five snapshots in `data/prep/freshqa/` (FETCH_ALL step 5): 2024-02-26, 2024-10-07, 2025-02-03, 2025-08-11 and 2026-04-21.
- FreshQA **reassigns `id` between snapshots**: only 119 of 600 ids keep their question from 2024 to 2026. Snapshots are therefore joined on normalized question text.
- 413 questions survive: 149 false-premise and 38 unmatched questions were dropped (`build_report.json`).
- Of those, 146 are never-changing, 137 slow-changing and 130 fast-changing.
- The aged → current answer changed for 1 never-changing, 43 slow and 116 fast questions, which is plausible.

**Scripts:**
- `freshqa_protocol.py` (build / claims / analyze)
- `freshqa_pooled.py`
- `freshqa_threshold.py`
- runs in `../run_queue_d3.sh`: wall times in `../queue_d3.log`, 37–1,249 s per model

**Scoring:** correctness is containment of any accepted answer, with exact match as a sensitivity. There is no LLM judge. Containment undercounts when a key is a full sentence ("Leonardo DiCaprio does not have any children."). That undercount is the same for the oracle and the constant, so it does not change which one wins.

## 1. Calibration controls (the hedge-trained Qwen2.5-7B adapters; `freshqa_results.json`)

| adapter | key | accuracy (fast) | ECE model | ECE oracle | ECE constant [UNKNOWN] | oracle / constant [95% CI] | model resolution |
|---|---|---|---|---|---|---|---|
| TSCT seed 0 | aged 2024-02 | 0.177 (0.054) | 0.733 | 0.550 | 0.077 | 7.2× [4.5, 14.3] | 0.006 |
| TSCT seed 0 | current 2026-04 | 0.167 (0.038) | 0.743 | 0.559 | 0.067 | 8.3× [5.1, 17.9] | 0.006 |
| TSCT seed 1 | current | 0.162 (0.046) | 0.777 | 0.564 | 0.062 | 9.1× [5.4, 22.7] | 0.002 |
| SFT seed 0 | current | 0.160 (0.023) | 0.763 | 0.566 | 0.060 | 9.5× [5.7, 22.7] | 0.0001 |
| SFT seed 1 | current | 0.160 (0.038) | 0.767 | 0.566 | 0.060 | 9.5× [5.6, 23.0] | 0.003 |

- **The headline replicates on a second benchmark:** the input-independent constant beats the volatility-label oracle by 7–9.5×.
- **The trained hedge carries almost no resolution** (≤ 0.006).
- **Reference repair.** Moving from the aged to the current key lowers fast-fact accuracy (0.054 → 0.038 for TSCT seed 0) and widens the ratio (7.2× → 8.3×). This is the same direction as Table 2 on TemporalDelta.

## 2. The threshold curve (`threshold.png` / `.pdf`, `threshold_table.md`, `threshold.json`): the intended main figure

- **Panel A:** the exact TemporalDelta curve, with the crossover at volatile accuracy 0.314 and the test set at 0.026.
- **Panel B:** FreshQA. Every model's answers are re-scored against each dated key, plotting ECE(oracle) − ECE(constant); above 0 the constant wins.
  - **The three strongest models cross over as the key moves past their knowledge:**

    | model | fast accuracy, 2024 key → 2026 key | winner, 2024 → 2026 |
    |---|---|---|
    | Qwen3.5-27B | 0.19 → 0.09 | oracle → constant |
    | Qwen3.6-27B | 0.19 → 0.11 | oracle → constant |
    | Qwen3.6-35B | 0.24 → 0.10 | oracle → constant |

  - gemma-3-4B shows the same pattern: fast accuracy 0.32 against the 2024 key, 0.08 against 2026.
  - All weaker models and all four hedge adapters are in the constant-wins region at every key date.
- **Panel C:** every point against its own benchmark-specific crossover. FreshQA's class mixture differs from TemporalDelta's (35% never-changing, at 0.30–0.67 accuracy), so its crossover is point-specific (0.04–1.0). This panel is an identity, since the threshold is computed from the same class accuracies. It illustrates the rule rather than testing it.

**The oracle has real resolution on FreshQA** (0.03–0.05, AUROC about 0.75), because never-changing questions are answered better than fast-changing ones. Yet at the 2026 key the constant still wins on ECE for every model. The metric is blind to resolution even when resolution is present, which is a stronger form of the TemporalDelta result, where the oracle had essentially none.

**Two unreliable rows.** gpt-oss-20B and Qwen3-4B-Thinking reach only 6% and 14% on never-changing questions when forced to answer without reasoning (`--answer-only`). This is the same caveat as Table 3 (`../evergreen/`). Their points are shown but should not be interpreted.

## 3. No-model form screen (fast vs never-changing questions)

| no-model feature | AUROC |
|---|---|
| **temporal cue in the question** ("current", "latest", "now", …) | **0.755** |
| past-tense cue | 0.658 |
| character / word count, digit, starts-with-who | 0.51–0.53 |
| each model's claim log-probability (current answer) | 0.52–0.58 |

**The screen fires.** On FreshQA, a lexical cue in the question predicts the volatility label far better than any model's log-probability. A log-probability "temporal representation" claim on FreshQA would therefore not be established beyond surface form. This is the screen doing its job on a new benchmark.

## 4. Staleness and truth contrasts (log-probability; `fq_*.jsonl`, `freshqa_pooled.json`)

**Staleness** (current vs aged answer, changed questions, 159 pairs):
- Every one of the 9 models is below 0.5 (0.457–0.490).
- Pooled: **0.481 [0.456, 0.504], p = 0.12**; paired win rate 0.463 [0.414, 0.514].

**Truth manipulation check** (never-changing questions; the true answer against another question's true answer of the same coarse type, 145 pairs):
- Only 2 of 9 models reach 0.70 (0.647–0.712).
- These pairs are weaker than the office-holder design, because answer types (numbers, dates, free text) are only coarsely matched.

**Reading.** The familiarity direction replicates (9/9 below chance), but on FreshQA it is smaller and not significant at n = 159. This is a partial replication. The paper should report it as direction-consistent, not as an independent confirmation.

## What transfers

| component | on FreshQA |
|---|---|
| constant vs oracle (the headline), resolution, reference repair | **replicates** |
| form screen | **transfers; flags FreshQA's own surface cue** |
| staleness direction | **consistent but not significant** |
| four-arm entity-matched design | does not transfer (no founder-style counterparts) |
