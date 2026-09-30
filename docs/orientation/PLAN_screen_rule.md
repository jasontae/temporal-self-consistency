# Plan: one decision rule for the form-validity screen, applied to all four sets

**Date:** 2026-09-30. Written and committed before any new number was computed. Results go in `docs/orientation/results_sep10/screen_rule/`. Any departure from this plan is listed under "Deviations" in that folder's README.

## Why

Section 4.3 and Appendices A, G and J use two different decision rules:

- **"The cue beats the model"** fires the screen on our first stress set (person-name regex 0.8333 vs best model 0.9000, ΔAUROC +0.067 [−0.083, 0.214]) and on FreshQA (temporal cue words 0.755 vs models 0.52–0.58).
- **"Holding the cue fixed removes only part of the signal"** passes EverGreenQA, even though its past-tense marker (0.665) beats every model there (pooled 0.57–0.61).

Under the first rule EverGreenQA would fire too. The fix is one rule, fixed now and applied identically to every set.

## The primary rule

The screen asks whether the model's discrimination survives conditioning on the strongest no-model feature.

1. **Cue-stratified AUROC of the model score.** Items are split into strata by the value of the strongest feature. A binary feature gives 2 strata. A continuous or count feature gives quintile strata, with cut points at the 20/40/60/80th percentiles of the feature over all analysed items of the set; duplicate edges merge. Cut points are computed once on the full sample and held fixed in every bootstrap draw. Positive–negative pairs are compared only within a stratum. The pooled value is (within-stratum wins + ½ ties) / (total within-stratum pairs). This equals the per-stratum AUROCs weighted by their number of within-stratum pairs.
2. **Confidence interval.** 95% percentile bootstrap, 2,000 draws, seed 20260930. Items are resampled within class, so class sizes are fixed. On the form-matched set the unit is the entity (cluster bootstrap): each draw resamples entities, and each entity brings both of its claims for the contrast.
3. **Verdict.** The screen **fires** when the cue-stratified CI includes 0.5.

### Amendments to the proposed rule, and why

I adopt the proposed rule with three amendments. Each fixes a concrete flaw.

- **(a) A precondition: there must be a signal to screen.** Without it, a set where the model has no signal at all "fires", and absence of signal gets blamed on form. The concrete case is the matched set's regime contrast: the model reaches 0.521 there, and the paper does not claim a regime signal. So the rule first checks the unconditional model AUROC. If its CI includes 0.5, the verdict is **no signal** (nothing to screen; the model claim already fails without the screen). Otherwise the verdict is **passes** or **fires** as above.
- **(b) Two-sided.** The matched set's claimed result is staleness below chance (0.415). "Signal" means the CI excludes 0.5 on either side. "Survives" means the stratified CI excludes 0.5 **on the same side** as the unconditional estimate. A stratified CI that excludes 0.5 on the opposite side also counts as fires, since the direction did not survive.
- **(c) Degenerate strata.** If the cue separates the classes perfectly on the full sample, there are no within-stratum pairs. The stratified AUROC is then undefined, and the screen **fires**: nothing in the model's discrimination can be told apart from the cue. Bootstrap draws with zero within-stratum pairs are dropped. If more than 5% of draws are dropped, the README says so.

### Set-level verdict

- **The set's verdict is the rule applied to the panel-mean AUROC.** This is the mean over panel models of the unconditional AUROC, and separately of the stratified AUROC, bootstrapped jointly: the same resampled items are used for every model in a draw.
- **Per-model verdicts are also reported**, as a count (e.g. "fires for 3 of 5 models").

## The secondary comparison (reported, not decisive)

- **The comparison.** Unconditional cue AUROC minus unconditional model AUROC, from a paired bootstrap on the same draws. It gives a 95% percentile CI and a two-sided p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)).
- **Orientation.** The cue is oriented in its better direction on the full sample. This is a post hoc choice between two directions and slightly favours the cue. The model score is oriented as fixed below and is never flipped.

## The feature suite (fixed now)

Seven features, all computed on the **screen text** of each item (defined per set below). No model is consulted.

| feature | definition |
|---|---|
| `n_chars` | character count |
| `n_words` | whitespace word count |
| `has_digit` | any digit, `\d` |
| `name_len` | the number of words in the answer person name, where the set gives it as a field (matched set: `person`). Otherwise, the number of words in the longest run of consecutive capitalised words `[A-Z][\w'\-.]*`, excluding the text's first word; 0 if none |
| `person_name` | `has_person_name` from `src/evaluation/b11_surface_control.py`: two or more capitalised tokens outside its stop list |
| `past_tense` | `PAST` from `src/evaluation/evergreen_surface_screen.py`: `\b(did\|was\|were\|had\|ended\|founded\|discovered\|invented\|born)\b`, case-insensitive. This is the regex behind the paper's 0.665. FreshQA's separate past-tense regex is not used |
| `temporal_cue` | `\b(current\|currently\|latest\|most recent\|now\|this year\|today\|recent)\b`, case-insensitive (the FreshQA list quoted in Section 4.3) |

- **Feature AUROC.** Direction-free: max(a, 1 − a) against the set's label.
- **The strongest feature** is the one with the highest feature AUROC on the analysed items of that set. Ties go to the first feature in the table order. A feature that is constant on the set cannot be the strongest.
- **Robustness.** The rule is also run with every non-constant feature as the conditioning feature, and the result is reported as a table.

## The four sets

The model panel is the Appendix A panel: Qwen3-4B-Instruct, Qwen2.5-7B, Qwen3.5-27B, Qwen3.6-27B and Qwen3.6-35B-A3B. gpt-oss-20B is excluded. Where a panel model was never scored on a set, it is omitted and the omission is stated; no model is run.

**Stress set** (first stress set, Appendix J)
- **Items:** the 22 distinct claim texts of `mixed_*.jsonl` (`b11_dedup.distinct_claims`).
- **Label:** positive = stable (`not is_volatile`).
- **Score:** `mean_logprob`; higher predicts stable.
- **Screen text:** the claim.
- **Unit:** the claim.
- **Panel on file:** Qwen3-4B-Instruct, Qwen2.5-7B, Qwen3.6-27B and Qwen3.6-35B. Qwen3.5-27B was never scored on this set.

**EverGreenQA** (Appendix A)
- **Items:** the balanced 200/200 English subset. Answer-only files `eg_ao_*.jsonl`; for Qwen2.5-7B, `eg_qwen25_7b.jsonl`, as in `table3_rebuild.py`.
- **Label:** positive = evergreen.
- **Score:** −perplexity; higher predicts evergreen.
- **Screen text:** the question.
- **Unit:** the question.

**FreshQA** (Section 4.3, item 1)
- **Items:** the matched subset only, so that cue and model are computed on the same items.
  - Fast-changing questions with an `fq_current` claim, i.e. the changed fast questions.
  - Never-changing questions with an `fq_true` claim.
- **Label:** positive = never-changing.
- **Score:** `pmi` of that claim, the score `freshqa_protocol.py` uses. Higher predicts never-changing. The fixed orientation replaces the protocol's direction-free max(a, 1 − a) for the model.
- **Screen text:** the question.
- **Unit:** the question.
- **Reference:** the paper's full-set cue AUROC (all fast vs all never questions, 0.755) is recomputed for comparison.

**Form-matched set** (Table 6)
- **Items:** `disjoint4b_*.jsonl`, 106 entities × 4 claims. Scores are pooled over all pairs, as in `screen_matched_set.py`.
- **Contrasts:**
  - **staleness** (positive = current office-holder, negative = expired): the paper's claimed result.
  - **truth** (true vs false founder): the manipulation check.
  - **regime** (founder vs volatile-role current holder): reported. The paper claims no regime signal.
- **Score:** `pmi`; higher predicts the positive arm.
- **Screen text:** the claim.
- **Unit:** the entity (cluster bootstrap).

## Expectations written before computing

- **Stress set:** fires, since within name-bearing claims the paper already reports chance.
- **FreshQA:** fires, or no signal.
- **Form-matched set:** passes on truth and staleness; no signal on regime.
- **EverGreenQA:** uncertain. The paper's form-stratified mean of 0.568 on 400 questions may or may not have a CI clear of 0.5. The strongest feature may also turn out not to be the past-tense marker.

If EverGreenQA fires, the paper has no published benchmark on which the screen returns a negative, and Section 4.3 must say so. No threshold, feature or model will be changed after seeing results.
