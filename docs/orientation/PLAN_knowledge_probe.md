# Plan: knowledge-conditioned staleness probe (worker E, 2026-09-30)

Written and committed before any generation is run. Inference only, on existing checkpoints; no training.

## Why

The paper's lead model-side result is that all nine admitted models give the expired office-holder a higher
name-controlled log-probability than the current one (staleness AUROC 0.442 [0.404, 0.477] at 4-bit). The ARR
review panel (`ARR_REVIEW_PANEL.md` §1 item 9) flagged the first-order confound and it was never tested: many
"current" holders took office in 2025-2026 (for example Nestlé, BP), after several models' training cutoffs. A
model that has never seen the current holder *should* find the expired one more familiar. That is outdated
knowledge (the effect Jiang et al. 2026 and Hossain et al. 2026 already describe), not a preference that survives
knowledge. The time-since-change strata (`results_sep10/a5_prominence/`) do not settle it: they use calendar time,
not what each model knows.

## Hypotheses

- **H-gap:** the preference is carried by entity-model pairs where the model does not know the current holder.
  Among pairs where the model names the current holder, the current holder is preferred (paired win rate >= 0.5).
- **H-persist:** the preference survives knowledge. Among pairs where the model names the current holder when
  asked, the expired holder still gets the higher PMI (paired win rate < 0.5).

## Data

- Claims: `data/stress_tests/matched_regime_claims_disjoint.jsonl`, 106 entities; only the current and expired
  office-holder arms are used (identical in the original and disjoint builds).
- PMI scores: the existing fixed 4-bit files `data/prep/predictions_7b/disjoint4b_<tag>.jsonl` for the nine
  admitted models (the Figure 4 inputs). No rescoring.
- Knowledge labels (new, generated here): for each of the nine admitted checkpoints (same paths and flags as
  `results_sep10/date_probe/run_date_probe.sh`), greedy decoding, max 24 new tokens:
  - **Primary, QA probe:** chat prompt "Who is the current {role} of {entity}? Answer with only the name, no
    explanation." Reasoning checkpoints are started on the answer (the existing `answer_prompt_ids`:
    `enable_thinking=False`; gpt-oss steered into its final channel; for Qwen3-4B-Thinking an empty think block is
    prefilled). The TSCT adapter stops at any hedge token.
  - **Secondary, frame completion:** raw text "The {role} of {entity} is", the scoring frame itself.
- Answer labels: `current` / `stale` / `other` by a deterministic name matcher (accent- and case-folded; the gold
  surname must appear as a token, plus the given name when the two holders share a surname). Every `current` and
  `stale` match is listed for a manual audit, plus a random 50 of the `other` rows; audit corrections are recorded
  in the output, not silently applied.

## Metric

- Paired win rate 1[PMI(current) > PMI(expired)] pooled over (model, entity) pairs within each knowledge group,
  with a 4,000-draw entity-cluster bootstrap (resample entities, keep every model's row), as in `a5_prominence`.
- Secondary: mean within-model z-scored PMI difference per group; per-model counts of each group.

## Decision rule

- If the `current` group has fewer than 30 (model, entity) pairs: **inconclusive**; report counts only.
- If the `current` group's win-rate CI lies at or above 0.5, or includes 0.5 with point estimate >= 0.5:
  **H-gap**. Recommend the paper say the preference mostly reflects what the models do not know (or know in an
  outdated form), and drop "persists" language that implies a bias beyond knowledge.
- If the `current` group's win-rate CI lies entirely below 0.5: **H-persist**. Recommend one sentence in §4.4
  ("the preference holds even on entities where the model names the current holder when asked") plus an appendix
  table. This is the strongest available answer to the Hossain/Jiang overlap.
- Otherwise (point estimate < 0.5, CI includes 0.5): **direction only**; appendix table and a limitation sentence.
- The `other` and `stale` groups are reported whichever way this goes.

## Budget and safety

About 1 h wall on the local GPU, serial, one model in memory at a time, in a titled herdr pane with the memory
guard of `results_sep10/item5_llama3/run_llama3_queue.sh` (stop before a model if memory pressure is critical).
Outputs: `docs/orientation/results_sep10/knowledge_probe/` (code, log, per-model generations, analysis). Nothing
in the paper is edited.
