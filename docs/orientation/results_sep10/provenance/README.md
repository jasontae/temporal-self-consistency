# Provenance of the Sep 2 numbers

Each number in Jason's Sep 2 PDF that this round touched, with whether it recomputes from a file in this repo. The scripts sit in the sibling folders.

| Sep 2 number | Recomputes? | Where checked | Action |
|---|---|---|---|
| Table 1 ECE (0.4504 / 0.4506 / 0.4552 / 0.4227 / 0.0727) | **yes, exactly** | `../ece_check/` | keep |
| Table 1 caption "exact on four levels without binning" | numbers yes, wording no (the code bins) | `../ece_check/` | reword: 10 equal-frequency bins, identical to per-level here |
| Table 1 Brier and resolution columns | **no**. The constant-0.45 Brier of 0.187 is impossible (minimum 0.2025). | `../ece_check/` | **replace** with `../ece_check/ece_levels.json` values |
| Table 2 ECE | not rebuilt (the train split is not on disk). Binned = per-level is proven from counts. | `../ece_check/` | rebuild pending (FETCH_ALL step 2) |
| 6.2x | yes: 6.20, 95% CI [5.85, 6.62] | `../crossover/` | keep; add CI |
| §4.3 paired bootstrap ΔAUROC +0.067 [0.000, 0.182], p = 0.24 | point estimate yes. CI and p **no**: [−0.083, +0.214], p = 0.39 (stratified: [−0.083, +0.208], p = 0.38). The conclusion is unchanged. | `paired_bootstrap_a3.py` | **replace** CI and p with `paired_bootstrap_a3.json` |
| Table 4, Figure 3 AUROCs (all 12 rows) | **yes, exactly** | `../inverse/` | keep for the 9 admitted; Gemma-4 rows move to the exclusion footnote |
| β = −0.890 | yes (−0.889) | `../inverse/` | **replace**: −0.961 on the 9 admitted (`../inverse/`) |
| clustered SE 0.196, p = 0.020 | no. It is a t-test on 3 df with 3 clusters; CR1 gives SE 0.236, p = 0.064 | `../inverse/` | **remove**; use permutation p and the LMM |
| Spearman −0.673, p = 0.033 | yes | `../inverse/` | **replace**: −0.717 (p = 0.030) on the 9 |
| staleness mean 0.444, CI [0.427, 0.463] | mean yes. The CI is close to an across-model t-interval; the entity-bootstrap CI is [0.407, 0.480] | `../inverse/` | **replace**: 0.439 [0.400, 0.476], entity bootstrap, 9 models |
| omnibus sign test p = 0.002 | yes (10 of 10 below 0.5 gives 2/1024 = 0.00195) | arithmetic | **replace**: 9 of 9, p = 0.0039; lead with the entity bootstrap (`../a5_prominence/`) |
| sitelinks p = 0.39; ≤2 y / >2 y means 0.451 / 0.438 | **no source**, rebuilt: prominence slope p = 0.47–0.59 (conclusion holds); split 0.440 / 0.451 (values do not reproduce; no trend) | `../a5_prominence/` | **replace** with `../a5_prominence/` (slope p = 0.47–0.59; intercept p = 0.028 / 0.054) |
| Appendix C "resampled disjoint pool" row (0.500) and "every person appears once" | **not in the scored data**. All 106 false founders in the claims file and in every `matched_*.jsonl` are true founders of other entities. | `../inverse/` README | **replace** with the A6 rescore (`../a6_disjoint/`) |
| three TSCT seeds 42/123/456 and the direction reversal | **not in this repo** (team LLaMA-3 corrected build) | `../../SEP10_PLAN.md`, item 5 | pending item 5 (FETCH_ALL step 4) |
| §4.3 "twelve models … 0.5414 to 0.9183" | the range reproduces as the within-passage AUROC (`mixed_range.py`), but over **eleven** models (11 `mixed_*.jsonl`), unchanged without Gemma-4 (8 models) | `mixed_range.py` | **replace** "twelve" with "eleven" (review item A10) |
| "ten of twelve admitted"; exclusions "traced to a missing BOS token" | ten reproduces from the stored files; the BOS explanation is incomplete (`../MODELS.md`) | `../MODELS.md` | **replace**: nine admitted, three Gemma-4 rows excluded, footnote with the log-prob evidence |
| Table 3 (evergreen replication) PPL/entropy | 4 of 6 rows measured on reasoning preambles, not answers (99–100% of generations) | `../MODELS.md`, rerun in `../evergreen/` | **replace** with the answer-only rerun |

## Upstream search for the untraced numbers (orchestrator, 2026-09-28)

**Searched:**
- `github.com/jasontae/temporal-self-consistency`, last pushed Aug 19, branches `main` and `eval-toolkit-and-real-figures`
- the HF repos `jasontae/temporal-delta`, `alltriples`, `DavidS64/llama3-hedge-sft` and `tanviv/llama3-hedge-sft`

**Result.** No source exists for the Table 1 Brier/resolution columns, the sitelinks control, the ≤2 y / >2 y split, or the §4.3 CI.
- The only hits are the hardcoded `PLACEHOLDER_RESULTS` in `src/evaluation/generate_results_table.py`, present in this repo since the initial commit `8c945c2`. Its own message reads "Using placeholder data — substitute real results when eval completes."
- The HF repos hold data only.

**Placeholder check (this session).** Two untraced Sep 2 values equal placeholder entries that measure different quantities:

| Sep 2 value | also appears in the placeholder as |
|---|---|
| CE-baseline Brier **0.224** | "SFT only" `em_mean` 0.224 |
| §4.3 CI upper bound **0.182** | "Base LLM" `em_mean` 0.182 |

The placeholder table holds about 30 distinct values between 0.002 and 0.31, so matches this close are plausible by chance. None of the other untraced values appear there (0.218, 0.219, 0.187, 0.031, 0.39, 0.451, 0.438). This is not evidence that the placeholders leaked into the paper, but it is worth asking Jason.

**Remaining possible sources:** Jason's Overleaf or his local machine. The draft message in `../../SEP10_STATUS.md` stands.
