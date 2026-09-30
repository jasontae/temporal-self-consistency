# Pre-registration: does the stale-value preference track pretraining co-occurrence?

Written and committed 2026-09-28, before any count for a study entity was fetched and before any count was joined with any model output. Branch `altj-reframe`. Author: worker B.

## What had been seen before this was written

- **The outcome data.** Worker A's analyses of the same PMI scores: every admitted model's staleness AUROC is below 0.5, with mean 0.439. A5 found no trend with time since the office changed, and a null prominence (sitelinks) slope (`../a5_prominence/README.md`).
- **The 106 entities**, with their current and expired office-holders (`data/stress_tests/matched_regime_claims.jsonl`).
- **Two API pilot queries on a pair outside the study** ("Microsoft", "Satya Nadella") to check reachability and the semantics of CNF AND. The result: AND is order-insensitive, and counts above the clause-frequency cap are sampled estimates that can vary by about 2×.
- **No count for any study entity or person.**

## Hypothesis

The preference for the expired value tracks how often each value co-occurs with the entity in pretraining-scale text. Entities whose expired holder co-occurs with them more often, relative to the current holder, show a stronger preference for the expired holder.

H1: b > 0 in the primary analysis below.

## Data

**Outcome.** Scores are `pmi` from `data/prep/predictions_7b/matched_<name>.jsonl`, original claim build. The staleness arms are identical in the original and A6 builds.
- For entity e and model m: y_em = PMI(volatile_stale) − PMI(volatile_current). Positive means the expired value is less surprising.
- Scaled within model: z_em = y_em / SD_m(y). This divides by the model's SD over entities and does not center, so the sign is preserved.
- Models are the nine admitted in `../inverse/inverse_truth.py` (`ADMITTED`): g3_4b, qwen3_4b_th, tsct, qwen3_4b_it, qwen25_7b, gptoss_20b, qwen36_35b, qwen35_27b, qwen36_27b_stock.
- Entity-level outcome: ȳ_e = mean over the nine models of z_em.

**Counts.** From the public infini-gram API (`https://api.infini-gram.io/`, `query_type: count`). Only API queries are used: no corpus download, sequential requests, at least 1 s between requests, and up to 5 retries with exponential backoff.
- E = the `entity` string of the claim, verbatim.
- P = the `person` string, verbatim (e.g. "Fr. Francesco Bamonte", "Ben Vinson III").
- R = the `role` string.

| id | form | query | parameters |
|---|---|---|---|
| F1 (**primary**) | CNF AND, window 100 | `E AND P` | max_diff_tokens 100, max_clause_freq 500000 |
| F2 | CNF AND, window 20 | `E AND P` | max_diff_tokens 20, max_clause_freq 500000 |
| F3 | exact phrase, OR of templates | `R of E is P OR R of E was P OR P, R of E OR P, the R of E OR E R P` | one disjunctive clause |
| F4 | person marginal | `P` | exact n-gram |
| F5 | entity marginal | `E` | exact n-gram |

| index | corpus | role |
|---|---|---|
| `v4_dolma-v1_7_llama` | Dolma v1.7, 2.6T tokens | **primary**: the largest and most recent of the three corpora the brief names |
| `v4_rpj_llama_s4` | RedPajama, 1.4T tokens | secondary |
| `v4_piletrain_llama` | Pile-train, 0.38T tokens | secondary |

All five forms are run on Dolma. F1, F2 and F4 are run on RedPajama and Pile. The `approx` flag is recorded for every count.

**Predictor.** x_e = ln((c_exp + 0.5) / (c_cur + 0.5)), where c is the F1 Dolma count for the expired or current holder. The 0.5 is added because zero counts are expected.

## Primary analysis (confirmatory)

1. **Statistic:** the OLS slope b of ȳ_e on x_e over all 106 entities.
2. **Test:** two-sided permutation p, with 10,000 permutations of x across entities (seed 20260928). The entity is the exchangeable unit, because x is an entity-level property and all nine models score the same entities.
3. **Decision:**
   - **supported:** b > 0 and p < 0.05
   - **contradicted:** b < 0 and p < 0.05, in which case worker B stops and reports before the result is used anywhere
   - **null:** otherwise, reported as null in plain words
4. **Model-based estimate reported beside it:** a crossed random-intercept model, z_em = b0 + b1·x_e + u_e + v_m + ε, fitted by REML with the same `reml_crossed` routine as `../inverse/inverse_truth.py`. b1 is tested on t with 104 df (entities − 2). This is reported, not used for the decision.

## No-count baseline

This asks whether the counts add anything over what is predictable without them.

| model | predictors of ȳ_e |
|---|---|
| M0 | intercept only |
| M1 (no-count covariates) | Δlog(1 + sitelinks) (current − expired) and years since the office changed, to 2026-08-11. Both come from A5's inputs (`data/prep/wikidata_a6/set_persons.json`, `data/prep/wikidata_matched_cache/`), on the 98 entities that have both. |
| M2 | x_e |
| M3 | x_e + the M1 covariates |

- **Procedure:** on the 98-entity subset, 10-fold cross-validation by entity, repeated 50 times, reporting mean out-of-fold MSE for M0–M3.
- **"Counts add over the no-count baseline"** requires both of these:
  - CV-MSE(M3) < CV-MSE(M1), and
  - in M3, the permutation p for x (10,000 permutations of x across entities, covariates held fixed) is below 0.05.
- The correlation of x with years since the change is reported. The two are expected to be related, because current holders who took office after a corpus was collected have few mentions in it.

## Secondary analyses (exploratory, labelled as such)

- **S1.** Replace F1 with F2 on Dolma. Use F3 on Dolma only if at least 30 entities have c_exp + c_cur > 0 under F3; otherwise report its coverage and nothing else.
- **S2.** Use F1 on RedPajama and on Pile-train. Report Holm-adjusted p across the three indexes, alongside unadjusted p.
- **S3 (co-occurrence vs name frequency).** Let x_name = ln((n_exp + 0.5) / (n_cur + 0.5)) from F4. Fit ȳ_e on x and x_name jointly, and report each slope with its permutation p. The PMI outcome already subtracts each name's log-prob in a neutral frame, so x_name is expected to matter less than x if the hypothesis is right.
- **S4 (binary outcome).** Entity-mean paired win rate, the mean over models of 1[PMI stale > PMI current], regressed on x.
- **S5.** Per-model slopes of z_em on x_e (nine), and how many are positive.
- **S6 (robustness):**
  - (a) Drop entities with c_exp = c_cur = 0 under F1.
  - (b) Drop entities where either F1 count is approximate.
  - (c) Use ±1 instead of ±0.5 as the zero correction.
- **S7 (sensitivity model sets, as in the appendix):** add Qwen3.8-27B; add gemma-4-E4B.
- **S8 (descriptive).** AUROC of x_e for the sign of ȳ_e.

No multiplicity correction is applied to S1–S8. None of them can change the primary decision.

## Outputs

- `fetch_counts.py` → `counts.jsonl`: one row per (index, form, entity, value), with the query string, parameters, count, approx flag, latency and timestamp.
- `analyze_frequency.py` → `frequency_results.json`, `frequency_results.md`, `frequency.png` and `frequency.pdf`. The figure shows ȳ_e against x_e with the OLS line, plus the permutation null for b with the observed value marked.
- `RESULTS.md`: the decision in the first sentence; any deviation from this document, with its reason.

## Process commitments

- The analysis script is committed before it is first run on the fetched counts.
- The counts file may be inspected for data-quality problems (errors, approx flags, obviously broken queries) before the analysis runs. It is not joined with model outputs outside `analyze_frequency.py`.
- Every deviation from this document is listed in `RESULTS.md`.
- A null result is reported as null.

## Known limitations, stated in advance

- The corpora are proxies. None of them is the pretraining data of any evaluated model.
- Entity strings are ambiguous for some entities ("Consensus", "Doctrine", "Democrats", "Chicago"). The window constraint with a full person name limits this but does not remove it.
- Counts above the clause cap are sampled estimates (hence S6b).
- Several "current" values in the claim file may themselves be wrong or dated (the Wikidata snapshot). That adds noise in the outcome and in x alike.

---

## Amendment 1 (2026-09-28, before any count was joined with any outcome)

Requested through the orchestrator, from the fused Fusion critique (`../../FUSION_FUSED.md` (b)). Status when this was written:
- the count fetch was about 20% complete;
- `analyze_frequency.py` had run only on synthetic random counts;
- no real count had been joined with any model output.

Everything above stays in force. This amendment adds controls and one reading rule.

### Dolma, RedPajama and the Pile are exposure proxies

None of the three corpora is the pretraining data of any evaluated model. The Qwen, Gemma and gpt-oss corpora are unreleased. A count here measures how often a pairing appears in large open web-and-books text of roughly the same era, which serves as a proxy for how much exposure a model trained on similar data would have had. Even a positive result shows only that the preference tracks *proxy* exposure. The paper must say "open-corpus co-occurrence", never "the model's pretraining frequency".

### Added controls

These are defined per entity, and "expired − current" is written Δ.

| control | definition | source |
|---|---|---|
| **token length** | Δ number of Llama-2 tokens in the person name, as tokenized by the infini-gram API (the tokenization the counts are measured in) | new query pass `fetch_token_lengths.py`: one `count` query per distinct person on Dolma, recording `len(token_ids)` |
| token length (secondary) | Δ character length of the person name | claims file |
| **relation** | role fixed effects with three levels: chairperson (61), CEO (34), other (head of government 8 + director or manager 3) | claims file |
| **entity** | log(1 + F5 count), the entity's own Dolma frequency. The entity random intercept stays in the REML model. An entity fixed effect cannot be used with an entity-level x. | `counts.jsonl` |
| **time in office (current)** | years from the current holder's start (P580) to 2026-08-11; this is A5's "years since change" | tracked `data/prep/wikidata_cache/` |
| **time in office (expired)** | years between the expired holder's P580 and P582 on the entity's volatile property (the latest statement for that person's QID; QID from `set_persons.json` by label) | same |
| prominence | Δlog(1 + sitelinks), as in M1 | `set_persons.json` |

### Added analysis: the controlled test (C1)

- **Model:** ȳ_e ~ 1 + x_e + Δtoken length + role FE + log(1 + entity count) + current time in office + expired time in office + Δlog sitelinks, on the entities where every control is defined.
- **Test:** the coefficient on x, with a two-sided permutation p over 10,000 permutations of x across entities, all controls held fixed (Frisch–Waugh, as in M3).
- **Model-based estimate reported beside it:** a crossed REML model with the same fixed effects plus entity and model random intercepts.
- **Missingness:** the number of entities dropped for a missing control is reported. If more than 25 are lost, C1 is also run without the expired-tenure control (C1b), since that is the control most likely to be missing.

### Reading rule for the thesis

The primary decision (unadjusted slope) is unchanged and is still reported first. For the thesis clause "is familiarity" / "tracks exposure" to be used, **both** of these must hold:
- the primary decision is "supported", and
- C1's x coefficient is positive with p < 0.05.

**Supported but C1 not** means the paper says the preference tracks proxy co-occurrence but that this is not separable from the listed controls, and the thesis uses the "behaves like familiarity" wording (`REFRAME.md` §2, check 3).

The "contradicted" stop rule applies to the primary decision only. A significantly negative C1 coefficient with a null primary result is reported, not treated as a stop.

### Added to the secondary list

- **S9.** C1 with each control dropped in turn (seven fits), to show which control moves the x coefficient.
- **S10.** Per-model token length (Δ tokens under each model's own tokenizer) is **not** added. Two of the nine tokenizers (Qwen2.5-7B, TSCT) are not on disk (`../MODELS.md` finding 7), and substituting one would not be the same model. This is noted as a limitation.
