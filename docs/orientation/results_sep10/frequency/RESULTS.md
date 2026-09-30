# Frequency experiment: results

**The pre-registered primary result is null.** In Dolma v1.7, the ratio of entity–expired-holder to entity–current-holder co-occurrence does not significantly predict the per-entity preference for the expired value:
- slope +0.038 per log unit, in the hypothesized direction;
- two-sided permutation p = 0.119 (10,000 permutations);
- crossed REML with entity and model random intercepts: p = 0.121;
- n = 106 entities, nine admitted models.

With the amendment-1 controls (token length, relation, entity frequency, time in office for both holders, prominence), the coefficient is −0.021 (p = 0.39). The counts do not add over the no-count baseline. The result neither supports nor contradicts the familiarity reading. The stop rule was for a significant *negative* primary result, and it did not trigger.

Pre-registration: `PREREG.md` (committed `2b0f6f3` before any count was fetched) and amendment 1 (`6b44327`, before any join). Scripts were committed before the analysis ran on real counts (`05ccebe`, `d51dbb2`); the raw counts were committed before the run (`dfe31da`). Full tables: `frequency_results.md` and `.json`. Figure: `frequency.png` and `.pdf`.

## Primary and pre-registered tests

| test | estimate | p | decision |
|---|---|---|---|
| **Primary:** OLS slope of ȳ_e on x_e, Dolma, F1 (entity AND person within 100 tokens) | +0.0378 (r = +0.15, ρ = +0.13) | 0.119 (permutation); 0.121 (REML) | **null** |
| **No-count baseline:** CV-MSE, M3 (counts + covariates) vs M1 (covariates only), 98 entities | 0.4624 vs 0.4583; x in M3: +0.029, p = 0.25 | — | **counts do not add** |
| **C1, amendment 1, all controls,** 96 entities | −0.0213 | 0.39 (permutation); 0.49 (REML) | not positive |
| **Reading rule (amendment 1)** | primary not supported | — | the thesis uses the **"behaves like familiarity"** wording, and the title "Familiarity, Not Foresight" is dropped (`../../REFRAME.md` §2, check 3) |

Only one control changes the sign of the controlled coefficient when dropped: entity frequency (S9: +0.008 without it). x correlates with Δlog sitelinks at r = −0.34, meaning x is higher where the expired holder is more prominent. Whatever raw association exists overlaps with prominence and entity frequency.

## What the data do show (descriptive)

- **In the aggregate, corpus and models agree on direction.** Among the 77 entities with any F1 co-occurrence, the expired holder co-occurs more often than the current one for 68 (9 go the other way; 29 have both counts zero). The models prefer the expired value on 61% of entities (ȳ_e > 0). Both lean toward the expired value, as a familiarity account expects. But **per-entity magnitudes do not line up** beyond noise, and that per-entity alignment is what the test measures.
- **All nine per-model slopes are positive** (S5: +0.004 to +0.067), but each is small.
- **Exploratory, not confirmatory:**
  - On the oldest corpus, Pile-train (2020), the F1 slope is +0.095, p = 0.0015, Holm-adjusted across the three indexes p = 0.0045 (S2). RedPajama gives +0.046, p = 0.065.
  - With the marginal name-frequency ratio held fixed, the Dolma coefficient is +0.060, p = 0.012 (S3).

  These were pre-registered as secondary analyses with no power over the primary decision. One reading is that exposure from before the changes is the relevant signal, and the Pile predates most of them. That is a hypothesis for a new pre-registered test, not a finding. **The paper must not headline the Pile result.**
- **Robustness of the null** (S6): dropping both-zero entities gives +0.017, p = 0.58; dropping approximate counts gives +0.023, p = 0.42; a zero correction of 1 gives +0.043, p = 0.096. Adding Qwen3.8 or gemma-4-E4B gives the same picture (p 0.23 and 0.10).
- **Exact-phrase templates** (F3): only 26 entities have any count, below the pre-registered 30, so F3 is not analysed.

## What the paper can say (§7.4 → one sentence in §7.3)

> "The preference is not predicted by open-corpus co-occurrence: across 106 entities, the log ratio of expired to current entity–holder co-occurrence in Dolma v1.7 does not predict the per-entity preference (pre-registered; slope +0.04, permutation p = 0.12), and the estimate falls to zero with token-length, relation, entity-frequency, tenure and prominence controls."

The appendix (App. M) gets the pre-registration, the query forms, the count-quality table, the figure, and the exploratory Pile result, labelled as such.

**What this does to the thesis:** "familiarity" stays as a description of the scores (the expired value is less surprising), not as a mechanism. The Dolma, RedPajama and Pile counts are exposure proxies, not these models' pretraining data (amendment 1). So the null says only that *open-corpus* co-occurrence does not explain which entities show the preference. It does not show that exposure is irrelevant.

## Deviations from PREREG.md

1. **F3 is sent as two queries and summed.** The API accepts at most four terms per OR clause ("Please enter at most 4 terms in each disjunctive clause!"). The five pre-registered templates are therefore sent as F3a (four templates) and F3b (`E R P`), and the counts are added. The templates are distinct n-grams that cannot match at the same position, so the sum equals the pre-registered single-clause OR count. Both parts are kept in `counts.jsonl`.
2. **Start dates come from the tracked `data/prep/wikidata_cache/`.** A5 read `wikidata_matched_cache/`, which is untracked in worker A's checkout and absent from this worktree. The same A5 `start_years` code, pointed at the tracked cache, reproduces A5's covariate summary: 98 entities with a start date, median 2.61 years. Sitelinks come from the tracked `set_persons.json`, as in A5.
3. **First fetch aborted and restarted.** The first run stopped after 6 rows, on the F3 error above; the file was deleted and the fetch restarted from zero. The progress log had printed two study counts: SACD / Brigitte Buc, Dolma F1 = 0, and SACD / Anne Rambach, Pile F4 = 3. No count was joined with any outcome before `analyze_frequency.py` was committed (`05ccebe`).
4. **Retry policy.** A query error the API returns as JSON (not a network or server error) is retried once, not five times. It still gets up to five attempts on transient failures.
5. **Throttling (HTTP 403) during the fetch.** From about 10:12 UTC, the API intermittently returned HTTP 403 `{"message": "Forbidden"}`, with no count and no `error` field. The original script recorded those rows with `count: null` and did not retry them; 101 rows were affected in all (78 had been seen at the 10:25 check; the rest arrived before the stop). After the resume, every one of the 2,438 query keys has a count, and 2,539 rows were written in total. The fetch was stopped. The request gap was raised from 1 s to 2 s, and a missing count is now retried with a 30 s backoff that doubles on each retry, for up to 6 attempts. On resume, the failed rows are re-fetched, and the analysis keeps the last row per key. No count was used before this was fixed.
6. **S9 "seven fits".** Amendment 1 says "C1 with each control dropped in turn (seven fits)". There are six control groups: token length, relation, entity frequency, both tenures, and prominence. The seventh fit swaps token length for character length (the secondary length control). The label was a miscount; the fits are those.
7. **Covariate check.** The covariates computed from the tracked cache reproduce A5's summary (98 entities with a start date, median 2.61 years). Expired-holder tenure is available for 103 entities, and the full C1 control set for 96.
