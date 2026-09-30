# A5: the staleness result against name prominence and time since change

**Scripts:** `a5_prominence.py` (prominence covariate and time strata), `staleness_estimands.py` (the headline under three estimands). Both take `--with-e4b` for the appendix sensitivity.

**Outputs:** `a5_table_{nine,with_e4b}.md`, `a5_*.json`, `a5_*.out`, `staleness_estimands_*.{json,out}`.

**Inputs:**
- `data/prep/predictions_7b/matched_*.jsonl`
- sitelinks from `data/prep/wikidata_a6/set_persons.json`: Wikidata SPARQL fetched 2026-09-28; three names resolved via `wbsearchentities`, recorded in the file
- office start dates (P580) from `data/prep/wikidata_matched_cache/`
- the audit date 2026-08-11 from `gold_currency_audit.json`

Staleness uses only the current and stale office-holder arms. Those are identical in the original and A6 builds, so nothing here depends on the false-founder rebuild.

## The Sep 2 numbers these replace

The Sep 2 PDF reports "p = 0.39 for the prominence covariate" and "≤ 2 years 0.451, > 2 years 0.438". The repo had no code or data for either.

## Results (nine admitted models; Gemma-4 dropped)

**1. The staleness headline, with entities resampled (all 106 entities, 4,000 resamples):**

| estimand | value [95% CI] | p (bootstrap) |
|---|---|---|
| unpaired AUROC (current vs stale), mean over models | 0.439 [0.400, 0.476] | 0.001 |
| paired win rate 1[current > stale] | 0.426 [0.369, 0.482] | 0.008 |
| PMI difference, z within model | −0.21 [−0.34, −0.09] | 0.0005 |
| across-model sign test (Sep 2 style; models treated as independent) | 9 of 9 below 0.5 | 0.0039 |

The Sep 2 style test treats models as independent, but all nine score the same 106 entities. The entity bootstrap is the appropriate test, and it still excludes chance on every estimand.

**2. Prominence.** On 98 entities that have both sitelinks and a start date:
- The **stale** holder is the more prominent name on 70% of entities (mean Δlog(1+sitelinks), current − stale, is −0.17). Prominence therefore runs in the direction that could produce a familiarity preference for the stale name.
- It does not explain the effect. A crossed random-effects model (entity, model) with Δlog sitelinks as covariate gives:

  | outcome | intercept (preference at equal prominence) | p | prominence slope | p |
  |---|---|---|---|---|
  | paired win | 0.434 | 0.054 | −0.027 | 0.47 |
  | z PMI difference | −0.160 | 0.028 | +0.045 | 0.59 |

- The prominence slope is null, consistent with the Sep 2 conclusion (p = 0.39 printed; 0.47–0.59 here).
- At equal prominence the preference remains, but on this 98-entity subset it is significant only for the continuous outcome (p = 0.028). The binary win rate gives p = 0.054.

**3. Time since the office changed** (to 2026-08-11; median 2.6 years):

| stratum | entities | mean staleness AUROC [95% CI] |
|---|---|---|
| ≤ 2 years | 25 | 0.440 [0.340, 0.542] |
| > 2 years | 73 | 0.451 [0.411, 0.490] |
| tertile 1 (≤ 2.4 y) | 35 | 0.447 [0.371, 0.519] |
| tertile 2 | 31 | 0.471 [0.414, 0.527] |
| tertile 3 (> 3.0 y) | 32 | 0.433 [0.362, 0.500] |

Spearman(years, mean paired win) = 0.005 (p = 0.96). The printed Sep 2 values (0.451 / 0.438) do not reproduce; here the order is reversed (0.440 / 0.451). The conclusion "present in both recent and older transitions" holds at the point-estimate level, but the ≤ 2-year stratum (25 entities) has a CI that includes 0.5. There is no trend with time.

**With gemma-4-E4B (the Sep 2 ten; appendix sensitivity):**
- intercept at equal prominence: paired win 0.444 (p = 0.092), z difference −0.143 (p = 0.042)
- prominence slope p = 0.51–0.55
- strata: 0.453 / 0.455

See `a5_table_with_e4b.md` and `staleness_estimands_with_e4b.json`.

## What the paper can say

- The staleness preference holds when entities are resampled (p ≤ 0.008 on every estimand).
- It is not explained by prominence: the stale holder is usually more prominent, but the prominence slope is null.
- It shows no dependence on time since the change.
- State it with the entity bootstrap, not the across-model sign test.
- Report that the binary paired-win version is marginal (p = 0.054) once prominence is conditioned on.
