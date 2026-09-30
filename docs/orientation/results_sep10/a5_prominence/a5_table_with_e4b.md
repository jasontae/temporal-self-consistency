### A5 (with_e4b: 10 models, 98 entities)

Current holder more prominent than stale holder on 30% of entities (mean Δlog(1+sitelinks) -0.17). Median time since change 2.6 y (to 2026-08-11).

| outcome | staleness preference at equal prominence (intercept) | p | prominence covariate (slope on Δlog sitelinks) | p |
|---|---|---|---|---|
| paired win 1[current > stale] | 0.444 (null 0.5), SE 0.033 | 0.092 | -0.022, SE 0.036 | 0.554 |
| pmi difference, z within model | -0.143 (null 0.0), SE 0.069 | 0.042 | +0.053, SE 0.081 | 0.510 |

| stratum (years since change) | entities | mean staleness AUROC [95% CI] |
|---|---|---|
| <= 2 years | 25 | 0.453 [0.361, 0.547] |
| > 2 years | 73 | 0.455 [0.415, 0.493] |
| tertile 1 (<= 2.4 y) | 35 | 0.459 [0.387, 0.526] |
| tertile 2 | 31 | 0.477 [0.424, 0.529] |
| tertile 3 (> 3.0 y) | 32 | 0.430 [0.361, 0.498] |

Spearman(years since change, mean paired win across models) = -0.010 (p = 0.925).
