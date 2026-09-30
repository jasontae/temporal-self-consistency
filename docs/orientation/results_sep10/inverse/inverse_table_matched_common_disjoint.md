| model | family | quant | truth AUROC [95% CI] | regime AUROC [95% CI] | regime paired win | sign p | staleness [95% CI] |
|---|---|---|---|---|---|---|---|
| gemma-3-4B | gemma | 4-bit | 0.719 [0.655, 0.778] | 0.620 [0.561, 0.679] | 0.670 | 0.0006 | 0.454 [0.397, 0.514] |
| Qwen3-4B-Thinking | qwen | 8-bit | 0.746 [0.692, 0.797] | 0.622 [0.567, 0.677] | 0.651 | 0.0024 | 0.456 [0.393, 0.517] |
| Qwen3-4B-Instruct | qwen | 8-bit | 0.771 [0.717, 0.823] | 0.568 [0.503, 0.629] | 0.519 | 0.7709 | 0.480 [0.418, 0.539] |
| gpt-oss-20B | gpt-oss | 6.5-bit | 0.806 [0.754, 0.856] | 0.570 [0.506, 0.634] | 0.547 | 0.3821 | 0.415 [0.353, 0.473] |
| Qwen3.6-35B-A3B | qwen | nvfp4 | 0.844 [0.795, 0.891] | 0.505 [0.449, 0.562] | 0.509 | 0.9227 | 0.451 [0.396, 0.505] |
| Qwen3.5-27B | qwen | 4-bit | 0.845 [0.795, 0.892] | 0.521 [0.461, 0.579] | 0.500 | 1.0000 | 0.415 [0.362, 0.466] |
| Qwen3.6-27B | qwen | 4-bit | 0.855 [0.808, 0.899] | 0.510 [0.453, 0.567] | 0.519 | 0.7709 | 0.442 [0.387, 0.492] |
| gemma-4-E4B (excluded) | gemma | 4-bit | 0.738 [0.683, 0.793] | 0.601 [0.533, 0.667] | 0.632 | 0.0084 | 0.498 [0.441, 0.554] |

**Model level (n = 7 admitted).** OLS slope -0.884; classical SE 0.112, p = 0.0005 (t, 5 df); family-clustered CR1 SE 0.090 (3 clusters), p = 0.010 (t, 2 df). Pearson r = -0.962 (p = 0.0005); Spearman rho = -0.821 (p = 0.0234); permutation p = 0.0036 (exact). Entity-bootstrap 95% CI for the slope [-1.510, -0.303].

Leave-one-model-out slopes: -0.970 to -0.812. Leave-one-family-out: without gemma: slope -0.970, rho -0.771 (n=6); without gpt-oss: slope -0.893, rho -0.771 (n=6).

**Split-half control** (truth on one half of entities, regime on the other, 2000 splits): median slope -0.867, 95% of splits in [-1.393, -0.388], 100.0% negative; median Spearman -0.893.

**Crossed random-effects model** (REML; random intercepts entity, model, family; b1 tested on t with n_models - 2 df):

| outcome | b1 (per unit truth AUROC) | SE | t | p | variances entity / model / family / residual |
|---|---|---|---|---|---|
| paired win 1[T > V] | -1.148 | 0.293 | -3.92 | 0.0112 (5 df) | 0.0867 / 0.0000 / 0.0000 / 0.1577 |
| pmi difference, z within model | -3.016 | 0.536 | -5.63 | 0.0025 (5 df) | 0.4019 / 0.0000 / 0.0000 / 0.5274 |

Staleness: mean over admitted models 0.445, entity-bootstrap 95% CI [0.404, 0.484].
