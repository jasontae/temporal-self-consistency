| model | family | quant | truth AUROC [95% CI] | regime AUROC [95% CI] | regime paired win | sign p | staleness [95% CI] |
|---|---|---|---|---|---|---|---|
| gemma-4-E4B | gemma | 4-bit | 0.750 [0.692, 0.804] | 0.601 [0.533, 0.667] | 0.632 | 0.0084 | 0.498 [0.441, 0.554] |
| gemma-3-4B | gemma | 4-bit | 0.770 [0.711, 0.826] | 0.620 [0.561, 0.679] | 0.670 | 0.0006 | 0.454 [0.397, 0.514] |
| Qwen3-4B-Thinking | qwen | 8-bit | 0.777 [0.725, 0.826] | 0.622 [0.567, 0.677] | 0.651 | 0.0024 | 0.456 [0.393, 0.517] |
| Qwen3-4B-Instruct | qwen | 8-bit | 0.797 [0.745, 0.848] | 0.568 [0.503, 0.629] | 0.519 | 0.7709 | 0.480 [0.418, 0.539] |
| gpt-oss-20B | gpt-oss | 6.5-bit | 0.799 [0.754, 0.845] | 0.570 [0.506, 0.634] | 0.547 | 0.3821 | 0.415 [0.353, 0.473] |
| Qwen3.6-35B-A3B | qwen | nvfp4 | 0.846 [0.799, 0.892] | 0.506 [0.449, 0.563] | 0.509 | 0.9227 | 0.450 [0.396, 0.505] |
| Qwen3.5-27B | qwen | 4-bit | 0.852 [0.807, 0.895] | 0.521 [0.461, 0.579] | 0.500 | 1.0000 | 0.415 [0.362, 0.466] |
| Qwen3.6-27B | qwen | 4-bit | 0.856 [0.813, 0.898] | 0.510 [0.453, 0.567] | 0.519 | 0.7709 | 0.442 [0.387, 0.492] |
| gemma-4-26B-A4B (excluded) | gemma | 4-bit | 0.530 [0.457, 0.601] | 0.373 [0.298, 0.451] | 0.321 | 0.0003 | 0.548 [0.482, 0.609] |
| gemma-4-31B (excluded) | gemma | 8-bit | 0.590 [0.514, 0.664] | 0.554 [0.477, 0.629] | 0.519 | 0.7709 | 0.536 [0.478, 0.594] |

**Model level (n = 8 admitted).** OLS slope -1.105; classical SE 0.162, p = 0.0005 (t, 6 df); family-clustered CR1 SE 0.139 (3 clusters), p = 0.015 (t, 2 df). Pearson r = -0.941 (p = 0.0005); Spearman rho = -0.810 (p = 0.0149); permutation p = 0.0030 (exact). Entity-bootstrap 95% CI for the slope [-1.787, -0.282].

Leave-one-model-out slopes: -1.307 to -1.027. Leave-one-family-out: without gemma: slope -1.275, rho -0.771 (n=6); without gpt-oss: slope -1.107, rho -0.750 (n=7); without qwen: slope -0.709, rho -0.500 (n=3).

**Split-half control** (truth on one half of entities, regime on the other, 2000 splits): median slope -1.032, 95% of splits in [-1.806, -0.484], 100.0% negative; median Spearman -0.850.

**Crossed random-effects model** (REML; random intercepts entity, model, family; b1 tested on t with n_models - 2 df):

| outcome | b1 (per unit truth AUROC) | SE | t | p | variances entity / model / family / residual |
|---|---|---|---|---|---|
| paired win 1[T > V] | -1.457 | 0.375 | -3.89 | 0.0081 (6 df) | 0.0755 / 0.0001 / 0.0000 / 0.1678 |
| pmi difference, z within model | -3.825 | 0.689 | -5.55 | 0.0014 (6 df) | 0.3700 / 0.0000 / 0.0000 / 0.5884 |

Staleness: mean over admitted models 0.451, entity-bootstrap 95% CI [0.413, 0.488].
