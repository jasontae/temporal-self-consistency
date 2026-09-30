| model | family | quant | truth AUROC [95% CI] | regime AUROC [95% CI] | regime paired win | sign p | staleness [95% CI] |
|---|---|---|---|---|---|---|---|
| gpt-oss-20B | gpt-oss | 6.5-bit | 0.665 [0.603, 0.728] | 0.573 [0.511, 0.635] | 0.585 | 0.0982 | 0.459 [0.400, 0.517] |
| Qwen3-4B-Thinking | qwen | 8-bit | 0.763 [0.713, 0.813] | 0.601 [0.546, 0.655] | 0.575 | 0.1448 | 0.466 [0.409, 0.522] |
| gemma-3-4B | gemma | 4-bit | 0.770 [0.711, 0.826] | 0.620 [0.561, 0.679] | 0.670 | 0.0006 | 0.454 [0.397, 0.514] |
| Qwen3-4B-Instruct | qwen | 8-bit | 0.782 [0.733, 0.831] | 0.524 [0.467, 0.582] | 0.472 | 0.6274 | 0.458 [0.400, 0.511] |
| trained model (Qwen2.5-7B + TSCT) | qwen | 4-bit | 0.794 [0.739, 0.844] | 0.640 [0.585, 0.696] | 0.632 | 0.0084 | 0.410 [0.353, 0.463] |
| Qwen2.5-7B | qwen | 4-bit | 0.818 [0.762, 0.865] | 0.634 [0.582, 0.691] | 0.642 | 0.0046 | 0.426 [0.368, 0.476] |
| Qwen3.5-27B | qwen | 4-bit | 0.852 [0.807, 0.895] | 0.521 [0.461, 0.579] | 0.500 | 1.0000 | 0.415 [0.362, 0.466] |
| Qwen3.6-27B | qwen | 4-bit | 0.856 [0.813, 0.898] | 0.510 [0.453, 0.567] | 0.519 | 0.7709 | 0.442 [0.387, 0.492] |
| Qwen3.6-35B-A3B | qwen | nvfp4 | 0.866 [0.821, 0.908] | 0.553 [0.498, 0.608] | 0.575 | 0.1448 | 0.450 [0.394, 0.503] |

**Model level (n = 9 admitted).** OLS slope -0.268; classical SE 0.290, p = 0.3865 (t, 7 df); family-clustered CR1 SE 0.263 (3 clusters), p = 0.416 (t, 2 df). Pearson r = -0.330 (p = 0.3865); Spearman rho = -0.400 (p = 0.2861); permutation p = 0.4040 (exact). Entity-bootstrap 95% CI for the slope [-0.653, 0.095].

Leave-one-model-out slopes: -0.729 to -0.145. Leave-one-family-out: without gemma: slope -0.231, rho -0.429 (n=8); without gpt-oss: slope -0.729, rho -0.429 (n=8).

**Split-half control** (truth on one half of entities, regime on the other, 2000 splits): median slope -0.278, 95% of splits in [-0.654, 0.029], 96.5% negative; median Spearman -0.417.

**Crossed random-effects model** (REML; random intercepts entity, model, family; b1 tested on t with n_models - 2 df):

| outcome | b1 (per unit truth AUROC) | SE | t | p | variances entity / model / family / residual |
|---|---|---|---|---|---|
| paired win 1[T > V] | -0.224 | 0.431 | -0.52 | 0.6182 (7 df) | 0.0695 / 0.0029 / 0.0011 / 0.1732 |
| pmi difference, z within model | -1.017 | 1.058 | -0.96 | 0.3685 (7 df) | 0.3492 / 0.0298 / 0.0000 / 0.5364 |

Staleness: mean over admitted models 0.442, entity-bootstrap 95% CI [0.404, 0.477].
