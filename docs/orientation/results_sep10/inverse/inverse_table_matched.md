| model | family | quant | truth AUROC [95% CI] | regime AUROC [95% CI] | regime paired win | sign p | staleness [95% CI] |
|---|---|---|---|---|---|---|---|
| gemma-3-4B | gemma | 4-bit | 0.719 [0.655, 0.778] | 0.620 [0.561, 0.679] | 0.670 | 0.0006 | 0.454 [0.397, 0.514] |
| Qwen3-4B-Thinking | qwen | 8-bit | 0.746 [0.692, 0.797] | 0.622 [0.567, 0.677] | 0.651 | 0.0024 | 0.456 [0.393, 0.517] |
| trained model (Qwen2.5-7B + TSCT) | qwen | 4-bit | 0.768 [0.712, 0.822] | 0.640 [0.585, 0.696] | 0.632 | 0.0084 | 0.410 [0.353, 0.463] |
| Qwen3-4B-Instruct | qwen | 8-bit | 0.771 [0.717, 0.823] | 0.568 [0.503, 0.629] | 0.519 | 0.7709 | 0.480 [0.418, 0.539] |
| Qwen2.5-7B | qwen | 4-bit | 0.794 [0.739, 0.848] | 0.634 [0.582, 0.691] | 0.642 | 0.0046 | 0.426 [0.368, 0.476] |
| gpt-oss-20B | gpt-oss | 6.5-bit | 0.806 [0.754, 0.856] | 0.570 [0.506, 0.634] | 0.547 | 0.3821 | 0.415 [0.353, 0.473] |
| Qwen3.6-35B-A3B | qwen | nvfp4 | 0.844 [0.795, 0.891] | 0.505 [0.449, 0.562] | 0.509 | 0.9227 | 0.451 [0.396, 0.505] |
| Qwen3.5-27B | qwen | 4-bit | 0.845 [0.795, 0.892] | 0.521 [0.461, 0.579] | 0.500 | 1.0000 | 0.415 [0.362, 0.466] |
| Qwen3.6-27B | qwen | 4-bit | 0.855 [0.808, 0.899] | 0.510 [0.453, 0.567] | 0.519 | 0.7709 | 0.442 [0.387, 0.492] |
| gemma-4-26B-A4B (excluded) | gemma | 4-bit | 0.584 [0.512, 0.662] | 0.373 [0.298, 0.451] | 0.321 | 0.0003 | 0.548 [0.482, 0.609] |
| gemma-4-31B (excluded) | gemma | 8-bit | 0.669 [0.598, 0.741] | 0.554 [0.477, 0.629] | 0.519 | 0.7709 | 0.536 [0.478, 0.594] |
| gemma-4-E4B (excluded) | gemma | 4-bit | 0.738 [0.683, 0.793] | 0.601 [0.533, 0.667] | 0.632 | 0.0084 | 0.498 [0.441, 0.554] |

**Model level (n = 9 admitted).** OLS slope -0.961; classical SE 0.238, p = 0.0050 (t, 7 df); family-clustered CR1 SE 0.227 (3 clusters), p = 0.051 (t, 2 df). Pearson r = -0.836 (p = 0.0050); Spearman rho = -0.717 (p = 0.0298); permutation p = 0.0082 (exact). Entity-bootstrap 95% CI for the slope [-1.568, -0.330].

Leave-one-model-out slopes: -1.171 to -0.876. Leave-one-family-out: without gemma: slope -1.171, rho -0.762 (n=8); without gpt-oss: slope -0.964, rho -0.643 (n=8).

**Split-half control** (truth on one half of entities, regime on the other, 2000 splits): median slope -0.916, 95% of splits in [-1.512, -0.428], 100.0% negative; median Spearman -0.767.

**Crossed random-effects model** (REML; random intercepts entity, model, family; b1 tested on t with n_models - 2 df):

| outcome | b1 (per unit truth AUROC) | SE | t | p | variances entity / model / family / residual |
|---|---|---|---|---|---|
| paired win 1[T > V] | -1.203 | 0.323 | -3.73 | 0.0074 (7 df) | 0.0759 / 0.0003 / 0.0000 / 0.1662 |
| pmi difference, z within model | -3.324 | 0.881 | -3.77 | 0.0070 (7 df) | 0.3989 / 0.0090 / 0.0000 / 0.5361 |

Staleness: mean over admitted models 0.439, entity-bootstrap 95% CI [0.399, 0.476].
