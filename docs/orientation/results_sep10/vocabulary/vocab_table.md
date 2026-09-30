### Floor sweep (exact; oracle ECE unchanged by the floor)

| lowest level | best-constant ECE, released key | oracle / constant | best-constant ECE, repaired key | oracle / constant |
|---|---|---|---|---|
| 0.10 | 0.0727 | 6.2x | 0.0974 | 4.9x |
| 0.05 | 0.0227 | 19.9x | 0.0474 | 10.1x |
| 0.01 | 0.0173 | 26.0x | 0.0074 | 64.7x |
| 0.00 | 0.0273 | 16.5x | 0.0026 | 184.2x |

Released key: accuracy 0.0273, oracle ECE 0.4504. Repaired key: accuracy 0.0026, oracle ECE 0.4788 (ledger F-1; the repaired set is not rebuilt here).

### TSCT seed 0 (n = 3622, accuracy 0.0273)

| confidence | ECE (10 eq-freq) | ECE (15 eq-width) | Brier | reliability | resolution | AUROC |
|---|---|---|---|---|---|---|
| emitted hedge (fixed levels) | 0.4506 | 0.4506 | 0.2363 | 0.2098 | 0.00003 | 0.530 |
| oracle (fixed levels) | 0.4504 | 0.4504 | 0.2360 | 0.2095 | 0.00003 | 0.530 |
| constant [UNKNOWN] = 0.10 | 0.0727 | 0.0727 | 0.0319 | 0.0053 | 0.00000 | 0.500 |
| constant 0.01 | 0.0200 | 0.0173 | 0.0269 | 0.0003 | 0.00000 | 0.500 |
| oracle, levels refit per class (cross-fit) | 0.0087 | 0.0000 | 0.0266 | 0.0000 | 0.00000 | 0.513 |
| constant at base rate (cross-fit) | 0.0107 | 0.0000 | 0.0266 | 0.0000 | 0.00000 | 0.496 |
| exp(mean logprob), raw | 0.5345 | 0.5345 | 0.3354 | 0.3099 | 0.00143 | 0.841 |
| logprob, isotonic (cross-fit) | 0.0086 | 0.0012 | 0.0253 | 0.0000 | 0.00132 | 0.824 |
| logprob, Platt (cross-fit) | 0.0064 | 0.0053 | 0.0254 | 0.0002 | 0.00104 | 0.840 |

Brier(isotonic logprob) - Brier(base-rate constant): 95% CI [-0.0020, -0.0006]

### TSCT seed 1 (n = 3622, accuracy 0.0210)

| confidence | ECE (10 eq-freq) | ECE (15 eq-width) | Brier | reliability | resolution | AUROC |
|---|---|---|---|---|---|---|
| emitted hedge (fixed levels) | 0.4572 | 0.4572 | 0.2365 | 0.2160 | 0.00002 | 0.533 |
| oracle (fixed levels) | 0.4568 | 0.4568 | 0.2359 | 0.2154 | 0.00002 | 0.533 |
| constant [UNKNOWN] = 0.10 | 0.0790 | 0.0790 | 0.0268 | 0.0062 | 0.00000 | 0.500 |
| constant 0.01 | 0.0139 | 0.0110 | 0.0207 | 0.0001 | 0.00000 | 0.500 |
| oracle, levels refit per class (cross-fit) | 0.0043 | 0.0000 | 0.0205 | 0.0000 | 0.00000 | 0.510 |
| constant at base rate (cross-fit) | 0.0086 | 0.0000 | 0.0205 | 0.0000 | 0.00000 | 0.495 |
| exp(mean logprob), raw | 0.5386 | 0.5386 | 0.3366 | 0.3168 | 0.00095 | 0.824 |
| logprob, isotonic (cross-fit) | 0.0042 | 0.0019 | 0.0200 | 0.0002 | 0.00071 | 0.813 |
| logprob, Platt (cross-fit) | 0.0035 | 0.0009 | 0.0198 | 0.0000 | 0.00073 | 0.822 |

Brier(isotonic logprob) - Brier(base-rate constant): 95% CI [-0.0011, -0.0000]

### SFT seed 1 (n = 3622, accuracy 0.0240)

| confidence | ECE (10 eq-freq) | ECE (15 eq-width) | Brier | reliability | resolution | AUROC |
|---|---|---|---|---|---|---|
| emitted hedge (fixed levels) | 0.4551 | 0.4551 | 0.2378 | 0.2144 | 0.00004 | 0.534 |
| oracle (fixed levels) | 0.4537 | 0.4537 | 0.2359 | 0.2125 | 0.00003 | 0.535 |
| constant [UNKNOWN] = 0.10 | 0.0760 | 0.0760 | 0.0292 | 0.0058 | 0.00000 | 0.500 |
| constant 0.01 | 0.0178 | 0.0140 | 0.0236 | 0.0002 | 0.00000 | 0.500 |
| oracle, levels refit per class (cross-fit) | 0.0052 | 0.0000 | 0.0234 | 0.0000 | 0.00000 | 0.510 |
| constant at base rate (cross-fit) | 0.0108 | 0.0000 | 0.0234 | 0.0000 | 0.00000 | 0.493 |
| exp(mean logprob), raw | 0.5245 | 0.5245 | 0.3229 | 0.3005 | 0.00149 | 0.860 |
| logprob, isotonic (cross-fit) | 0.0059 | 0.0048 | 0.0223 | 0.0003 | 0.00081 | 0.850 |
| logprob, Platt (cross-fit) | 0.0062 | 0.0057 | 0.0223 | 0.0003 | 0.00102 | 0.858 |

Brier(isotonic logprob) - Brier(base-rate constant): 95% CI [-0.0019, -0.0005]

### base Qwen2.5-7B (n = 3622, accuracy 0.0392)

| confidence | ECE (10 eq-freq) | ECE (15 eq-width) | Brier | reliability | resolution | AUROC |
|---|---|---|---|---|---|---|
| emitted hedge (fixed levels) | 0.9108 | 0.9108 | 0.8672 | 0.8295 | 0.00000 | 0.500 |
| oracle (fixed levels) | 0.4385 | 0.4385 | 0.2362 | 0.1986 | 0.00006 | 0.529 |
| constant [UNKNOWN] = 0.10 | 0.0629 | 0.0608 | 0.0414 | 0.0037 | 0.00000 | 0.500 |
| constant 0.01 | 0.0325 | 0.0292 | 0.0385 | 0.0009 | 0.00000 | 0.500 |
| oracle, levels refit per class (cross-fit) | 0.0129 | 0.0008 | 0.0376 | 0.0000 | 0.00000 | 0.522 |
| constant at base rate (cross-fit) | 0.0189 | 0.0000 | 0.0377 | 0.0000 | 0.00000 | 0.496 |
| exp(mean logprob), raw | 0.4717 | 0.4717 | 0.2782 | 0.2464 | 0.00590 | 0.881 |
| logprob, isotonic (cross-fit) | 0.0054 | 0.0053 | 0.0318 | 0.0004 | 0.00580 | 0.873 |
| logprob, Platt (cross-fit) | 0.0092 | 0.0137 | 0.0328 | 0.0010 | 0.00552 | 0.879 |

Brier(isotonic logprob) - Brier(base-rate constant): 95% CI [-0.0086, -0.0034]
