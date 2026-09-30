### LLaMA-3-8B, 4 paired seeds [0, 1, 2, 3] (dryrun_seeds0-3)

TemporalDelta test (3,622 per arm-seed):

| endpoint | TSCT mean | plain FT mean | Δ (TSCT − FT) | 95% t-interval | sign-consistent | FT seed SD |
|---|---|---|---|---|---|---|
| auroc_conf_correct | 0.5269 | 0.5298 | -0.0029 | [-0.0276, +0.0218] | no | 0.0152 |
| resolution | 0.0000 | 0.0000 | -0.0000 | [-0.0001, +0.0001] | no | 0.0000 |
| ece_by_level | 0.4527 | 0.4548 | -0.0021 | [-0.0065, +0.0024] | no | 0.0040 |
| reliability | 0.2121 | 0.2140 | -0.0020 | [-0.0062, +0.0023] | yes | 0.0038 |
| brier | 0.2374 | 0.2371 | +0.0002 | [-0.0028, +0.0032] | no | 0.0014 |
| aurc | 0.9515 | 0.9544 | -0.0029 | [-0.0112, +0.0054] | no | 0.0082 |
| selective_risk_at_50 | 0.9498 | 0.9536 | -0.0039 | [-0.0141, +0.0063] | no | 0.0087 |
| accuracy | 0.0260 | 0.0237 | +0.0023 | [-0.0032, +0.0077] | no | 0.0044 |
| hedge_label_accuracy | 0.9970 | 0.9976 | -0.0006 | [-0.0066, +0.0055] | no | 0.0022 |

Effect by the prespecified rule: **False**

Controls on the same correctness (range over arm-seeds): oracle ECE 0.4471–0.4592; constant [UNKNOWN] 0.0694–0.0815; oracle / constant 5.63–6.45×.

FreshQA (current key): ΔAUROC -0.0501 [-0.1810, +0.0808], ΔECE -0.0363 [-0.0822, +0.0095], Δresolution +0.0001; effect by rule: False

FreshQA (aged key): ΔAUROC -0.0511 [-0.1655, +0.0633], ΔECE -0.0332 [-0.0851, +0.0187], Δresolution +0.0005; effect by rule: False

Sanity: identical prediction files []; identical adapters []; all 3,622 rows: True; tokenizer round-trip: True; compute-matched: True
