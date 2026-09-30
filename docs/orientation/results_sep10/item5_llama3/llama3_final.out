### LLaMA-3-8B, 5 paired seeds [0, 1, 2, 3, 4] (final)

TemporalDelta test (3,622 per arm-seed):

| endpoint | TSCT mean | plain FT mean | Δ (TSCT − FT) | 95% t-interval | sign-consistent | FT seed SD |
|---|---|---|---|---|---|---|
| auroc_conf_correct | 0.5270 | 0.5293 | -0.0024 | [-0.0191, +0.0144] | no | 0.0132 |
| resolution | 0.0000 | 0.0000 | -0.0000 | [-0.0000, +0.0000] | no | 0.0000 |
| ece_by_level | 0.4537 | 0.4551 | -0.0014 | [-0.0050, +0.0022] | no | 0.0035 |
| reliability | 0.2130 | 0.2143 | -0.0013 | [-0.0047, +0.0022] | no | 0.0033 |
| brier | 0.2373 | 0.2372 | +0.0002 | [-0.0019, +0.0022] | no | 0.0012 |
| aurc | 0.9533 | 0.9552 | -0.0019 | [-0.0082, +0.0044] | no | 0.0073 |
| selective_risk_at_50 | 0.9515 | 0.9541 | -0.0025 | [-0.0104, +0.0053] | no | 0.0076 |
| accuracy | 0.0250 | 0.0235 | +0.0015 | [-0.0028, +0.0058] | no | 0.0038 |
| hedge_label_accuracy | 0.9969 | 0.9976 | -0.0007 | [-0.0047, +0.0034] | no | 0.0019 |

Effect by the prespecified rule: **False**

Controls on the same correctness (range over arm-seeds): oracle ECE 0.4471–0.4592; constant [UNKNOWN] 0.0694–0.0815; oracle / constant 5.63–6.45×.

FreshQA (current key): ΔAUROC -0.0576 [-0.1484, +0.0333], ΔECE -0.0442 [-0.0820, -0.0064], Δresolution +0.0004; effect by rule: False

FreshQA (aged key): ΔAUROC -0.0497 [-0.1271, +0.0277], ΔECE -0.0412 [-0.0826, +0.0003], Δresolution -0.0002; effect by rule: False

Sanity: identical prediction files []; identical adapters []; all 3,622 rows: True; tokenizer round-trip: True; compute-matched: True
