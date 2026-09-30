# Item 5 on LLaMA-3-8B: five paired TSCT seeds (final)

**Script:** `llama3_analysis.py` → `llama3_results_final.json`, `llama3_table_final.md`, `llama3_final.out`. The same endpoint function (`../item5/seed_variance.py: metrics`) and 95% t-intervals over seeds as the Qwen five-seed result (`../item5/seed_variance_routeB.json`). The endpoints were prespecified in `../item5/ENDPOINTS.md` and not changed.

**Runs** (`llama3_seeds.log`; one commit per seed pair: `a2574c5`, `4cd6a1a`, `4b3a413`, `1643cd3` and the seed-0 pair):
- **Base:** `meta-llama/Meta-Llama-3-8B`, the base in all 49 team adapter configs, via `mlx-community/Meta-Llama-3-8B-4bit` @ 7f296f1d.
- **Tokenizer:** declared as `PreTrainedTokenizerFast` (the `-tokfix` directory; weights symlinked). The checkpoint's own `LlamaTokenizer` declaration drops spaces under transformers 5.8.1. The first seed-0 run was trained that way and was discarded (`data/prep/*_llama3_invalid_spaceless/`).
- **Arms:** TSCT (λ 0.5 / 0.5 / 0.3) and plain fine-tuning (λ 0), each with 8,008 examples and 6,006 steps, and identical LoRA, learning rate and schedule. Seeds 0–4, paired.
- **Deviations from the team's setup:**
  - Hedges ride on four reserved special tokens (128002–128005), because LLaMA-3 has no spare embedding rows.
  - Prompts use a plain "Question: … Answer:" template, because the base ships no chat template.
  - The team's own hedge mechanism cannot be reconstructed: their tokenizer has no hedge tokens and their training code is deleted.
- **Wall time per arm-seed:** training 3,226–4,523 s, TemporalDelta eval 781–834 s, FreshQA eval 78–87 s.

**Sanity checks (all pass):**
- no byte-identical prediction files or adapters across the 10 arm-seeds
- 3,622 test predictions per arm-seed
- the tokenizer round-trips "Mark Parker and Susan Wojcicki"
- all runs compute-matched

## Result: no TSCT benefit, agreeing with Qwen2.5-7B

**TemporalDelta test, TSCT − plain fine-tuning, 5 paired seeds:**

| endpoint | TSCT mean | plain FT mean | Δ | 95% t-interval | per-seed Δ (s0 … s4) | FT seed SD |
|---|---|---|---|---|---|---|
| **AUROC (primary)** | 0.5270 | 0.5293 | **−0.0024** | [−0.0191, +0.0144] | −0.017, −0.015, +0.014, +0.006, −0.000 | 0.0132 |
| ECE (per level) | 0.4537 | 0.4551 | −0.0014 | [−0.0050, +0.0022] | −0.001, −0.002, −0.006, +0.001, +0.002 | 0.0035 |
| resolution | 0.00003 | 0.00003 | −0.0000 | [−0.0000, +0.0000] | | 0.0000 |
| reliability | 0.2130 | 0.2143 | −0.0013 | [−0.0047, +0.0022] | | 0.0033 |
| Brier | 0.2373 | 0.2372 | +0.0002 | [−0.0019, +0.0022] | | 0.0012 |
| AURC | 0.9533 | 0.9552 | −0.0019 | [−0.0082, +0.0044] | | 0.0073 |
| selective risk at 50% | 0.9515 | 0.9541 | −0.0025 | [−0.0104, +0.0053] | | 0.0076 |
| accuracy | 0.0250 | 0.0235 | +0.0015 | [−0.0028, +0.0058] | | 0.0038 |
| hedge-label accuracy | 0.9969 | 0.9976 | −0.0007 | [−0.0047, +0.0034] | | 0.0019 |

**Effect by the prespecified rule: no.** No endpoint is sign-consistent or excludes zero.

**Controls on the same LLaMA-3 correctness** (per arm-seed):
- oracle ECE 0.447–0.459
- constant [UNKNOWN] ECE 0.069–0.082
- **oracle / constant 5.6–6.5×**, which reproduces the paper's 6.2× on a second base model

**Qwen2.5-7B, for comparison:** ΔAUROC −0.0003 [−0.0219, +0.0212]; ΔECE −0.0001 [−0.0033, +0.0031].

**FreshQA (413; containment scoring):**

| key | ΔAUROC (primary) [95% CI] | ΔECE [95% CI] | Δresolution | effect by rule |
|---|---|---|---|---|
| current (2026-04) | −0.058 [−0.148, +0.033] | **−0.044 [−0.082, −0.006]** | +0.0004 | no |
| aged (2024-02) | −0.050 [−0.127, +0.028] | −0.041 [−0.083, +0.000] | −0.0002 | no |

## The one surprise, and why it is not a TSCT benefit

On FreshQA under the current key, TSCT's ECE is lower than plain fine-tuning's, and the sign agrees in all 5 seeds. The CI excludes zero. Qwen showed nothing comparable: ΔECE −0.007 [−0.027, +0.014]. The mechanism, however, is a level shift and not better discrimination:
- For seeds 2–4, TSCT moves hedge mass from [CONFIDENT] (0.95) to [COND_CONFIDENT] (0.75). Seed 2, for example, has 218 [CONFIDENT] and 178 [COND_CONFIDENT] answers, against 377–395 [CONFIDENT] for every plain-FT seed.
- Stating lower confidence against about 15% accuracy lowers ECE. The whole change is in reliability (−0.057), while resolution does not move (+0.0004).
- Discrimination gets worse if anything: ΔAUROC −0.058, ΔAURC +0.070, hedge-label accuracy −0.049, all with CIs spanning zero.

This is the paper's own mechanism, calibration without resolution, appearing inside the TSCT audit: the ECE gain comes from moving confidence toward the base rate. Per `ENDPOINTS.md`, ECE is reported and not interpreted alone.

## Summary

On the team's base model (Meta-Llama-3-8B), five paired, compute-matched seeds show no TSCT effect on the primary endpoint, AUROC of stated confidence for correctness, on either benchmark. There is also no effect on resolution or selective risk. This matches the Qwen2.5-7B result. The constant-policy control beats the volatility-label oracle 5.6–6.5× on these predictions, as it does on Qwen. The only difference, lower ECE on FreshQA, comes from TSCT asserting less confidence out of distribution, which is exactly the kind of ECE change the paper argues does not reflect item-level resolution.
