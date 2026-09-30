# Model audit (2026-09-28)

Scope: every checkpoint the paper uses or will use. Per-model records are in `model_audit/runs/*.json`, and the table below is generated from them (`model_audit/build_table.py` → `model_audit/models_table.md`).

| script | what it checks |
|---|---|
| `model_audit/audit_models.py` | metadata, loader, tokenizer, known-answer set, fluency, generation |
| `model_audit/diag_format.py` | raw vs chat-template NLL, and which BOS setting reproduces the stored Aug 13 scores |
| `model_audit/diag_positions.py` | full-sequence vs prefix-only logits |
| `model_audit/diag_temperature.py` | the NLL-minimizing temperature |

The driver is `model_audit/run_audit.sh`, with wall-clock times in `model_audit/audit.log`.

## How the harness runs a model (all rows)

- **Matched set (Table 4, Figure 3, item 7):** each claim is scored as **raw text with no chat template**. The score is the mean per-token log-probability of `hf_tok(text, add_special_tokens=True)`, with the tokenizer's declared BOS prepended if it is missing. PMI is that score minus the same person in the frame "`<name> is a person.`" (`src/training/score_matched_regime.py`).
- **Loading:** `load_base_only` tries mlx_lm first and falls back to mlx_vlm. Only gemma-4-E4B takes the fallback.
- **Generation-based sets** (the temporal-delta test set, TSCT arms): these use the chat template (`generate_predictions.generate_one`). They are not part of the matched-set comparison.
- **Library versions today:** mlx 0.31.2, mlx_lm 0.31.3, mlx_vlm 0.5.0, transformers 5.8.1.

## Sanity checks

- **Known-answer set.** 36 true/false pairs in the matched set's own frame, e.g. "The founder of Microsoft is Bill Gates." against "… is Larry Page." Scored by the scorer's own `claim_logprob`. Reported as pairwise accuracy on raw mean log-prob and on PMI.
- **Fluency.** Mean NLL per token on a plain four-sentence English paragraph.
- **Generation.** Five capital-city questions, answered greedily through the chat template.

## Table

| tag | repo id | revision | converter | quant | loader | BOS prepended | known-answer raw / PMI (n=36) | same, no BOS | NLL/token (T=1) | best T | generation 5/5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g3_4b | mlx-community/gemma-3-4b-it-4bit | 93724907d4 | mlx-vlm 0.1.18 | 4-bit affine g64 | mlx_lm | 2 | 0.92 / 0.78 | 0.92 / 0.78 | 2.68 | 1 | 5 |
| g4_e4b | lmstudio-community/gemma-4-E4B-it-MLX-4bit | fa6f15978b | LM Studio (mlx_vlm) | 4-bit affine g64 | mlx_vlm (fallback) | 2 | 0.97 / 0.94 | 0.69 / 0.58 | **4.71** | 1.5 | 5 |
| qwen3_4b_th | lmstudio-community/Qwen3-4B-Thinking-2507-MLX-8bit | e2e1d452c2 | LM Studio (mlx_lm) | 8-bit affine g64 | mlx_lm | none | 0.97 / 0.81 | n/a | 2.89 | | 5 |
| qwen3_4b_it | lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit | 8761816394 | LM Studio (mlx_lm) | 8-bit affine g64 | mlx_lm | none | 0.94 / 0.92 | n/a | 2.61 | | 5 |
| gptoss_20b | publisher unverified (dir `openai-gpt-oss-20b-MLX-6.5bit`; base `openai/gpt-oss-20b`) | b9715ed9ab | not stated | 6-bit affine g64 (dir says 6.5) | mlx_lm | **199998 (today)** | 0.89 / 0.69 | 0.97 / 0.86 | 2.95 | 1 | 5 |
| qwen36_35b | mlx-community/Qwen3.6-35B-A3B-nvfp4 | 9c1a3a223d | mlx-vlm 0.4.4 | 4-bit nvfp4 g16 | mlx_lm | none | 0.97 / 0.86 | n/a | 2.49 | | 5 |
| qwen35_27b | mlx-community/Qwen3.5-27B-4bit | 45797d2985 | mlx-vlm 0.3.12 | 4-bit affine g64 | mlx_lm | none | 0.92 / 0.92 | n/a | 2.31 | 1 | 5 |
| qwen36_27b | mlx-community/Qwen3.6-27B-4bit | c000ac2c20 | mlx-vlm 0.4.4 | 4-bit affine g64 | mlx_lm | none | 0.94 / 0.94 | n/a | 2.18 | | 5 |
| qwen38_27b | mlx-community/Qwen3.8-27B-4bit | 3e6447f082 | mlx-vlm 0.6.8 | 4-bit affine g64 | mlx_lm | none | 0.94 / 0.92 | n/a | 2.07 | | 5 |
| g4_31b | lmstudio-community/gemma-4-31B-it-MLX-8bit | 244e29d3b1 | LM Studio (mlx_vlm) | 8-bit affine g64 | mlx_lm | 2 | 0.97 / 0.83 | 0.69 / 0.61 | **4.73** | 1.5 | 5 |
| gemma4_26b | mlx-community/gemma-4-26b-a4b-it-4bit | not recorded | mlx-vlm 0.4.3 | 4-bit affine g64 | mlx_lm | 2 | 0.92 / 0.61 | 0.58 / 0.58 | **10.94** | 2 | 5 |
| gemma4_26b (forced mlx_vlm) | same | | | | mlx_vlm | 2 | 0.92 / 0.61 | 0.58 / 0.58 | **10.94** | | 5 |
| qwen25_7b, tsct | mlx-community/Qwen2.5-7B-Instruct-4bit (+ LoRA `tcl_mlx_7b/tsct_seed1/adapter_fixed`) | — | — | 4-bit | — | — | **not audited: the base checkpoint is no longer on disk and the download was blocked** | | | | |

A blank "best T" means the temperature fit was not run for that model. Revision SHAs come from `.cache/huggingface/download/*.metadata`; the 26B checkpoint has none.

## Findings

1. **The Gemma-4 family's log-probabilities are anomalous under this harness. This includes gemma-4-E4B, which is one of the ten admitted rows.**
   - On plain English, all three Gemma-4 checkpoints score 4.7–10.9 nats/token. Every other model scores 2.1–2.9, including gemma-3-4B at 2.68.
   - Ordinary mid-sentence words get extreme surprisal: gemma-4-26B gives " flows" −25.7 and " water" −25.3; E4B gives " river" −20.9 (`runs/diag_*.json`).
   - Things ruled out:
     - **Loader:** mlx_lm and mlx_vlm give identical numbers for 26B.
     - **Causal masking:** full-sequence and prefix-only logits agree within 0.44 nats for E4B. The largest gap is 3.8 for 26B, where both give NLL about 11.3.
     - **Chat formatting:** the same paragraph as an assistant turn still gives 10.3 for 26B and 4.35 for E4B.
     - **Missing softcap:** final-logit softcapping is applied in both implementations.
   - Pure logit scale is not the cause either: the best temperature (1.5–2) still leaves NLL at 3.7–7.5.
   - Greedy generation is correct 5/5 on all three, and known-answer raw accuracy is 0.92–0.97. So the argmax is right while the probabilities are not. Teacher-forced log-probs and PMI, which is what the matched set uses, are exactly what is affected.
   - **Root cause (2026-09-29, `gemma4/NOTES.md`):** the instruction-tuned Gemma-4 checkpoints themselves. The pretrained Gemma-4-31B, with the same architecture, mlx code, converter and bits, is normal: NLL 2.19, best T = 1, staleness PMI 0.427. llama.cpp reproduces the instruct anomaly independently (PPL 208 against mlx's 201 on the same text; pretrained 5.0). Quantization is not the cause (4-bit is no better than 8-bit).
   - *Earlier note, superseded:* the cause was not identified. Candidates: an mlx gemma-4 implementation detail common to both libraries, or the conversions. A reference check against the original bf16 weights in PyTorch would settle it, but it needs a download.
   - **Consequence.** The Sep 2 statement that both exclusions "traced to a missing beginning-of-sequence token" is incomplete. BOS mattered: without it known-answer accuracy falls to 0.58–0.69. But after the BOS fix the Gemma-4 rows are still anomalous, and the same anomaly is present in the admitted E4B row.
   - **Impact on the headline.** Dropping E4B does not change the inverse result: the leave-one-model-out slopes span −0.99 to −0.80 (`inverse/`). E4B has the staleness value closest to chance (0.498), so dropping it strengthens "every admitted model below chance".
   - **Decision for Edward:** exclude all Gemma-4 rows, or keep E4B with a footnote.
2. **gpt-oss-20B: BOS handling has drifted since Aug 13.**
   - The stored Aug 13 scores (`matched_gptoss_20b.jsonl`) reproduce exactly **without** BOS (max difference 0.0) and differ by up to 1.05 nats with it. So the Sep 2 sentence "gpt-oss declares no BOS and is unaffected" was true on Aug 13.
   - With transformers 5.8.1 the tokenizer now declares `<|startoftext|>` (199998), and the scorer prepends it. That lowers known-answer PMI accuracy from 0.86 to 0.69.
   - **Harness fix:** `score_matched_regime.py` now takes `--bos off`, and every gpt-oss rescore uses it.
3. **gemma-3-4B:** today's tokenizer adds BOS itself, so the explicit prepend is redundant and harmless. The stored scores reproduce exactly (`runs/diag_g3_4b.json`).
4. **Qwen rows, including Qwen3.8-27B, are clean.** NLL 2.1–2.9, known-answer accuracy 0.92–0.97, stored scores reproduce exactly (Qwen3.5-27B checked). Qwen3.8 loads through mlx_lm and needs none of the mlx-vlm 0.6.8 features.
5. **Quantization claims.**
   - The "6.5-bit" gpt-oss checkpoint's config says 6 bits, g64. The paper's "quantized between 4 and 8 bits" still holds.
   - The vintage pair Qwen3.5-27B and Qwen3.6-27B are both 4-bit affine g64, so the recipe matches. They were converted with mlx-vlm 0.3.12 and 0.4.4 respectively; Qwen3.8 with 0.6.8.
6. **"Ten admitted of twelve" reproduces** from the stored files (truth AUROC ≥ 0.70 for exactly ten; `inverse/`). Given finding 1, three of the twelve rows (E4B, 26B, 31B) rest on anomalous log-probs.
7. **Not audited:** Qwen2.5-7B-Instruct-4bit and the TSCT adapter built on it. The base checkpoint is gone from the HF cache, and the auto-mode classifier blocked the re-download. The item 4 4-bit replacements have not been downloaded for the same reason.

## What this changes for the approved runs

- gpt-oss rescores (A6, item 4) use `--bos off`, which is the Aug 13 behaviour.
- Gemma-4 rows are scored for completeness but flagged, pending Edward's decision.
- The Qwen2.5-7B and TSCT rows cannot be rescored (A6), retrained (item 5), or re-audited until the base checkpoint is back.
