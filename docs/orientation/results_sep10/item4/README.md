# Item 4: the fixed 4-bit ladder, with tokenizer, prompt and answer-length controls

**Runner:** `run_4bit_ladder.sh` (log with wall times: `item4.log`, 2026-09-28 09:4x–10:40 UTC). **Analysis:** `item4_analysis.py` → `item4_table.md`, `item4_controls.json`, `item4_inverse.json`, `inverse_*.out`, `item4_analysis.out`.

**Checkpoints:** every admitted row at MLX 4-bit affine g64. gpt-oss uses its native MXFP4-experts + Q4 checkpoint (`mlx-community/gpt-oss-20b-MXFP4-Q4`), scored with `--bos off`. The four new checkpoints were audited first (`../model_audit/runs/*_4bit.json`):
- known-answer raw accuracy 0.92–0.97, 5/5 generations
- NLL 2.32–3.03

Rows already at 4-bit g64 reuse their outputs (exact reproduction shown in `../a6_disjoint/unchanged_arms.json`).

## Result: the inverse truth-discrimination result does not survive fixed quantization

| claim set, models | slope | permutation p | Spearman | crossed random-effects model, paired win p |
|---|---|---|---|---|
| disjoint, Aug mixed quantization, 9 | −1.367 | 0.013 | −0.667 | 0.007 |
| **disjoint, 4-bit, all 9** | **−0.268** [−0.65, +0.10] | **0.40** | −0.400 | 0.62 |
| **disjoint, 4-bit, admitted (truth ≥ 0.70; excludes gpt-oss) 8** | −0.729 | **0.15** | −0.429 | |
| original set, 4-bit, 9 | −0.571 [−1.08, −0.05] | 0.099 | −0.600 | 0.17 |
| original set, 4-bit, without gpt-oss, 8 | −0.729 | 0.076 | −0.643 | |
| disjoint, 4-bit, chat frame (prompt control), 9 | −0.423 [−0.87, +0.13] | 0.42 | −0.167 | 0.50 |

The direction stays negative in every variant, and the split-half control is negative in 94–99.9% of splits. Significance, however, is lost at fixed quantization. The Sep 2 paper already says the ladder is "descriptive … until all ten admitted checkpoints are rerun at one fixed quantization". This is that rerun, and it does not confirm the ordering.

**What moves (disjoint set, 4-bit minus Aug, entity-paired 95% CI):**
- **gpt-oss-20B (6.5-bit → MXFP4-Q4):**
  - truth **−0.134** [−0.184, −0.088], which drops it below the 0.70 admission bar
  - staleness +0.044 [+0.007, +0.083]
- **Qwen3.6-35B-A3B (nvfp4 → 4-bit):** regime +0.048 [+0.016, +0.081], truth +0.019
- **Qwen3-4B-Instruct (8-bit → 4-bit):** regime −0.044 [−0.080, −0.006]
- **Qwen3-4B-Thinking (8-bit → 4-bit):** regime −0.021, truth −0.013, both within noise

All five rows that were already 4-bit g64 are unchanged by construction.

## The staleness result survives

Mean staleness AUROC over the nine at 4-bit is 0.442 [0.404, 0.477], against 0.439 [0.399, 0.476] at mixed quantization. With the chat frame it is 0.460 [0.424, 0.494]. It excludes 0.5 in all three.

## Controls (per model, 4-bit; `item4_table.md`)

- **Tokenizer.** Regressing PMI on the claim's and the name's token counts raises every AUROC by 0.01–0.05 and changes no ordering. Token length is not driving any contrast.
- **Answer length.** On pairs whose two names have equal token counts (21–29 per model):
  - truth paired win rates rise (0.84–1.00)
  - staleness paired win rates stay at or below 0.50 for 8 of 9 models (gpt-oss 0.48; gemma-3 exactly 0.50)
  - equal-length regime wins vary widely (0.42–0.83), consistent with regime being the weakest contrast
- **Prompt.** Scoring each claim as an assistant turn (`--frame chat`) lowers truth AUROC for most models and flattens the inverse relationship (above). Staleness stays below chance.

## Reading

- **Robust across quantization, prompt format and length controls:** the familiarity (staleness) result.
- **Not robust to fixed quantization:** the inverse truth–regime relationship. It is significant only at the mixed quantization of the original ladder.
- The paper cannot make it central without this caveat. At minimum it must be reported with the fixed-quantization table, where p = 0.08–0.40 depending on the claim set and the admission rule.
