# Gemma-4 root cause: timeboxed notes (off the critical path)

Budget: about 2 h. Start 2026-09-28, while the critical-path items were blocked on downloads. The decision to drop all three Gemma-4 rows is already made; nothing here changes it.

## Ruled out (see ../MODELS.md)

- Loader: mlx_lm and mlx_vlm give identical numbers.
- Causal masking.
- Chat formatting of the scored text.
- Missing final-logit softcap.
- Pure logit scale.
- Tokenization: identical and round-trip clean for Gemma-3 and all three Gemma-4 checkpoints (checked 2026-09-28).

## Per-position surprisal (../model_audit/runs/positions_*.json)

| token | gemma-3-4B | gemma-4-E4B | gemma-4-26B | Qwen3.5-27B |
|---|---|---|---|---|
| first after BOS ("The") | 1.7 | **15.0** | **13.8** | n/a (no BOS) |
| content words | 0.1–13.5 | mostly 0.6–2.8, spikes to 13.5 | **17–24 on most content words** | 0.1–9.8 |

Two separate effects:

1. **A first-token penalty after BOS (all Gemma-4).** The instruct models appear to expect a turn marker right after `<bos>`. This matters for PMI in particular. Every matched-set claim starts "The …", so the penalty is roughly constant across claims. But the neutral frame "<Name> is a person." starts with the person's name, so the penalty becomes name-dependent noise in the subtraction. That is consistent with Gemma-4 losing more truth AUROC from raw to PMI than other models do (known-answer raw 0.92–0.97 vs PMI 0.61–0.94).
2. **Pervasive content-word surprisal in 26B-A4B only** (4-bit MoE). It is not explained by (1). Candidate: the 4-bit quantization of this MoE checkpoint.

## Next tests (once the GPU is free, inside the timebox)

- Score the matched set and known-answer pairs excluding the first scored token (condition on BOS + first token), and with a neutral prefix. Does Gemma-4 truth AUROC on PMI recover?
- 26B: compare against the bf16 reference (`FETCH_ALL.sh --optional`, E4B only, so this can only separate implementation from checkpoint for E4B).

## Test 1: first-token-after-BOS (done, 2026-09-28)

`first_token_test.py` via `run_first_token.sh`; results in `first_token.log` and `runs/first_token_*.json`. The standard-mode numbers reproduce the stored Aug 13 values exactly.

| model | truth PMI: standard → skip first token | regime PMI | staleness PMI | SD of first-token NLL in the name frame |
|---|---|---|---|---|
| gemma-3-4B (control) | 0.719 → 0.739 | 0.620 → 0.615 | 0.454 → 0.445 | 3.52 |
| Qwen3.5-27B (control) | 0.845 → 0.866 | 0.521 → 0.552 | 0.415 → 0.406 | 2.69 |
| gemma-4-E4B | 0.738 → 0.743 | 0.601 → 0.592 | 0.498 → 0.498 | 2.48 |
| gemma-4-26B-A4B | 0.584 → 0.580 | 0.373 → 0.377 | 0.548 → 0.545 | 2.79 |

**Refuted.** Dropping the first scored token moves Gemma-4 no more than the controls, and the first-token NLL is no noisier for Gemma-4 than for the controls. The first-token penalty is real (13–15 nats), but it is roughly constant and does not explain the degraded PMI.

The 31B run was stopped: it was I/O-starved while FETCH_ALL.sh saturated the disk (2 min of CPU in 13 min). Deferred until the fetch finishes.

## Status of the timebox

About 55 of 120 minutes used (tokenizer check, per-position analysis, test 1). The remaining hypothesis needs the bf16 reference:
- **E4B:** compare PyTorch/transformers log-probs against mlx on the same paragraph and the 36 known-answer pairs. This separates the implementation or conversion from the model itself.
- **26B:** its pervasive content-word surprisal has no reference in the fetch (only E4B is in `--optional`), so it stays unexplained.
timebox start 2026-09-29T01:55:38Z (2 h, Directive 3)

## 31B design (Directive 3): result, 2026-09-29, about 40 of 120 minutes

Runners: `run_31b.sh` (G1, G2, G4; `run_31b.log`), `mlx_chunk_ppl.py` and `llama-perplexity` (G3; `g3_mlx.out`, `g3_llamacpp.log`). Checkpoints: FETCH_EXTRA.sh `--gemma` plus the local LM Studio 8-bit.

| test | checkpoint | plain-text NLL (audit paragraph) | best T | known-answer raw / PMI | first token after BOS | matched set: truth / regime / staleness (PMI) |
|---|---|---|---|---|---|---|
| G1 | 31B-it, LM Studio 8-bit | 4.73 | 1.5 | 0.97 / 0.83 | 10.0 | 0.669 / 0.554 / 0.536 |
| G1 | 31B-it, mlx-community 4-bit | 3.96 | 1.5 | 0.97 / 0.78 | 10.1 | see `runs/first_token_g4_31b_it_4bit.json` |
| **G2** | **31B pretrained, mlx-community 4-bit** | **2.19** | **1** | 0.94 / 0.86 | **3.7** | **0.831 / 0.651 / 0.427** |

**G3, independent implementation.** Perplexity on `plain_text.txt` (601 tokens; 4 chunks of 128, second half scored, BOS per chunk):

| | 31B-it | 31B pretrained |
|---|---|---|
| mlx 8-bit | **201.1** | 5.0 (4-bit) |
| llama.cpp build 9430, Q8_0 GGUF | **208.1 ± 73.4** | not run (no GGUF) |

## Root cause

**The anomaly is a property of the instruction-tuned Gemma-4 checkpoints on raw (non-chat) text.** It is not the harness, the mlx implementation or quantization.
- **Not quantization or conversion (G1):** 4-bit (mlx-community) is no better than 8-bit (LM Studio).
- **The pretrained model is normal (G2):** same architecture, same mlx code, same converter and bits, with fluency and contrasts in the healthy range. Its staleness PMI of 0.427 is below chance, like every admitted model.
- **An independent implementation reproduces the instruct anomaly (G3):** llama.cpp 208 against mlx 201.
- **Chat formatting does not rescue it either** (test in `../model_audit/runs/diag_*`, 26B: 10.3). The instruct models assign extreme surprisal to ordinary content words in any teacher-forced frame the harness uses.

G5 (the E4B bf16 PyTorch reference) was skipped under the stopping rule, because G1–G3 locate the cause.

**For the footnote:** "Gemma-4 instruction-tuned checkpoints assign anomalously low probability to ordinary text: perplexity about 200 on plain English, against 5 for the pretrained Gemma-4-31B. We confirmed this in two independent implementations (MLX and llama.cpp) and at two quantizations, so their teacher-forced log-probabilities are not comparable with the other models' and they are excluded. The pretrained 31B behaves normally."

## Exclusion unchanged

Edward's decision stands: all three Gemma-4 instruct rows are dropped. The pretrained 31B is not added, since Directive 3 adds no new models. Its row is recorded here only as the diagnostic control.

timebox end 2026-09-29, about 40 minutes used
