| tag | repo id | revision | converter | quant | loader | BOS prepended | known-answer raw / PMI (n=36) | same, no BOS | NLL/token (T=1) | best T | generation 5/5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g3_4b | mlx-community/gemma-3-4b-it-4bit | 93724907d4 | mlx-vlm 0.1.18 | 4-bit affine g64 | mlx_lm | 2 | 0.92 / 0.78 | 0.92 / 0.78 | 2.68 | 1 | 5 |
| g4_e4b | lmstudio-community/gemma-4-E4B-it-MLX-4bit | fa6f15978b | LM Studio (mlx_vlm) | 4-bit affine g64 | mlx_vlm (fallback) | 2 | 0.97 / 0.94 | 0.69 / 0.58 | 4.71 | 1.5 | 5 |
| qwen3_4b_th | lmstudio-community/Qwen3-4B-Thinking-2507-MLX-8bit | e2e1d452c2 | LM Studio (mlx_lm) | 8-bit affine g64 | mlx_lm | None | 0.97 / 0.81 | n/a (no BOS) | 2.89 |  | 5 |
| qwen3_4b_it | lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit | 8761816394 | LM Studio (mlx_lm) | 8-bit affine g64 | mlx_lm | None | 0.94 / 0.92 | n/a (no BOS) | 2.61 |  | 5 |
| gptoss_20b | unverified: openai-gpt-oss-20b-MLX-6.5bit | b9715ed9ab | None | 6-bit affine g64 | mlx_lm | 199998 | 0.89 / 0.69 | 0.97 / 0.86 | 2.95 | 1 | 5 |
| qwen36_35b | mlx-community/Qwen3.6-35B-A3B-nvfp4 | 9c1a3a223d | mlx-vlm 0.4.4 | 4-bit nvfp4 g16 | mlx_lm | None | 0.97 / 0.86 | n/a (no BOS) | 2.49 |  | 5 |
| qwen35_27b | mlx-community/Qwen3.5-27B-4bit | 45797d2985 | mlx-vlm 0.3.12 | 4-bit affine g64 | mlx_lm | None | 0.92 / 0.92 | n/a (no BOS) | 2.31 | 1 | 5 |
| qwen36_27b | mlx-community/Qwen3.6-27B-4bit | c000ac2c20 | mlx-vlm 0.4.4 | 4-bit affine g64 | mlx_lm | None | 0.94 / 0.94 | n/a (no BOS) | 2.18 |  | 5 |
| qwen38_27b | mlx-community/Qwen3.8-27B-4bit | 3e6447f082 | mlx-vlm 0.6.8 | 4-bit affine g64 | mlx_lm | None | 0.94 / 0.92 | n/a (no BOS) | 2.07 |  | 5 |
| g4_31b | lmstudio-community/gemma-4-31B-it-MLX-8bit | 244e29d3b1 | LM Studio (mlx_vlm) | 8-bit affine g64 | mlx_lm | 2 | 0.97 / 0.83 | 0.69 / 0.61 | 4.73 | 1.5 | 5 |
| gemma4_26b | mlx-community/gemma-4-26b-a4b-it-4bit | None | mlx-vlm 0.4.3 | 4-bit affine g64 | mlx_lm | 2 | 0.92 / 0.61 | 0.58 / 0.58 | 10.94 | 2 | 5 |
| gemma4_26b_vlm | mlx-community/gemma-4-26b-a4b-it-4bit | None | mlx-vlm 0.4.3 | 4-bit affine g64 | mlx_vlm (forced) | 2 | 0.92 / 0.61 | 0.58 / 0.58 | 10.94 |  | 5 |
