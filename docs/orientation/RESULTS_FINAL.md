# Sep 10 round: final numbers (scope freeze in effect)

Branch `altj-sep10-revisions`, 2026-09-29. Paths are relative to `docs/orientation/results_sep10/`; each folder holds the script that produced its file. Framing per Edward (REFRAME.md, reframe worktree `77d91f2`): **the staleness/familiarity finding leads; the inverse result is secondary and direction-only.**

Admitted set: **nine models.** All three Gemma-4 instruct checkpoints are dropped. CIs are entity bootstraps unless noted.

| # | Result | Final numbers | Source |
|---|---|---|---|
| 1 | **Constant beats oracle, TemporalDelta** | oracle ECE 0.4504 vs constant [UNKNOWN] 0.0727, **6.20× [5.85, 6.62]**. Binned ECE equals per-level ECE on every row. Crossover at volatile accuracy **0.314**; the test set sits at 0.026. | `ece_check/ece_levels.json`, `crossover/crossover.json` |
| 2 | Table 1 Brier / resolution (corrected) | Brier 0.2360 / 0.2363 / 0.2368 / 0.2052 / 0.0319 (oracle / TSCT / CE / constant TH / constant UNK); resolution ≤ 3e-5. The Sep 2 column does not reproduce. | `ece_check/ece_levels.json` |
| 3 | Table 2, repaired key | 1,849 items (Sep 2's 1,953 not reproducible); accuracy 0.27% / 0.49%; ratio **4.84× [4.75, 4.96]**; the Brier column reproduces | `table2/table2_full.json` |
| 4 | Confidence vocabulary | A 0.01 level raises the constant's win to 26× (repaired key 65×). A recalibrated log-prob score (AUROC 0.82–0.88) ties a base-rate constant on ECE (about 0.01); only Brier and resolution separate them. | `vocabulary/vocab.json` |
| 5 | **Constant beats oracle, FreshQA** | **7.2–9.5×** (CIs 4.5–23) under both keys; hedge resolution ≤ 0.006. The oracle has real resolution (0.03–0.05) yet still loses on ECE at the 2026 key. | `item1/freshqa_results.json`, `item1/threshold.json` |
| 6 | Threshold figure (main) | The three strongest models cross from oracle-wins to constant-wins as the key moves from 2024 to 2026 (fast accuracy 0.19–0.24 → 0.09–0.11) | `item1/threshold.png`, `item1/threshold_table.md` |
| 7 | **Staleness (lead finding), matched set** | mean AUROC over 9 models **0.439 [0.400, 0.476]**, p = 0.001; paired win 0.426 [0.369, 0.482]; z-scored PMI difference −0.21 [−0.34, −0.09] | `a5_prominence/staleness_estimands_nine.json` |
| 8 | Staleness vs prominence and time | The stale holder is more prominent on 70% of entities, but the prominence slope is null (p = 0.47–0.59). At equal prominence: p = 0.028 (z-score outcome) / 0.054 (win-rate outcome). No trend with years since the change (ρ = 0.005). | `a5_prominence/a5_nine.json` |
| 9 | Staleness at fixed 4-bit | 0.442 [0.404, 0.477]; chat-frame prompt control 0.460 [0.424, 0.494] | `item4/item4_inverse.json` |
| 10 | **Base vs instruct (Qwen2.5-7B)** | staleness 0.428 vs 0.426, **Δ +0.003 [−0.022, +0.028]**: a pretraining property. Post-training raises regime by 0.051. | `base_instruct/base_vs_instruct.json` |
| 11 | **Date probe** | "As of 2020" minus "As of 2026": **−0.019 [−0.027, −0.012]**, in 9/9 models. Under "As of 2026" staleness is 0.454 [0.417, 0.489], still below chance. | `date_probe/date_probe.json` |
| 12 | Vintage ladder (27B) | Qwen3.5 / 3.6 / 3.8: 0.415 / 0.442 / 0.447. No step differs from zero, so the ladder does not localize the effect. | `vintage/vintage_matched.json` |
| 13 | Staleness on FreshQA | 9/9 models below 0.5, but pooled **0.481 [0.456, 0.504]**, p = 0.12: direction-consistent, not significant | `item1/freshqa_pooled.json` |
| 14 | Inverse result (secondary, direction only) | Mixed quantization, disjoint set: slope −1.367, permutation p = 0.013. **Fixed 4-bit: slope −0.27 to −0.73, p = 0.08–0.40** (appendix table). The direction is negative in every variant. | `inverse/inverse_disjoint.out`, `item4/item4_inverse.json`, `item4/README.md` |
| 15 | A6, disjoint false founders | Each person appears once; name-frequency screen 1.000 → 0.500; truth AUROC rises +0.02 to +0.05 for the weaker models | `a6_disjoint/truth_before_after.json` |
| 16 | Form screen | Our set passes after A6. FreshQA's lexical cue ("current/latest/now") reaches **0.755**, above every model (0.52–0.58). | `a6_disjoint/screen_disjoint_qwen35.out`, `item1/freshqa_results.json` |
| 17 | **TSCT seeds** | 5 paired, compute-matched seeds (8,008 examples, 6,006 steps). **ΔAUROC −0.0003 [−0.022, +0.021]**, ΔECE −0.0001, Δresolution 0; FreshQA ΔAUROC −0.014. No effect: TSCT is audit-only. | `item5/seed_variance_routeB.json`, `item5/freshqa_seeds.json`, `item5/ENDPOINTS.md` |
| 17b | **TSCT seeds on LLaMA-3-8B (the team's base)** | 5 paired, compute-matched seeds. TemporalDelta **ΔAUROC −0.0024 [−0.019, +0.014]**, ΔECE −0.0014 [−0.005, +0.002], Δresolution 0; no effect, agreeing with Qwen. Constant / oracle 5.6–6.5× on these predictions. FreshQA ΔAUROC −0.058 [−0.148, +0.033]; ΔECE −0.044 [−0.082, −0.006] is a confidence level shift (reliability only, resolution +0.0004), not a benefit. Hedges on reserved tokens; tokenizer class fixed. | `item5_llama3/RESULTS.md`, `item5_llama3/llama3_results_final.json` |
| 18 | Team LLaMA-3 seeds | Unusable: 7 of 8 `exp3_v7` seeds are byte-identical, with identical loss curves; the other seeds mix setups | `item5/README.md`, `item5/seed_variance_routeA_*.json` |
| 19 | Table 3 (evergreen), rebuilt on answers | Four of six rows had scored reasoning preambles; rebuilt. r(PPL) is 0.11–0.19 without gpt-oss; no size trend in any version; the vintage-pair gap is 0.038 (was 0.061). gpt-oss is flagged. | `evergreen/table3_rebuild.json` |
| 20 | **Gemma-4 root cause** | The instruct checkpoints themselves. Pretrained 31B: NLL **2.19** vs instruct 3.96–4.73. llama.cpp PPL **208** vs mlx 201 (instruct), pretrained 5.0. Not quantization. | `gemma4/NOTES.md`, `gemma4/g3_llamacpp.log` |
| 21 | Harness fixes | gpt-oss scored with `--bos off` (reproduces Aug 13 exactly); answer-only prompts for reasoning models; training fails instead of falling back to 49 samples; seed data size recorded | `MODELS.md`, git log |
| 22 | **Frequency test (pre-registered), null** | Across 106 entities, the log ratio of expired to current co-occurrence in Dolma v1.7 does not predict the per-entity preference: slope **+0.038**, permutation p = **0.12**. With the amendment-1 controls (96 entities) the coefficient is **−0.02** (p = 0.39). Worker B, `43ca289`. | `frequency/RESULTS.md`, `frequency/frequency_results.json` |

**Gemma-4 footnote (draft, from `gemma4/NOTES.md`):** "Gemma-4 instruction-tuned checkpoints assign anomalously low probability to ordinary text: perplexity about 200 on plain English, against 5 for the pretrained Gemma-4-31B. We confirmed this in two independent implementations (MLX and llama.cpp) and at two quantizations, so their teacher-forced log-probabilities are not comparable with the other models' and they are excluded."

**Open, for Edward:**
- Send the Jason message (`SEP10_STATUS.md`) about the Sep 2 numbers with no source: the Brier column, the sitelinks control, the time split, the §4.3 CI and the 1,953 count.
- Decide on the stray temporal-delta copy in the HF cache (unused).
