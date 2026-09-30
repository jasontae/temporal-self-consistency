# Base vs instruct: Qwen2.5-7B (Directive 3, Worker A item 5)

**Script:** `base_vs_instruct.py` → `base_vs_instruct.{md,json,out}`. Scored by `../run_queue_d3.sh` (wall 41 s and 45 s, `../queue_d3.log`).

**The pair:**

| | base | instruct |
|---|---|---|
| checkpoint | `mlx-community/Qwen2.5-7B-4bit` @ fd815ff0 (converted from `Qwen/Qwen2.5-7B`) | `mlx-community/Qwen2.5-7B-Instruct-4bit` @ c26a38f6 |
| converter and recipe | mlx-lm 0.18.1, 4-bit g64 | mlx-lm 0.18.1, 4-bit g64 |

Pretraining is fixed; only post-training differs.

**Base audit** (`../model_audit/runs/qwen25_7b_base.json`):
- NLL 2.50, against 2.46 for instruct
- known-answer raw 0.89 / PMI 0.81
- no chat generation, since it is a base model

## Result (PMI, paired within entity; entity-paired bootstrap, 4,000 resamples)

| claim set | contrast | base | instruct | base − instruct [95% CI] | p |
|---|---|---|---|---|---|
| original | truth | 0.776 | 0.794 | −0.018 [−0.036, +0.000] | 0.053 |
| original | regime | 0.583 | 0.634 | **−0.051 [−0.077, −0.026]** | < 0.001 |
| original | **staleness** | **0.428** | **0.426** | **+0.003 [−0.022, +0.028]** | 0.84 |
| disjoint | truth | 0.792 | 0.818 | −0.026 [−0.044, −0.009] | 0.003 |
| disjoint | regime | 0.583 | 0.634 | −0.051 [−0.076, −0.027] | < 0.001 |
| disjoint | staleness | 0.428 | 0.426 | +0.003 [−0.022, +0.027] | 0.83 |

## Reading

- **The stale-value preference is a pretraining property.**
  - The base model already shows it at the same level as the instruct model (0.428 vs 0.426).
  - The difference is +0.003, with a CI of about ±0.025.
  - This is the clean localization the vintage ladder could not provide (`../vintage/`: flat within noise).
  - It replaces the Sep 2 "suggestive, not localizing" single-pair argument with a direct test, for one family.
- **Post-training changes the other two contrasts.** Regime separation rises by 0.05 and truth discrimination by 0.02–0.03. Both move away from the familiarity result. So the regime effect behind the inverse result is partly a post-training property, which is consistent with its fragility under re-quantization (`../item4/`).
- **Scope.** One family (Qwen2.5) at one size. There is no base/instruct pair for the other admitted families on disk, and Directive 3 adds no new families.
