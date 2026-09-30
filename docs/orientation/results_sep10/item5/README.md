# Item 5: TSCT seeds (5 per arm, paired, compute-matched)

The endpoints were prespecified in `ENDPOINTS.md` before any new seed data.

## Route A (the team's LLaMA-3 runs, HF `DavidS64/llama3-hedge-sft` @ d66a438): not usable

`seed_variance.py --route A` was run on every prediction directory (`seed_variance_routeA_<dir>.json`), and the trainer states were compared across seeds:

1. **`exp3_v7` is not eight seeds.**
   - Seeds 42, 123, 456, 501, 502, 504 and 505 have byte-identical predictions on all four benchmarks (`predictions_v7`, `predictions_v8`).
   - Their `trainer_state.json` loss curves are identical at every logged step over 1,825 steps (first losses 5.37285, 3.83109, 3.50896; final 2.58479).
   - Only seed 503 differs. `exp3_v7` is two runs, and the seed argument evidently did not reach training.
2. **`exp2_v2` seeds 502–505 are identical to one another** (predictions and loss curves).
3. **The runs labelled as seeds are different setups.**
   - `exp3_v6` seeds 42/123/456 start at loss about 2.0; seeds 501–505 start at about 24.
   - `exp2_v2` seed 123 starts at 1.05, against 2.40 for seeds 42 and 456.
   - A July diagnosis note in the team's private material (`docs/orientation/sources/`, not committed) reports that `exp3_v6` seeds 501–505 behave as though the SFT/TSCT labels were inverted.
4. **Gradient-fix status cannot be established.** The team repo deleted its training code (upstream `8ff91b1`, "Remove training directory").

**Consequence for the Sep 2 PDF.** Its "three paired seeds 42/123/456 … seed 456 reverses the direction" cannot be checked against clean seeds.
- If the TSCT arm is `exp3_v7`, all three TSCT "seeds" are the same run, and any reversal comes from the SFT side alone.
- If the TSCT arm is `exp3_v6`, seeds 42/123/456 are a different configuration from 501–505.

Either way, the team runs do not meet the prespecified requirements: distinct paired seeds, the same configuration, and post-fix.

## Route B (local Qwen2.5-7B-4bit): done, 5 paired seeds per arm

- `run_local_seeds.sh`, via `../run_queue_d3b.sh`. Seeds 2–4 were trained 2026-09-28 12:47 → 2026-09-29 01:44 UTC (`local_seeds.log`; train 6,479–7,674 s, eval 869–930 s per run). Seeds 0–1 are the August adapters.
- **Compute-matched.** Every one of the ten runs has 8,008 examples, 6,006 steps and identical LoRA, learning rate and schedule; only λ differs (TSCT 0.5/0.5/0.3, plain fine-tuning 0) (`data/prep/tcl_mlx_7b/*/run_meta.json`).
- The first seed-2 launch used `--n-per-volatility 2000` (4,008 examples) and was stopped and rerun with 4000, which matches the August runs. `run_meta` now records the value.
- **Duplicate check:** 10 distinct prediction files, no identical outputs.

### TemporalDelta test (3,622 items; `seed_variance.py --route B` → `seed_variance_routeB.{json,out}`)

TSCT − plain fine-tuning, paired over seeds 0–4:

| endpoint | mean Δ | 95% t-interval | per-seed Δ | sign-consistent | plain-FT seed SD |
|---|---|---|---|---|---|
| **AUROC (primary)** | **−0.0003** | [−0.022, +0.021] | +0.028, −0.001, −0.012, +0.001, −0.017 | no | 0.0245 |
| resolution | 0.0000 | [−0.0000, +0.0000] | | no | 0.0000 |
| AURC | +0.0010 | [−0.004, +0.006] | | no | 0.0073 |
| selective risk at 50% | +0.0011 | [−0.002, +0.004] | | no | |
| ECE (per level) | −0.0001 | [−0.003, +0.003] | | no | 0.0019 |
| reliability | −0.0003 | [−0.004, +0.003] | | no | |
| accuracy | −0.0004 | [−0.004, +0.003] | | no | |

**By the prespecified rule, TSCT has no effect.** Every arm-seed sits at ECE 0.451–0.460, resolution ≤ 8e-5 and AUROC 0.50–0.57, with the same hedge distribution (about 3,280 [TEMPORAL_HEDGE] and 335 [COND_CONFIDENT]).

### FreshQA (413 items, containment scoring; `freshqa_seeds.py` → `freshqa_seeds.{json,out}`)

| key | ΔAUROC (primary) [95% CI] | effect by rule |
|---|---|---|
| current (2026-04) | −0.014 [−0.080, +0.052] | no |
| aged (2024-02) | −0.014 [−0.071, +0.043] | no |

One secondary endpoint excludes zero: AURC under the aged key, +0.024 [+0.003, +0.044], sign-consistent. TSCT's selective risk is slightly *worse*. It fails the size criterion (|Δ| < 2× the plain-FT SD of 0.028), and it is 1 of 28 secondary comparisons, so it is reported and not interpreted.

## What this changes for the paper

- The Sep 2 text says "the observed ECE difference is less than 0.005 and its direction is not stable across the three paired seeds". It rests on team runs that cannot be verified: the `exp3_v7` seeds are one run, and the others mix setups.
- The local five-seed result supports the same conclusion with clean, compute-matched seeds and discriminative endpoints: **no TSCT effect on ECE, AUROC, resolution or selective risk on either benchmark.** TSCT stays an audit result.
- The model differs (Qwen2.5-7B, not the team's LLaMA-3-8B). That should be stated.
