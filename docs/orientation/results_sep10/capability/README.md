# External capability proxy and null simulation for the inverse result (directive 3, worker B)

**Scripts:**
- `capability_proxy.py` → `capability_proxy.{md,json}`
- `null_simulation.py` → `null_simulation.json`

**Inputs:**
- `../inverse/inverse_matched.json`: per-model truth, regime and paired win (nine admitted, derangement build)
- `data/prep/predictions_7b/matched_*.jsonl`
- `proxies.json`: every value with its official source; looked up 2026-09-28 from model cards and tech reports

## 1. External capability proxy (fused attack 3: "truth AUROC is in-instrument")

| x-axis | n | regime AUROC: slope, Spearman ρ, exact permutation p | paired win: ρ, p |
|---|---|---|---|
| truth AUROC (in-instrument; reference) | 9 | −0.961, ρ −0.717, p 0.008 | ρ −0.828, p 0.014 |
| **log10 total parameters** | 9 | −0.112 per decade, **ρ −0.633, p 0.018** | ρ −0.686, p 0.035 |
| **published MMLU-Pro** | 7 | −0.0029 per point, **ρ −0.714, p 0.022** | ρ −0.721, p 0.036 |
| log10 active parameters | 9 | −0.054, ρ +0.127, p 0.35 | ρ −0.085, p 0.40 |

- **Two proxies measured outside the instrument give the same sign as truth AUROC**, with exact permutation p ≈ 0.02. They also track truth AUROC: Spearman +0.84 for total parameters and +0.86 for MMLU-Pro. So the inverse result is not an artifact of putting the same instrument on both axes.
- **Active parameters show nothing.** The two MoE models (gpt-oss-20B, 3.6B active; Qwen3.6-35B-A3B, 3B active) behave like large models, not small ones. That fits the knowledge account: parametric knowledge scales with total parameters, not with per-token compute. Treat this as an observation, not a test.
- **Leave-one-family-out:** the sign holds without Gemma and without gpt-oss for every proxy. Without Qwen only two models remain, so it is not estimable. With seven of nine models from Qwen, this is the limit of what the ladder can show, and the paper should say so.
- **Caveats:**
  - The MMLU-Pro settings differ across cards: Gemma is zero-shot, Qwen3.x states no shots or mode, and Qwen2.5 is the instruct table.
  - The two missing models are the TSCT fine-tune (no published score) and gpt-oss (only an unverified community value, not used).
  - Plain MMLU is published for only 2 of the 9 checkpoints, so it was not used.
  - TSCT is assigned its base model's parameter count.

## 2. Null simulation (fused attack 10: "the two scores are algebraically coupled")

Truth (T vs F) and regime (T vs V) share the immutable_true arm. The simulation builds a generative null from the real data:
- **Parameters from the nine models' PMI:** entity variance 0.36, residual variance 0.47, cross-model entity correlation 0.73.
- **Truth effect:** each model's effect is set to reproduce its observed truth AUROC.
- **Regime effect:** held **fixed** across models, i.e. no dependence of temporal sensitivity on capability.
- **Replicates:** 2,000 nine-model panels, scored with the same unpaired AUROC as `inverse_truth.py`.

| scenario | simulated slope: mean [2.5%, 97.5%] | share ≤ observed −0.961 |
|---|---|---|
| no temporal sensitivity (r = 0) | +0.086 [−0.361, +0.456] | 0 of 2,000 |
| fixed sensitivity at the observed mean regime AUROC | +0.088 [−0.374, +0.442] | 0 of 2,000 |

- The generator reproduces the observed truth AUROCs (mean 0.795 simulated vs 0.794 observed).
- The shared arm couples the two scores *positively*, not negatively: a model whose immutable_true scores sit high by chance gains on both contrasts.
- A slope of −0.96 does not arise from the design when temporal sensitivity is fixed (p < 1/2,000). This complements the split-half control in `../inverse/`, which removes the shared items directly.

## For the paper (§7.2): superseded by §3

~~One sentence plus an appendix table: "The negative relation holds with capability measured outside the instrument (log parameters, ρ = −0.63; published MMLU-Pro, ρ = −0.71; exact permutation p ≈ 0.02) …"~~

Withdrawn: these are mixed-quantization numbers. At fixed 4-bit, the parameter-count result is not significant (§3).

## 3. Update: re-run on worker A's fixed 4-bit ladder (item 4). This supersedes §1 for the paper.

Worker A's item 4 (`altj-sep10-revisions` @ `8e02ed5`, `results_sep10/item4/README.md`) found that at fixed 4-bit the inverse truth–regime relationship loses significance. Its slope is −0.27 to −0.73, with p = 0.08–0.40. §1 above used the mixed-quantization ladder, so it is re-run here.

- **Script:** `capability_proxy_4bit.py` → `capability_proxy_4bit.{md,json}`.
- **Inputs:** A's committed `{matched4b,disjoint4b}_*.jsonl` @ `9829f89`, extracted read-only with `git show`; A's working tree was not touched.
- **Check:** the regime-on-truth slope reproduces A's values exactly: −0.571 on the original set and −0.268 on the disjoint set, all nine models.

| x-axis (4-bit) | n | regime AUROC: ρ, exact p | paired win: ρ, exact p | mixed quantization (§1), regime AUROC |
|---|---|---|---|---|
| truth AUROC, original set | 9 | −0.600, p 0.099 | −0.544, p 0.17 | −0.717, p 0.008 |
| truth AUROC, disjoint set | 9 | −0.400, p 0.40 | −0.293, p 0.53 | (disjoint −0.667, p 0.013, A6) |
| log10 total parameters | 9 | **−0.430, p 0.13** | −0.140, p 0.44 | −0.633, p 0.018 |
| published MMLU-Pro | 7 | **−0.857, p 0.036** | −0.595, p 0.062 | −0.714, p 0.022 |
| log10 active parameters | 9 | −0.143, p 0.28 | −0.072, p 0.34 | +0.127, p 0.35 |

**Reading:**
- The **sign stays negative for every proxy at 4-bit.**
- **Significance does not survive** for parameter count or for either paired-win outcome.
- Only MMLU-Pro against regime AUROC stays below 0.05. That is one test among several, on 7 points with card settings that differ across models, so it cannot carry the result alone.
- Regime AUROC does not use the false-founder arm, so it is identical across the two claim sets, and the proxy rows are too.
- **For the paper:** the inverse result is a *direction*, consistent across quantizations and proxies, not a significant capability dependence. Report it with A's fixed-quantization table.
- The null simulation (§2) still stands as a statement about the design: fixed temporal sensitivity does not produce a negative slope. It does not rescue the significance.
