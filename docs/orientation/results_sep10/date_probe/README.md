# Date-conditioning probe (Directive 3, Worker A item 6)

**Runner:** `run_date_probe.sh`, run by `../run_queue_d3.sh`: 27 runs, 47–172 s each, `date_probe.log`, finished 2026-09-28 12:46 UTC. **Analysis:** `date_probe_analysis.py` → `date_probe_table.md`, `date_probe.json`, `date_probe.out`.

**Design:**
- Every disjoint-set claim is scored after a context prefix, on the claim's own tokens only (`score_matched_regime --context`). The neutral name frame used for PMI is scored without the prefix.
- Contexts:
  - "As of 2026," and "In 2026,": the present
  - "As of 2020,": a past date when most stale holders were in office (control)
- Models: all nine admitted, at 4-bit (item 4 checkpoints; gpt-oss with `--bos off`).
- Staleness AUROC is current vs stale holder; below 0.5 means the stale value is preferred. The entity bootstrap uses the same resample across models.

## Result

| pooled over nine models | Δ staleness AUROC [95% CI] | p | models moving in the predicted direction |
|---|---|---|---|
| "As of 2026," − no context | +0.012 [+0.001, +0.024] | 0.030 | 5/9 |
| "In 2026," − no context | +0.017 [+0.007, +0.027] | 0.001 | 9/9 |
| "As of 2020," − no context | −0.008 [−0.018, +0.003] | 0.15 | 6/9 |
| **"As of 2020," − "As of 2026,"** | **−0.019 [−0.027, −0.012]** | < 0.001 | **9/9** |

| mean staleness AUROC | value [95% CI] |
|---|---|
| no context | 0.442 [0.405, 0.477] |
| "As of 2026," | **0.454 [0.417, 0.489]** |
| "In 2026," | 0.459 [0.422, 0.495] |
| "As of 2020," | 0.435 [0.397, 0.471] |

**Per model** (`date_probe_table.md`): Qwen3.6-35B-A3B goes from 0.450 to 0.505 under "As of 2026,"; Qwen3.5-27B and Qwen3.6-27B move by +0.035 to +0.037; the 4B–7B models and gemma-3 move by ≤ 0.02, some in the opposite direction.

## Reading

- **The models are date-sensitive, in the predicted direction.** A stated past date favours the stale holder relative to a stated present date in every one of the nine models.
- **The effect is small**, about 0.02 AUROC, and larger for the bigger models.
- **The stale-value preference survives conditioning on the present.** With "As of 2026," the mean stays below chance (CI upper bound 0.489), and with "In 2026," at 0.495. Only one model (Qwen3.6-35B) reaches chance.
- This fits the base-vs-instruct result (`../base_instruct/`): a pretraining familiarity effect that a date cue nudges but does not override.
