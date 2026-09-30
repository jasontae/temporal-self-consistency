# Item 5: prespecified endpoints

Written 2026-09-28, before any new seed data was analysed. Neither the LLaMA-3 predictions (FETCH_ALL step 4) nor any new local seed existed yet. `seed_variance.py` computes exactly these endpoints.

**Design:**
- **5 seeds per arm, paired:** the same seed trains the TSCT arm and the plain-fine-tuning arm.
- **Compute-matched baseline:** plain fine-tuning (the cross-entropy-only arm, all TCL λ = 0) with identical data, steps, epochs, batch size, learning rate and LoRA configuration. Only the loss differs.
- **Route A** (the team's LLaMA-3 runs) qualifies only if it passes three checks:
  - the duplicate-content check (identical outputs are one seed, not several)
  - the post-gradient-fix check (training config or date)
  - step-count equality between the arms
  
  Otherwise **route B**: local Qwen2.5-7B seeds 0–4.

**Primary endpoint (one per benchmark):**
- **ΔAUROC** = AUROC(stated hedge confidence → answer correct), TSCT minus plain fine-tuning, paired over seeds.
- Report the mean, the SD, a t-interval over seeds, and whether the sign is consistent across all seeds.
- Discrimination is the quantity the calibration loss claims to improve and the one ECE cannot see.

**Secondary endpoints** (same pairing and reporting):
1. Δ Murphy resolution (exact, by stated level)
2. Δ AURC: area under the selective risk–coverage curve, answers ranked by stated confidence, ties broken by the answer's mean log-probability where recorded
3. Δ selective risk at 50% coverage
4. Δ ECE (per level) and Δ Brier reliability, reported for completeness and not interpreted alone, per `results_sep10/vocabulary/`
5. Hedge-distribution shift (the Sep 2 "soft-confidence direction") as a descriptive endpoint only

**Controls reported with every endpoint:**
- the volatility-label oracle
- the constant policies
- the seed SD of the plain-fine-tuning arm alone, which is the noise floor any TSCT effect must exceed

**Decision rule, fixed in advance.** A TSCT effect is reported as an effect only if all three hold:
- the primary endpoint's 95% t-interval over seeds excludes 0
- the sign is consistent across all 5 seeds
- |mean Δ| is larger than 2× the plain-fine-tuning seed SD

Otherwise TSCT stays an audit result, as in Sep 2.
