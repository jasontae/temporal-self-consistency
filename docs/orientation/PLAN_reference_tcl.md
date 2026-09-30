# Plan: E0, the team's reference TCL (worker E, 2026-09-30)

Written and committed before any run. Training, under Edward's standing approval for experiments that improve
integrity, as ordered by the orchestrator: first in the gate after 2026-10-01 12:00 UTC, ahead of E1–E3. One GPU job
at a time, titled herdr pane, watchdog. Every result is reported, nulls included. No paper edits. Code and runs:
worktree `~/Projects/temporal-self-consistency-corrtarget`, branch `altj-corr-target`.

## Why

Worker D's upstream reconciliation (`ACCURACY_AUDIT_2026-09-29.md` §6, finding 1) shows that the team's reference
Temporal Calibration Loss is recoverable (`8c945c2:src/training/tcl_loss.py`, Jun 6). It is correctness-driven.
Our TSCT reimplementation drops that signal:
- its L_over and L_under pull ĉ toward the gold category's level;
- its R_hedge is an anti-collapse entropy term.

So the paper's TSCT null, on the prespecified primary endpoint (ΔAUROC of stated confidence against correctness),
is null by construction for our variant. It does not test the team's design. E0 runs the design itself, and answers
the objection "you didn't test their method".

## The arm (reference TCL, "REF")

Ported faithfully from `8c945c2` into `src/training/run_tcl_mlx.py` as `--loss reference`; the default remains
our variant, so no earlier run changes.

**Loss:**
- TCL = CE + 0.5·L_over + 0.5·L_under − 0.3·R_hedge, over the fast/slow ("volatile") examples of the batch.
- ĉ = the softmax expectation of the four hedge levels at the hedge position (identical to the reference and to
  our variant).
- L_over = Σ vmask·wrong·ĉ / n_vol.
- L_under = Σ vmask·correct·(1 − ĉ) / n_vol.
- R_hedge = Σ vmask·p(gold hedge) / n_vol, subtracted as a reward.

**`is_correct` ("was the factual answer correct"):** the reference leaves it to the trainer. We compute it from
the same forward pass: an example is correct iff the model's argmax equals the gold token at every answer position.
- Under teacher forcing this equals "the model's greedy answer is exactly the gold answer now", by induction on the
  prefix.
- It is stop-gradient and changes as training proceeds. That is the reference's semantics: correctness of the model
  being trained.
- Recorded per step as `train_correct_frac` in `loss_log.csv`.
- It is not the team's later run code, which by their diagnosis used NLL below the batch median. That code is not
  in any repo.

**Unit check (CPU, before this plan):** on toy logits, the label and all three terms match hand computation.

**Everything else is identical to the existing plain fine-tuning seeds**, verified from their `run_meta.json`:
- Qwen2.5-7B-Instruct-4bit, the same `load_slice(seed)` examples and order (8,008), 3 epochs, 6,006 steps
- batch 4, lr 5e-5 (the reference's own recommendation), LoRA r8 / scale 20 / 16 layers plus the LM head
- chat prompt, added hedge tokens

## Comparator and pairing

- **Primary comparator:** the existing plain fine-tuning seeds 0–4 (`sft_only_seed{0,1}`, `sft_only_seed{2,3,4}_r2`).
  They are CE only (λ = 0), which is exactly REF with the three terms removed, so the setup matches and they are
  reused, not retrained.
- **Also reported:** REF minus our variant (the existing TSCT seeds 0–4), and E3's knowledge-target arm once it
  exists.

## Metrics (item5 `ENDPOINTS.md`, unchanged)

- **Primary:** ΔAUROC(stated hedge confidence → correct), REF minus plain fine-tuning, paired over seeds 0–4, on
  TemporalDelta test (3,622 items, pipeline exact match).
- **Secondary:** Δ resolution, Δ AURC, Δ selective risk at 50%, Δ ECE and reliability (not interpreted alone),
  accuracy, the hedge distribution, and the same on FreshQA (containment, current and aged keys). The constant
  [UNKNOWN] and the volatility-label oracle are reported with them.
- **Training diagnostics, descriptive:** `train_correct_frac` over steps, the c_hat gradient norm (must be nonzero),
  and hedge collapse (any single hedge above 95% of test outputs).

## Decision rule (ENDPOINTS.md, unchanged)

A REF effect is reported only if all three hold:
- the primary 95% t-interval over the five seeds excludes 0;
- the sign is consistent across all five seeds;
- |mean Δ| > 2× the plain fine-tuning seed SD (0.0245, so > 0.049).

Otherwise: no demonstrated effect of the team's reference design on this setup.

**What each outcome would mean** (for Edward and the team; nothing is edited here):
- **Effect:** the negative method result applies to our variant only. Contribution 1 and abstract (ii) must say that
  the team's correctness-driven design does add item-level signal. This is a major change to the paper, and the
  most important reason to run E0 before submission.
- **No effect:** "neither our variant nor the team's reference TCL shows a demonstrated benefit (five paired seeds
  each)". The null then covers the team's design, which answers the reviewer objection directly.
- **Collapse** (one hedge above 95%) or a correct-fraction that saturates early: reported as the observed behaviour
  of the reference design, alongside the endpoint.

## Cost and schedule

About 2.0 h training plus about 20 min evaluation per seed, so about 12 h for five seeds. Gate order after
2026-10-01 12:00 UTC:
1. E0, this plan (about 12 h)
2. E1, knowledge probe (about 15 min)
3. E2, context control (about 30 min)
4. E3, knowledge target (about 13 h)

Adapters and predictions stay local. `run_meta.json`, `loss_log.csv`, predictions and the queue log are committed
per seed on `altj-corr-target`, not pushed.
