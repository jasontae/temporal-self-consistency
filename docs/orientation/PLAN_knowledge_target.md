# Plan: knowledge-aware hedge targets (worker E, 2026-09-30)

**Status (2026-09-30, later): deferred, not started.** Edward reserved the GPU for YBPA until 2026-10-01 12:00 UTC. The queue was stopped before labelling or training (`altj-corr-target` @ `45bd7ca`). To run after the reservation: `bash docs/orientation/results_sep10/knowledge_target/run_kt_queue.sh` plus `kt_watchdog.py` in the corr-target worktree, in a titled herdr pane. It needs about 13 h, so results land about Oct 2, 01:00 UTC if started at 12:00.

Written and committed before any training. New training runs, authorised by Edward's 2026-09-30 update (plan
first, one GPU job at a time in a titled herdr pane with the memory watchdog, every result reported, nulls
included, no paper edits). Code and runs: worktree `~/Projects/temporal-self-consistency-corrtarget`, branch
`altj-corr-target` (from `altj-sep10-revisions` @ `ebeb616`).

## Why this run, and not another

The paper's method result is that TSCT gives no benefit over plain fine-tuning (ΔAUROC −0.0003 [−0.022, +0.021],
five paired Qwen2.5-7B seeds; the same null on LLaMA-3 8B). A careful reviewer will say the null is close to
guaranteed by construction, not a finding about hedge training:

- The hedge target in *both* arms is the dataset's volatility label, which the question template fixes; both arms
  learn it (at least 3,606 of 3,622 test items). The calibration loss pulls ĉ toward that same label and never
  sees correctness (Appendix L). A model that emits the label perfectly equals the oracle, and the paper shows the
  oracle has resolution 3e-5. So neither arm *can* gain resolution.
- The same models' own answer log-probabilities already separate right from wrong answers (AUROC 0.80–0.90,
  Appendix M; 0.82–0.88 recalibrated, Figure 8). The information exists; the hedge target never asks for it.
- The team's original design also penalised confident wrong answers, which the paper says "we did not run", and
  prior work that supervises stated confidence with the model's own correctness reports gains (R-Tuning, Zhang et
  al. NAACL 2024; Kapoor et al. 2024; Lin et al. 2022), while Zhou et al. (ICLR 2026) study abstention training
  for temporal QA directly.

This run changes exactly one thing, the hedge target, and asks whether the four-token vocabulary can carry
item-level signal when it is supervised with knowledge rather than volatility. Either result improves the paper:
a positive result says what the vocabulary *can* learn and pins the null on the target, not on hedging; a null
closes the most obvious objection to the negative method result.

## Hypothesis

**H1:** hedge targets derived from the base model's own correctness give the emitted hedge real discrimination of
answer correctness on TemporalDelta test (AUROC above plain fine-tuning), where volatility-label targets give none.

## Arm and data

- **Knowledge-target arm (KT):** plain fine-tuning (cross-entropy only, all TCL λ = 0) with the hedge target
  replaced record by record: a record keeps its gold hedge when the untuned Qwen2.5-7B-Instruct-4bit answers it
  correctly, else [UNKNOWN] (0.10).
  - Known = the base model's greedy answer to the record's question (prompt: the question plus "Answer with only
    the name, no explanation.") matches the record's answer by the eval pipeline's exact match, contains it after
    normalisation, or contains every word of it. Computed once for all 12,753 unique training questions
    (`src/training/knowledge_labels.py`), before any training.
  - Everything else identical to the existing plain fine-tuning arm of the same seed: same `load_slice` examples
    and order (`--n-per-volatility 4000`: 8,008 examples, 6,006 steps), LoRA, learning rate, schedule, chat
    prompt, base checkpoint.
- **Comparator:** the existing plain fine-tuning seeds 0–4 (`data/prep/tcl_mlx_7b/sft_only_seed{0,1}`,
  `sft_only_seed{2,3,4}_r2`) and their predictions. No comparator is retrained. TSCT is reported alongside.
- **Seeds:** 0–4, paired with the comparator by seed.

## Metrics (the prespecified endpoints of `results_sep10/item5/ENDPOINTS.md`, unchanged)

- **Primary:** ΔAUROC = AUROC(stated hedge confidence → answer correct), KT minus plain fine-tuning, paired over
  the five seeds, on TemporalDelta test (3,622 items, correctness = pipeline exact match, as for every arm).
- **Secondary:** Δ Murphy resolution, Δ AURC, Δ selective risk at 50% coverage, Δ ECE and reliability (reported,
  not interpreted alone), hedge distribution; the same on FreshQA (containment scoring, both keys). The constant
  [UNKNOWN] policy and the volatility-label oracle are reported with every number.
- **Also reported:** accuracy of each arm (the target change could cost accuracy), the fraction of training
  examples relabelled, and the AUROC of the answer's own mean log-probability for each arm (the ceiling the hedge
  is being asked to reach).

## Decision rule (ENDPOINTS.md, unchanged)

A KT effect is reported as an effect only if the primary endpoint's 95% t-interval over seeds excludes 0, the sign
is consistent across all five seeds, and |mean Δ| > 2× the plain fine-tuning seed SD (0.0245, so > 0.049).
Otherwise it is reported as no demonstrated effect. Every seed is reported, whatever the outcome.

What each outcome would mean for the paper (for Edward and the team to decide; nothing is edited here):
- **Effect:** the negative method result narrows to "volatility-label targets give no resolution; knowledge
  targets do". One sentence in §3 plus an appendix table; the Contribution 1 wording "anyone planning to teach
  freshness through fine-tuning can start from this null" would need to change.
- **No effect:** one sentence in §3 ("a knowledge-target variant, supervising the hedge with the base model's own
  correctness, also shows no demonstrated gain") plus the appendix table; the null becomes much harder to dismiss.

## Budget and safety

About 1 h for the labels, then per seed about 2 h training and 20 min evaluation: about 13 h serial on the local
GPU, overnight. One job at a time, in the herdr workspace "ALTJ-E knowledge-target training (GPU)", after the
knowledge probe finishes; memory guard before each step and a step-time watchdog (pause, not kill, if memory
pressure is critical or step time exceeds 3× the Qwen baseline for 10 minutes). Adapters and predictions stay
local; `run_meta.json`, loss logs and the queue log are committed per seed on `altj-corr-target`, not pushed.
