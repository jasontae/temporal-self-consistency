# v6 Diagnosis and TCL Redesign Proposal

**Author:** Claude Code session, 2026-07-31
**Inputs:** context bundle (`CLAUDE.md`, `docs/*`, `eval_toolkit/*`) + the actual
prediction files at `~/Downloads/david_v6/predictions_v6` (64 files = 8 seeds x
4 benchmarks x 2 conditions, all present and complete).

Every number below is labelled with the script that produced it. Numbers marked
**[doc]** come from `docs/RESULTS_LOG.md`; numbers marked **[recomputed]** come
from running `eval_toolkit/per_benchmark_breakdown.py` and
`eval_pipeline.compute_ece` / `bonferroni_ece_comparison` directly on the v6
files in this session.

**Reproducing the contested numbers.** Section 0 contradicts the current
`RESULTS_LOG.md`, so it ships with the scripts to check it independently:

```bash
python3 repro/verify_v6_seed_split.py --pred-dir /path/to/predictions_v6   # section 0
python3 repro/ece_strategy_probe.py                                        # section 1.5
```

Both are CPU-only, take seconds, and import the project's own
`eval_toolkit/eval_pipeline.py` — they change which seeds get averaged, not how
any metric is computed.

---

## 0. Priority finding: the v6 result as documented does not reproduce

I ran the project's own `per_benchmark_breakdown.py` on the complete
`predictions_v6` folder. Two things came out that change what the next task
should be.

### 0.1 The "n=8" run is arithmetically an n=3 run

Every headline v6 number in `RESULTS_LOG.md` and `CLAUDE.md` reproduces
*exactly* when I restrict to seeds **42, 123, 456** — and only those three.

| Quantity | [doc] "v6, n=8" | [recomputed] seeds 42/123/456 | [recomputed] all 8 seeds |
|---|---|---|---|
| temporal_delta ECE SFT→TSCT | 0.854 → 0.419 | 0.854 → 0.418 | 0.507 → 0.590 |
| mmlu ECE | 0.754 → 0.375 | 0.754 → 0.375 | 0.513 → 0.492 |
| freshqa ECE | 0.756 → 0.354 | 0.756 → 0.354 | 0.474 → 0.439 |
| stress_test ECE | 0.847 → 0.416 | 0.847 → 0.416 | 0.502 → 0.587 |
| MMLU EM | 0.067 → 0.030 | 0.0674 → 0.0299 | 0.0469 → 0.0571 |
| FreshQA EM | 0.097 → 0.049 | 0.0972 → 0.0489 | 0.0727 → 0.0898 |
| ECE reduction / p / d | +0.435 / 0.0099 / 3.767 | +0.4350 / 0.0099 / +3.767 | −0.0831 / 0.494 / −0.351 |
| MMLU TSCT hedge dist | 10.9 CONF, 61.6 TH, 22.7 UNK | 10.9 / 61.6 / 22.7 | 4.5 CONF, 57.1 COND, 23.2 TH, 15.2 UNK |

The match on seeds 42/123/456 is exact to the printed precision on all eight
rows, including p and Cohen's d. The five seeds David added (501–505) were
downloaded but never entered the reported numbers.

**Consequence:** pooled over all 8 seeds, the headline calibration result
disappears. On the primary benchmark the ECE *reduction* becomes an ECE
*increase* (−0.083, p=0.494, d=−0.351). The "~50% ECE reduction, d=3.77"
claim currently in `PAPER_STATUS.md` as the honest headline is an n=3 result
labelled n=8.

### 0.2 Seeds 501–505 behave as though the condition labels are inverted

The split is not noise. It is a clean, total inversion across all four
benchmarks and all five seeds. [recomputed]

| benchmark | seed group | SFT ECE | TSCT ECE | SFT EM | TSCT EM |
|---|---|---|---|---|---|
| temporal_delta | 42/123/456 | 0.854 | **0.418** | 0.0368 | 0.0055 |
| temporal_delta | 501–505 | **0.299** | 0.693 | 0.0078 | 0.0402 |
| mmlu | 42/123/456 | 0.754 | **0.375** | 0.0674 | 0.0299 |
| mmlu | 501–505 | **0.368** | 0.563 | 0.0346 | 0.0734 |
| freshqa | 42/123/456 | 0.756 | **0.354** | 0.0972 | 0.0489 |
| freshqa | 501–505 | **0.305** | 0.489 | 0.0580 | 0.1143 |
| stress_test | 42/123/456 | 0.847 | **0.416** | 0.0408 | 0.0136 |
| stress_test | 501–505 | **0.295** | 0.690 | 0.0147 | 0.0428 |

In 42/123/456 the file labelled `exp2` (SFT) is the confident, higher-accuracy,
badly-calibrated model and `exp3` (TSCT) is the hedging, lower-accuracy,
well-calibrated one — the expected TSCT signature. In 501–505 those roles are
exactly reversed, on every benchmark and every metric.

The magnitudes line up across the groups too: MMLU EM for `exp3` in 501–505
(0.0734) sits near `exp2` in 42/123/456 (0.0674), and `exp2` in 501–505 (0.0346)
near `exp3` in 42/123/456 (0.0299).

Two candidate explanations, and they need different fixes:

1. **Condition labels swapped for the 501–505 batch** (`exp2_v2_*` / `exp3_v6_*`
   filenames assigned the wrong way round during export).
2. **The 501–505 TSCT runs mode-collapsed into `[COND_CONFIDENT]`** — those files
   are 88.3 / 99.9 / 99.6 / 99.9 / 98.6 % `[COND_CONFIDENT]` on temporal_delta.
   That is collapse, not calibration, and §1.1 below explains why the loss as
   written has a degenerate optimum exactly there.

Explanation 2 is independently supported by the loss code and by two earlier
runs, so I lean toward it — but it does not by itself explain why the *SFT* runs
also differ so sharply between the two groups (85–99% `[CONFIDENT]` in
42/123/456 vs 50–80% `[UNKNOWN]` in 501–505). SFT is CE-only; its hedge
distribution should not swing that far on seed alone. That residual points at a
config or data difference between the two batches.

**Either way, the two seed groups are not exchangeable and must not be pooled or
reported as "n=8" until David explains what differs between them.** This is the
blocking item, ahead of any loss redesign.

---

## 1. Why the current loss over-hedges

Reasoning from `eval_toolkit/corrected_tcl_training.py` (the reference
implementation; `TCL_LOSS_HISTORY.md` notes David's live v6 loss was never seen
in-session, so items 1.1–1.4 must be confirmed against his current file).

### 1.1 `[COND_CONFIDENT]` is an unpenalised free zone — the degenerate optimum

```python
def l_over_correct(hedge_probs, answer_is_correct, alpha=0.5):
    p_confident = hedge_probs[:, 0]                 # index 0 only
    ...
def l_under_correct(hedge_probs, answer_is_correct, beta=0.5):
    p_hedged = hedge_probs[:, 2:].sum(dim=-1)       # indices 2,3 only
```

`L_over` sees only index 0 `[CONFIDENT]`. `L_under` sees only indices 2–3
`[TEMPORAL_HEDGE]`, `[UNKNOWN]`. **Index 1 `[COND_CONFIDENT]` appears in neither
term.** A model that puts all hedge mass on index 1 drives both calibration
penalties to exactly zero regardless of whether its answer is right or wrong.
The only counter-pressure is `R_hedge` (λ3 = 0.3, vs λ1 = λ2 = 0.5), so the
escape is cheap whenever the correctness signal is noisy.

This predicted attractor is visible in the data at three separate points:

- Run 1 [doc]: "TSCT piling into `[COND_CONFIDENT]` (3267 vs SFT's 32)"
- Run 2 [doc]: MMLU TSCT 61.9% `[COND_CONFIDENT]`
- v6 seeds 501–505 [recomputed]: 88–99.9% `[COND_CONFIDENT]` on temporal_delta,
  94–99.4% on MMLU

It is the single most reproducible failure signature in the project, and it is a
direct consequence of the index ranges in those two functions. It has not been
named in `TCL_LOSS_HISTORY.md`'s bug lineage.

### 1.2 The correctness proxy is still batch-relative

```python
answer_is_correct = (per_ex_nll < per_ex_nll.median()).float()
```

The docstring of that file claims to have fixed "fake correctness," but the
change was `mean` → `median`. **Median is still a batch statistic, and worse: it
forces exactly 50% of every batch to be labelled correct, by construction**,
independent of how well the model is actually doing. `frac_correct_proxy` logged
to W&B will read ≈0.5 in every step, which makes it useless as the health check
that `TCL_LOSS_HISTORY.md` recommends ("check the `correct` rate on MMLU
examples specifically during training — if it's ~0%, the proxy is still
broken").

Bug 2 as documented is therefore **not fixed in the reference implementation**.
MMLU items carry higher token NLL than short Wikidata facts, so in a mixed batch
they sit above the median → labelled wrong → `L_over` suppresses `[CONFIDENT]`
on exactly the stable facts the regression check measures.

### 1.3 The proxy is contaminated by the hedge token it supervises

`per_ex_nll` averages cross-entropy over **all** non-masked label positions. The
training format is `"Question: … Answer: … Hedge: [X]"`, so the hedge token is
inside the averaged span. Predicting the hedge well lowers `per_ex_nll`, which
pushes the example below the median, which relabels it "correct", which changes
which calibration term fires on it. Correctness and hedge choice are coupled
through the proxy — a feedback loop, not a supervision signal.

### 1.4 Off-by-one on the hedge logit position

```python
hedge_pos = <index of last non-masked label>          # a *label* index
pos_logits = logits[torch.arange(B), hedge_pos]       # logits[t] predicts token t+1
```

In a causal LM `logits[:, t]` is the distribution over the token at position
`t+1`. To score the hedge token at label index `hedge_pos` the code needs
`logits[:, hedge_pos - 1]`. As written it reads the distribution over the token
*after* the hedge (EOS/pad). The same file gets this right 10 lines later in its
own `shift_logits` / `shift_labels` block, so it is internally inconsistent.

If this propagated into David's live loss, then `ĉ` and `R_hedge` have been
supervising the wrong position for the whole project, and the hedge head is
being trained only indirectly through CE. **This is the first thing to check in
his current file** — it is cheap to check and would reframe everything.

### 1.5 The eval metric structurally rewards hedging at these accuracy levels

`eval_pipeline.compute_ece` maps each hedge to a fixed scalar (0.95 / 0.75 /
0.45 / 0.10), so ECE is a pure function of *hedge distribution × correctness
rate*. I evaluated the pure strategies with the project's own `compute_ece`:

| true accuracy | all CONFIDENT | all COND_CONF | all TEMPORAL_HEDGE | all UNKNOWN | ECE-optimal |
|---|---|---|---|---|---|
| 0.03 | 0.921 | 0.721 | 0.421 | **0.071** | UNKNOWN |
| 0.10 | 0.848 | 0.648 | 0.348 | **0.005** | UNKNOWN |
| 0.30 | 0.652 | 0.452 | **0.152** | 0.198 | TEMPORAL_HEDGE |
| 0.60 | 0.351 | 0.151 | **0.149** | 0.499 | TEMPORAL_HEDGE |
| 0.75 | 0.198 | **0.007** | 0.302 | 0.652 | COND_CONFIDENT |
| 0.85 | **0.096** | 0.104 | 0.404 | 0.754 | CONFIDENT |
| 0.95 | **0.006** | 0.200 | 0.500 | 0.850 | CONFIDENT |

Analytically this is exact rather than empirical: for a pure strategy with
confidence `c` at true accuracy `a`, ECE = `|a − c|`, so the optimal token
switches at the midpoints between the four scalars — `[UNKNOWN]` below 0.275,
`[TEMPORAL_HEDGE]` to 0.60, `[COND_CONFIDENT]` to 0.85, `[CONFIDENT]` above 0.85.

`[CONFIDENT]` therefore only becomes the ECE-minimising choice above ~85%
accuracy. Measured MMLU accuracy (EM 0.030–0.067) is more than an order of
magnitude below that, and there the ECE-optimal policy is to emit `[UNKNOWN]` on
literally everything. **The model is not malfunctioning when it hedges MMLU —
under this metric, at this accuracy, hedging is correct.** Any loss term that
restores `[CONFIDENT]` on stable facts is pushing against the metric, and will
raise ECE, unless accuracy rises first.

### 1.6 ECE and the accuracy table use different definitions of "correct"

`compute_ece` reads the `correct` field; `compute_accuracy_metrics` recomputes
strict normalised EM. In David's files these disagree badly. On
`exp3_v6_seed42_mmlu` [recomputed]: `correct=True` on 271/4048 (6.7%) but strict
EM on 128/4048 (3.2%). The `correct` field is substring-flavoured and generates
false positives:

| gold | predicted | `correct` |
|---|---|---|
| `1` | `11/15` | true |
| `2` | `-2, 0, 2` | true |
| `2` | `(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)_{11` | true |

This is the same defect `BASELINES_STATUS.md` flags in Surya's notebook
("`is_correct` uses substring match against long paragraph gold answers"),
present in David's inference output too. **Every ECE number in the project is
anchored to this inflated correctness rate, while the accuracy table reports the
strict one.** The two headline metrics are not measuring the same event.

### 1.7 38% of MMLU answers contain a hedge token inline

[recomputed] Across the v6 MMLU files, 36% (SFT) / 38% (TSCT) of
`predicted_answer` strings contain a bracketed hedge token inside the answer
text, often with run-on repetition:

```
gold:  "abelian group"
pred:  "Abelian group.\nHedge: [CONFIDENT] their sentence [CONFIDENT] their [CONF"
```

Generation is not terminating at the answer, and the hedge parser is taking a
different token than the one in the text (that record's `predicted_hedge` is
`[TEMPORAL_HEDGE]`). Strict EM fails on these even when the answer is right,
which is a large part of why MMLU EM sits near 3–7% for a LLaMA-3 8B. **The
accuracy-regression criterion is currently being adjudicated on a number
dominated by a formatting bug.**

---

## 2. Candidate fixes

Ordered by what I would do first. Each states mechanism, expected effect on both
metrics, risk, and the specific verification.

### Fix 0 — Resolve the data-integrity blockers (not a loss change)

**Mechanism.** Four things, none requiring GPU: (a) David confirms what differs
between seed batches 42/123/456 and 501–505 and whether `exp2`/`exp3` are
correctly assigned in the second batch; (b) fix the generation stop condition
and hedge parser so the hedge stops leaking into `predicted_answer`; (c) replace
the substring `correct` field with strict EM computed by `eval_pipeline`, or
document the rule and use one definition for both ECE and the accuracy table;
(d) re-run `per_benchmark_breakdown.py` and update `RESULTS_LOG.md` to say n=3
where the numbers are n=3.

**Expected effect.** No model change. MMLU/FreshQA EM should rise materially
once (b) lands — how much is the key unknown, and it determines whether §1.5's
tension is even binding. ECE will rise for everyone once (c) lands, because the
inflated `correct` rate currently flatters low-confidence bins.

**Risk.** (c) will make the ECE numbers look worse across the board, including
TSCT's. That is a correction, not a regression, but it needs to be framed that
way in the paper before anyone sees both versions.

**Verification.** `url_leakage_check.py` style pass for inline hedge tokens;
`per_benchmark_breakdown.py` on the corrected files; the regression check
verdict recomputed with a consistent correctness definition.

**Cost:** low, no training. **Impact:** decisive — it determines whether there
is an accuracy regression at all.

---

### Fix A — Close the `[COND_CONFIDENT]` free zone (ordinal calibration penalty)

**Mechanism.** Replace the two index-sliced terms with a single distance-weighted
penalty over the ordinal hedge scale, so every token carries a cost proportional
to how far its confidence sits from the observed correctness. With
`c = [0.95, 0.75, 0.45, 0.10]` and `y ∈ {0,1}` the correctness label:

```
L_cal = Σ_k  p_k · |c_k − y|          # expected calibration cost, no free index
```

when wrong (`y=0`) this penalises high-confidence tokens in proportion to `c_k`;
when right (`y=1`) it penalises hedging in proportion to `1−c_k`. `[COND_CONFIDENT]`
is charged 0.75 when wrong and 0.25 when right — no longer free. Keep `R_hedge`
as-is.

**Expected effect.** Removes the degenerate attractor that produced the 88–99.9%
`[COND_CONFIDENT]` collapse in seeds 501–505 and the 61.9% in Run 2. ECE should
be *roughly neutral to slightly better* (the term is a differentiable surrogate
for ECE itself). Accuracy unaffected directly — this term does not touch the
answer tokens. The gain is distributional sanity, not a number.

**Risk.** Low. It makes the loss a closer surrogate for the eval metric, which
also means it inherits §1.5: at 3% accuracy it will still drive everything
toward `[UNKNOWN]`. Fix A alone does not restore confidence — it stops the
collapse into the unpenalised token.

**Verification.** Hedge distribution per benchmark from
`per_benchmark_breakdown.py`. Success = no single hedge token exceeds ~70% on
any benchmark, and the seed-to-seed variance in hedge distribution collapses
(currently `[COND_CONFIDENT]` ranges 5%→99.9% across v6 seeds).

**Cost:** low (one function). **Impact:** high — kills the most reproducible bug.

---

### Fix B — Absolute + per-source correctness threshold, with a live diagnostic

**Mechanism.** This is candidates B and C from `TCL_LOSS_HISTORY.md`, combined,
because neither is sufficient alone. Replace
`per_ex_nll < per_ex_nll.median()` with a fixed threshold calibrated per source
group:

```
answer_is_correct = per_ex_nll < TAU[source]     # source ∈ {wikidata, mmlu, ...}
```

`TAU` fitted once on the val set as the NLL that best separates known-EM-correct
from known-EM-wrong *within each source*. Also: exclude the hedge-token position
from `per_ex_nll` (fixes §1.3), and log `frac_correct_proxy` **split by source**,
not pooled.

**Expected effect.** Directly targets documented Bug 2. MMLU examples stop being
labelled wrong by construction, so `L_over` stops suppressing `[CONFIDENT]`
there. Expect MMLU `[CONFIDENT]` share to rise from ~11% (n=3 v6) toward the
training prior. ECE on MMLU will rise if accuracy stays low (§1.5) — that is the
tension, quantified in §3.

**Risk.** Medium. `TAU` is a new hyperparameter fitted on val, and NLL
thresholds transfer poorly across sequence lengths — MMLU questions are much
longer than Wikidata ones, which is what caused the original problem. Per-source
`TAU` mitigates this but adds a `source` field requirement to the training data.

**Verification.** The diagnostic `TCL_LOSS_HISTORY.md` already prescribes:
`frac_correct_proxy` restricted to MMLU rows during training. If it is ~0%, the
proxy is still broken; target is that it tracks actual MMLU EM within a few
points. Then MMLU `[CONFIDENT]` share in `per_benchmark_breakdown.py`.

**Cost:** medium (needs a val-set fit + a data field + a retrain).
**Impact:** high — it is the documented root cause.

---

### Fix C — Volatility-gated asymmetry instead of correctness-gated

**Mechanism.** Note that `R_hedge` already supervises the *right* thing: the gold
hedge for the fact's volatility class, which is ground truth and needs no proxy.
The correctness-gated terms are what keep going wrong. So: weight `R_hedge` by
inverse class frequency (immutable is 0.06% of training per `DATASET.md`), and
apply the calibration terms **only where the proxy is trustworthy** — i.e. gate
`L_over`/`L_under` to fast/slow items and let immutable items be supervised by
`R_hedge` alone.

```
L = CE + w_vol · R_hedge + 1[vol ≠ immutable] · (λ1 L_over + λ2 L_under)
```

**Expected effect.** On stable facts the only hedge pressure becomes "match the
gold `[CONFIDENT]` label," which is exactly the desired behaviour and is immune
to the correctness proxy entirely. Should restore MMLU `[CONFIDENT]` share
strongly. ECE on MMLU rises (§1.5); ECE on temporal_delta/stress_test — the
volatile benchmarks where the calibration claim actually lives — is untouched,
because those items keep the full loss.

**Risk.** Medium-low. Relies on the volatility label being right, and on
`inject_stable_seeds.py` being re-run on post-URL-fix data (`DATASET.md` warns
the existing balanced file was built pre-fix). Also concedes that the model is
not *learning* to be confident on stable facts so much as being *told* to be —
a fair reviewer criticism, and one the paper should state rather than hide.

**Verification.** Per-volatility hedge distribution; the two unrun adversarial
stress files (`stress_stable_facts.jsonl` n=49, `stress_mixed_paragraphs.jsonl`
n=60), which exist precisely to measure this and have never been through
inference.

**Cost:** medium (retrain + data regen). **Impact:** high on the over-hedging
symptom, neutral on the calibration claim — which is the combination H1 needs.

---

### Fix D — Two-stage training: freeze the answer, then calibrate

**Mechanism.** Stage 1: plain SFT on CE only, to convergence. Stage 2: freeze the
backbone (or attach a small hedge head on detached features) and train **only**
the hedge head with the calibration terms, using *decoded* answers so
correctness is real exact-match rather than an NLL proxy — candidate A in
`TCL_LOSS_HISTORY.md`, made tractable by doing it after the answer model is
fixed.

**Expected effect.** Structurally guarantees no accuracy regression: the answer
distribution is frozen, so EM cannot move. Eliminates the correctness proxy
entirely, which removes Bugs 2, 3 and §1.3 in one step. Calibration quality
should be at least as good, since the hedge head now sees ground-truth
correctness.

**Risk.** Highest implementation cost, and it partially concedes H2 ("TCL is the
critical component") — a frozen-backbone hedge head is closer to a learned
post-hoc calibrator than to a joint training objective, and a reviewer may say
it is temperature scaling with extra steps. It also needs a decode pass over the
training set to get correctness labels, which is real GPU time.

**Verification.** EM identical to SFT by construction (assert it). Then ECE and
hedge distribution across all four benchmarks + the stress sets.

**Cost:** high. **Impact:** highest ceiling, and the cleanest paper story if it
works — "joint training is unstable, staged training is not" is a legitimate
finding.

---

## 3. The tension, quantified

The request was to not hand-wave this, so: **at the currently measured accuracy
levels the tension is not a tradeoff, it is near-total opposition.**

From §1.5, moving a fraction `p` of probability mass from `[UNKNOWN]` (0.10) to
`[CONFIDENT]` (0.95) on a benchmark whose true accuracy is `a` changes ECE by
approximately `p · (|a − 0.95| − |a − 0.10|)`.

- On **temporal_delta**, TSCT accuracy is ~4% [recomputed, n=3 group: EM 0.0055;
  `correct` 0.0058]. Restoring `[CONFIDENT]` on just 10% of items costs
  **+0.085 ECE** — roughly a fifth of the entire claimed reduction, for a tenth
  of the mass. There is no version of this that is worth it. Volatile facts
  *should* be hedged; the method is right there.
- On **MMLU**, if the true accuracy after fixing §1.7 turns out to be ~0.6
  (plausible for LLaMA-3 8B on MMLU, which scores in the 60s), then the same
  shift *reduces* ECE by `p · (0.35 − 0.50) = −0.15p`. Confidence and calibration
  stop being opposed and start agreeing.
- If MMLU accuracy really is 3–7%, then no fix in §2 can make the model both
  confident and calibrated on MMLU, because a model that is right 5% of the time
  **should not** be confident. The honest conclusion in that case is that the
  over-hedging on MMLU is not a calibration bug at all — it is the calibration
  objective working correctly on top of a broken answer model.

This is why Fix 0 outranks every loss change. **The entire "restore confidence
on stable facts" project is only coherent if the model is actually right about
stable facts, and we do not currently have a trustworthy measurement of
that.** §1.7 gives strong reason to think the current measurement is wrong.

Net recommendation on the tradeoff: accept ECE increases on MMLU/FreshQA, hold
the line on temporal_delta and stress_test. Report ECE per benchmark rather than
pooled, and state plainly that the calibration claim is about volatile facts
while the no-regression claim is about stable ones. Those are different
populations and pooling them is what makes the result look like a wash.

---

## 4. Ranking

| Rank | Fix | Impact | Cost | Needs a retrain? |
|---|---|---|---|---|
| 1 | **Fix 0** — data/eval integrity | Decisive | Low | No |
| 2 | **Fix A** — close the free zone | High | Low | Yes (cheap, 1 seed to check) |
| 3 | **Fix C** — volatility-gated asymmetry | High on symptom | Medium | Yes |
| 4 | **Fix B** — absolute per-source TAU | High, root cause | Medium | Yes |
| 5 | **Fix D** — two-stage frozen backbone | Highest ceiling | High | Yes + decode pass |

Fixes A, B and C are compatible and I would land A+C together, then B if the
proxy is still needed. D is the fallback if joint training stays unstable.

---

## 5. What I need, and from whom

**From David (training/inference) — highest priority first:**

1. **The current v6 loss file.** `TCL_LOSS_HISTORY.md` records that the v6
   change was never seen in-session, so §1.1–§1.4 are diagnosed against the
   reference implementation and must be confirmed. Specifically: the index
   ranges in `L_over`/`L_under`, the exact `answer_is_correct` line, and the
   `logits[..., hedge_pos]` indexing.
2. **An explanation of the 42/123/456 vs 501–505 split**: same data, same config,
   same code? And confirmation that `exp2`=SFT / `exp3`=TSCT holds in the
   501–505 export.
3. **Fix the generation stop + hedge parser** so `predicted_hedge` is not also
   sitting inside `predicted_answer` (38% of MMLU rows), and re-emit v6
   predictions. No retrain needed — this is decode-time.
4. **Add `predicted_volatility`** to inference output (unblocks committed metric
   5). Note `eval_pipeline.volatility_discrimination` also reads
   `true_volatility`, which no file has either — it needs both keys, or the
   function needs to fall back to `volatility`. `run_results.py` gates on
   `true_volatility` while `week5_eval.py` gates on `predicted_volatility`; they
   should agree.
5. **Run the two adversarial stress files** — `stress_stable_facts.jsonl` (49)
   and `stress_mixed_paragraphs.jsonl` (60). 109 items, minutes of GPU. These
   are the only clean measurement of over-hedging on stable facts we have, and
   they have never been run.
6. **One ablation run of Fix A**, single seed, to confirm the
   `[COND_CONFIDENT]` collapse disappears before committing to a full 8-seed
   sweep.

**From you (Jason):**

7. A decision on the significance-reporting scope — though note this is now
   downstream of §0: with all 8 seeds the comparison is p=0.494 either way, so
   the α=0.05 vs α=0.0071 question only matters once the seed-group issue is
   resolved.
8. Whether to correct `RESULTS_LOG.md` / `PAPER_STATUS.md` now to say n=3, or
   wait for David's answer on the seed split. My recommendation is to annotate
   now — the current text asserts n=8 in a way the files do not support.

**From Aarav (data):** whether the training data carries a `source` field
suitable for Fix B's per-source `TAU`, and whether `inject_stable_seeds.py` can
be re-run on post-URL-fix data as `DATASET.md` requires.

---

## 6. Ambiguities and contradictions found in the bundle

Flagged rather than smoothed over, per the brief.

1. **`RESULTS_LOG.md` Run 3 and `CLAUDE.md` label an n=3 result as n=8.** (§0.1)
2. **`corrected_tcl_training.py`'s docstring claims to fix "fake correctness"
   but the code is still batch-relative** (`median` instead of `mean`). (§1.2)
3. **The two documents number the bugs differently.** `TCL_LOSS_HISTORY.md` uses
   Bug 1 = gradient cut, 2 = batch-relative proxy, 3 = v6 overcorrection.
   `corrected_tcl_training.py` uses Bug 1 = hardcoded gold hedge, 2 = fake
   correctness, 3 = fixed hedge position. "Bug 3" means two different things
   depending on which file you are reading.
4. **Sign convention.** Docs write `TCL = CE + λ1·L_over + λ2·L_under − λ3·R_hedge`
   with `R_hedge` as a *reward*; the code computes `+ γ·cross_entropy(...)`,
   a *penalty*. These are equivalent but the notation invites a sign error —
   worth fixing in the paper's methods section.
5. **`week5_eval.py` does not auto-detect seeds** — `task_hedge_scoring`
   hardcodes `for seed in (42, 123, 456)` and the other two tasks use seed 42
   only. `CLAUDE.md` states "Scripts auto-detect seeds." So the hedge
   appropriateness figure (SFT 2.514 → TSCT 4.227) is an n=3, seeds-42/123/456
   number even when pointed at the 8-seed folder — the same subset as §0.1, by
   coincidence of the hardcoded list.
6. **Hedge appropriateness cannot detect over-hedging on temporal_delta.** Per
   `hedge_quality_rubric.py`, for a `fast` fact `[UNKNOWN]` scores 4/5 while
   `[CONFIDENT]` scores 2/5. The test set is ~90% fast and 0% immutable
   (`DATASET.md`), so the metric rewards hedging up. The 2.514 → 4.227
   improvement is partly the same behaviour that fails the regression check,
   measured on a scale that likes it. It should be reported alongside the stress
   sets, not alone.
7. **`adapt_predictions.py` maps any unrecognised hedge to `[UNKNOWN]`**
   (`HEDGE_MAP.get(raw_hedge, "[UNKNOWN]")`), the lowest-confidence token. Given
   §1.7's parsing problems, any file passed through the adapter could have an
   inflated `[UNKNOWN]` rate that lowers ECE and mimics over-hedging. Worth
   confirming whether the v6 files went through it.
8. **`hedge_quality_rubric.py` has a duplicate dict key** —
   `("[TEMPORAL_HEDGE]", "[UNKNOWN]")` is defined twice (lines 160 and 169),
   both as 4, so behaviour is unaffected. Cosmetic.
9. **`ece_bar_chart.png` from `run_results.py` aggregates temporal_delta across
   whatever seeds are present**, so on the current folder it would render the
   pooled n=8 number (TSCT worse), not the n=3 number in the docs. Any figure
   regenerated now will silently disagree with the text. (Separate from, and in
   addition to, the known `paper_plots.py` placeholder-figure warning — I have
   not touched those.)

---

## 7. What I did not do

- Did not modify any file in the bundle or in `~/Downloads/david_v6`.
- Did not open, use, or regenerate the placeholder figures from `paper_plots.py`.
- Did not write loss code — this is analysis and proposal only, per the brief.
- Did not train or run inference; every recomputed number came from the existing
  `eval_toolkit` scripts run on existing prediction files on CPU.
