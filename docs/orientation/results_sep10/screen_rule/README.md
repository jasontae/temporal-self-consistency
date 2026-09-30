# The form screen under one decision rule (2026-09-30)

**Plan:** `../../PLAN_screen_rule.md`, committed before anything here was computed.

**Files:**
- `screen_rule.py`: the script. CPU only, and it reads existing score files; no model is run. Run it from the repo root: `python3 docs/orientation/results_sep10/screen_rule/screen_rule.py`.
- `screen_rule.json`: every number.
- `screen_rule.md`: the tables.

## The rule

- **Stratified AUROC.** The model's AUROC is computed only over pairs that share the value of the strongest no-model feature (2 strata for a binary feature, quintiles for a count). Strata are pooled by their within-stratum pair count.
- **Firing.** The screen **fires** when the 95% bootstrap CI of this stratified AUROC includes 0.5, or excludes it on the opposite side.
- **Precondition.** If the unconditional model AUROC's CI already includes 0.5, the verdict is **no signal**.
- **Set verdict.** Taken from the panel mean (Qwen3-4B-Instruct, Qwen2.5-7B, Qwen3.5-27B, Qwen3.6-27B and Qwen3.6-35B-A3B), bootstrapped jointly.
- **Secondary comparison, reported but not decisive.** The oriented cue AUROC minus the model AUROC, with a paired bootstrap.

## Verdicts

| set | strongest feature (AUROC) | model, panel mean | cue-stratified, panel mean | **primary verdict** | models fired | secondary: cue − model |
|---|---|---|---|---|---|---|
| First stress set (22 distinct claims; 4 panel models) | name length, capitalised span (0.887) | 0.894 [0.760, 0.985] | 0.796 [0.450, 1.000] | **fires** | 4/4 | −0.006 [−0.152, 0.137]: model does not beat cue |
| EverGreenQA (200/200) | past-tense marker (0.665) | 0.591 [0.546, 0.633] | 0.563 [0.512, 0.611] | **passes (narrowly)** | 3/5 | +0.074 [0.023, 0.127], p = 0.006: **cue beats model** |
| FreshQA, matched subset (145 never / 116 fast) | temporal cue words (0.774) | 0.577 [0.509, 0.650] | 0.644 [0.559, 0.728] | **passes** | 0/5 | +0.197 [0.111, 0.285], p < 0.001: **cue beats model** |
| Form-matched set, staleness (106 entities) | name length (0.511) | 0.438 [0.396, 0.481] | 0.454 [0.407, 0.499] | **passes (narrowly)** | 1/5 | model 0.562 vs cue 0.511, Δ −0.050 [−0.111, 0.010] |
| Form-matched set, truth | name length (0.508) | 0.835 [0.794, 0.873] | 0.833 [0.786, 0.875] | **passes** | 0/5 | model beats cue by 0.33 |
| Form-matched set, regime (no claim made) | name length (0.574) | 0.548 [0.502, 0.593] | 0.547 [0.495, 0.597] | **fires** | 1/5 | −, n.s. |

## Reading

**The two rules disagree.** The disagreement is not arbitrary: they answer different questions.
- **"Does the cue beat the model?"** asks which classifier is stronger.
- **"Does the model survive conditioning on the cue?"** asks whether the cue explains the model's separation.

Section 4.3's own wording, "the no-model baseline explains how the observed separation could arise", is the second question. The primary rule is therefore the one that matches the paper's claim.

**First stress set: fires under both rules.**
- The strongest feature under the fixed selection rule is the length of the capitalised name span (0.887). It is a sharper version of the person-name regex.
- Conditioning on the paper's person-name detector (0.833) gives the same verdict: the panel mean is 0.713 [0.425, 0.950], and every model's interval contains chance, with 40 within-stratum pairs. This matches Appendix J.
- The firing rests on low power: 22 claims, with only 27 to 40 pairs left inside strata.

**EverGreenQA: passes on the panel mean, but only narrowly.**
- The lower bound is 0.512. Model by model, only Qwen2.5-7B and Qwen3.5-27B pass; the other three fire.
- The past-tense stratum holds 71 evergreen and 5 mutable questions. The verdict therefore rests almost entirely on the 324 questions with no past-tense marker, where the models reach 0.53 to 0.58.
- The paper's "holding it fixed removes about a quarter" (0.592 → 0.568, item-weighted) is reproduced as 0.591 → 0.563 (pair-weighted).

**FreshQA: passes, and the cue suppresses the model's signal rather than explaining it.**
- Almost all cue-bearing questions are fast-changing (66 fast, 3 never). Within the fast class, questions with a cue word get *higher* PMI (e.g. Qwen3.5-27B, mean 1.62 vs 1.04), probably because "current/latest" in the question primes the answer.
- Among the 192 questions without a cue word, every model separates never-changing from fast-changing at 0.637 to 0.657.
- The cue is a much better classifier than the model (+0.197), but it is not the source of the model's separation.
- **Sensitivity (not planned):** scoring by `mean_logprob` instead of `pmi` raises the unconditional AUROC to 0.64–0.69. The stratified panel mean is 0.612 [0.531, 0.691], so FreshQA still passes, although 2 of 5 models fire.

**Form-matched set.**
- **Truth** passes clearly.
- **Staleness** passes, with an upper bound of 0.499. It survives every feature in the robustness table, but only just.
- **Regime** fires on name length, which matches the paper's note about "a small residual from name length". The paper claims no regime signal, so this costs nothing.

**The secondary rule on staleness shows its flaw.**
- The model (0.562 in its claimed direction) does not significantly beat a chance-level cue (0.511).
- Taken as the decision rule, the secondary would therefore "fire" on staleness, and on any weak signal, whether or not form is involved.

**The consistent answer for the paper:**
- Under the single primary rule, the screen fires on our first stress set and passes on both published benchmarks. **FreshQA no longer fires.**
- If the paper instead keeps "cue beats model" as the rule, then EverGreenQA fires as well (p = 0.006). In that case no published benchmark passes, and the stress set's firing (Δ ≈ 0) would become the same kind of verdict as a weak-signal failure.

## Deviations from the plan

1. **Secondary comparison, staleness only.** The plan said the model score is never flipped. For staleness, the claimed direction is below chance, so the fixed-orientation Δ was meaningless (the first run printed +0.073). The script now compares the cue with 1 − AUROC for that contrast (`claimed_direction = −1`). The primary verdicts do not change.
2. **Added outputs.** Two things not in the plan were added: the FreshQA `mean_logprob` sensitivity (last row of `screen_rule.md` §1) and a descriptive per-stratum breakdown (§4a). Neither changes any primary verdict.
3. **Name length on the stress set.** The stress set has no separate answer field, so `name_len` used the capitalised-span definition, as the plan specifies.
   - That feature, not the person-name regex, is the strongest feature there.
   - With `name_len`, 66 of 2,000 bootstrap draws (3.3%) had no within-stratum pairs and were dropped. This is under the plan's 5% flag.
4. **Constant features on the form-matched set.** `person_name`, `past_tense` and `temporal_cue` are constant there (every claim is "The ROLE of ENTITY is PERSON.") and are excluded, as planned.
