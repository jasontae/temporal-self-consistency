# Table 3 / Appendix A: the evergreen replication, rebuilt on answers

**Scripts:**
- `table3_rebuild.py` (this table)
- `../run_evergreen_answer_only.sh` (the rerun; wall times in `../evergreen_answer_only.log`, 194–823 s per model)
- `src/evaluation/score_evergreen_uncertainty.py --answer-only --subset-from`

## The defect

In the August runs, 99–100% of the 32 generated tokens for four of the six Table 3 models were reasoning preamble ("Thinking Process:", gpt-oss "analysis…"). Those models were gpt-oss-20B, Qwen3.5-27B, Qwen3.6-27B and Qwen3.6-35B-A3B. Their perplexity and entropy therefore measured the preamble, not an answer (`../MODELS.md`).

## The fix

The prompt now starts each model on its answer: an empty closed think block for Qwen3.x, and the `final` channel for gpt-oss. The rerun uses the same 400 questions. After the fix, preamble share is 0% for every model.

- **Control:** Qwen3-4B-Instruct, which never produced a preamble, reproduces its August row exactly (400 of 400 answers identical).
- **Qwen2.5-7B** keeps its August row. It also answered directly, and its base checkpoint was not on disk for the rerun.

## Result (`table3_rebuild.md`)

| model | Aug pooled / within-form | rebuilt pooled / within-form |
|---|---|---|
| Qwen3-4B-Instruct | 0.5857 / 0.5806 | 0.5857 / 0.5806 (identical) |
| Qwen2.5-7B | 0.6083 / 0.5621 | (Aug row) |
| gpt-oss-20B | 0.5829 / 0.5552 | **0.3964 / 0.4147 (inverted)** |
| Qwen3.5-27B | 0.6109 / 0.5833 | 0.6083 / 0.5847 |
| Qwen3.6-27B | 0.5594 / 0.5222 | 0.5711 / 0.5471 |
| Qwen3.6-35B-A3B | 0.5765 / 0.5532 | 0.5839 / 0.5658 |
| Qwen3.8-27B (item 6) | — | 0.6295 / 0.5986 |

| Appendix A statement | Sep 2 | rebuilt, same six | rebuilt, without gpt-oss |
|---|---|---|---|
| r(PPL) range | 0.04 to 0.18 | −0.175 to +0.190 | +0.114 to +0.190 |
| mean pooled → form-stratified | 0.588 → 0.556 (its own Table 3 gives 0.559) | 0.559 → 0.543 | 0.592 → 0.568 |
| share of above-chance excess removed by form | "roughly a third" (0.32) | 0.28 | 0.26 |
| size contrast, ≥ 20B minus ≤ 10B (within-form) | −0.0179 | −0.0433 | −0.0055 |
| vintage-pair gap, Qwen3.5 vs 3.6 | 0.0611 | 0.0376 | 0.0376 |

## Reading

- **The Qwen rows move modestly and in the direction of Pletenev et al.'s magnitudes.** Their reported band is 0.17–0.35; without gpt-oss the rebuilt r is 0.11–0.19. The paper's "at or below their band" still holds.
- **"No size trend is supported at this sample size" holds in every version.** There are only two models at 10B or below.
- **Form stratification** still removes about a quarter to a third of the excess.
- **gpt-oss inverts** (0.40), with lower perplexity on mutable than evergreen questions. Forcing its `final` channel skips the analysis it is trained to produce, so this row measures gpt-oss outside its intended mode, and the August row measured its analysis text. Neither is a clean answer-uncertainty measurement. **Recommendation:** drop gpt-oss from Table 3, or report it with this caveat. The rebuilt summary without it is the cleaner comparison.
- **The vintage-pair gap shrinks** (0.061 → 0.038). The "about three times the effect" sentence becomes about 7× the without-gpt-oss size contrast (0.038 vs 0.0055), so the conclusion is unchanged.
