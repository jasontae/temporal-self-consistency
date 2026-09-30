# Pletenev adjudication: results memo for the main text (directive 3, worker B)

This is a results memo, not paper prose. It gives the facts and numbers that §6's two or three main-text sentences should rest on.

## What Pletenev et al. (2025) claim

Checked against the paper itself: arXiv 2505.21115; ACL Anthology `2025.emnlp-main.434`, EMNLP 2025, pp. 8603–8620. The text was extracted from the arXiv PDF.

- **Measurement (§4.2).** Perplexity and mean token entropy of greedy generations on a balanced 400-question subset (200 evergreen, 200 mutable). They report the correlation of the gold evergreen label with each uncertainty score, across 11 open models.
- **Result (Table 3).** Correlations run from 0.17 to 0.35: the minimum is Phi-3 medium 4k, entropy 0.17; the maximum is Mistral 7B, entropy 0.35. All are reported significant.
- **Size, main text:** "We also observe a weak trend suggesting that larger models correlate more strongly with evergreen-ness". The takeaway box says "slightly stronger trends in larger models".
- **Size, their own Appendix E (Table 9, McFadden pseudo-R² 0.012–0.137):** "size does not correlate clearly with predictive performance; smaller models sometimes match or outperform their larger counterparts".
- **Conclusion:** "uncertainty signals capture some temporal information, but are noticeably weaker than explicit verbalized judgments".

So their size claim is already hedged within their own paper. The main text calls it weak; the appendix finds no clear correlation.

## What we find

**1. On their data, with their measure** (rebuilt Table 3, answers only; `README.md` in this folder, `table3_rebuild.md`):
- We reproduce their magnitudes. Without gpt-oss, r(PPL) runs from +0.11 to +0.19, at or just below their 0.17–0.35 band.
- Holding question form fixed (the past-tense marker) lowers mean pooled AUROC from 0.592 to 0.568, removing about a quarter of the above-chance excess. Part of their signal is surface form.
- No size trend is estimable. The within-form contrast of ≥20B minus ≤10B is −0.0055 without gpt-oss, and only two models are at or below 10B. The vintage-matched Qwen3.5/3.6 pair differs by 0.038, about 7× that contrast.

**2. On our form-matched set, with paired PMI** (`../inverse/`):
- The apparent volatility (regime) separation *falls* as factual competence rises. The slope is negative under every build, entity-bootstrap CI [−1.57, −0.33], and split-half 100% negative.
- A null simulation with temporal sensitivity held fixed gives slopes centred at +0.09, with 95% of slopes in [−0.36, +0.46] (`../capability/`).
- By size, the five ≤8B models sit at regime AUROC 0.568–0.640, and the three 27–35B Qwen models at 0.505–0.521. The 20B gpt-oss sits at 0.570.

## Is this a contradiction? No. There are three differences, all measurable.

| | Pletenev et al. | ours (§7) |
|---|---|---|
| unit | a question, answered by generation | a claim, scored teacher-forced |
| form control | none (evergreen and mutable questions differ in form) | form-matched: same entity, same frame, a person name in every arm |
| name-frequency control | none | PMI against the same name in a neutral frame |
| pairing | pooled across questions | paired within entity (sign test, random effects) |
| score | correlation of uncertainty with the label | AUROC of PMI (regime) as a function of truth AUROC |

- **Their pooled signal contains a form component**, which our form stratification of their own data removes (about a quarter).
- **Our design removes form by construction.** What is left falls with capability, which is what a knowledge account predicts: weak models find current office-holders unfamiliar, and strong models do not.
- A pooled signal that includes form cues could stay flat or rise with scale while the form-free residual falls. The two findings are compatible.
- Their size claim does not survive their own appendix, and it is not estimable on our replication. We should not claim to *overturn* it. We should say it is not supported at the scale either study has.

## Recommended main-text content for §6 (for Edward to phrase)

1. We reproduce Pletenev et al.'s magnitudes on their data (r 0.11–0.19 against their 0.17–0.35). About a quarter of the above-chance signal is question form.
2. On a form-matched, name-frequency-controlled design, the residual volatility signal falls as factual competence rises. The sign is robust; the null simulation shows the shared arm cannot produce it.
3. Their secondary size trend is weak in their main text and absent in their Appendix E, and it is not estimable at our six-model ladder. We make no size claim.

**Not to say:** "contradicts", "transient artifact of weaker base models" (dropped in the fused critique), or any size claim of our own.
