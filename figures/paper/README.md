# Paper figure pack (ARR, 2026-09-29)

This folder holds the data figures for the ARR version, sized for the ACL two-column template. Every figure is regenerated from committed source data by a script in `scripts/`; nothing is edited by hand. The slots follow the section map in `docs/orientation/REFRAME.md` §3 and §5.

**Rebuild everything:**

```
python3 figures/paper/scripts/make_all.py
```

It runs every figure script and the QA audit, checks the fonts with `pdffonts` (poppler), and rewrites `figure-pack-2026-09-29.zip`. It exits non-zero if any check fails.

**Figure 1 is not in this pack.** The concept/pipeline diagram went to a separate Codex + BioRender session (orchestrator, 2026-09-29). No Claude draft was made, so `fig1-draft-claude/` does not exist.

## Figures

In the "size" column, *full* is 7.0 in (`figure*`) and *col* is 3.03 in (`figure`). All scripts are in `scripts/`. All sources are under `docs/orientation/results_sep10/`.

| file | paper slot | size (in) | source data | script |
|---|---|---|---|---|
| `fig3_threshold.pdf` | Figure 3, §5 (main result) | full, 7.00 × 2.35 | `crossover/crossover.json` (summary; the curve is the closed form in `crossover/crossover_sweep.py`); `item1/threshold.json` (`contains` scoring) | `fig_threshold.py` |
| `fig3_threshold_col.pdf` | the same, single-column variant | col, 3.03 × 4.30 | same | `fig_threshold.py` |
| `fig4_staleness.pdf` | Figure 4, §7.2–7.3 | col, 3.03 × 3.15 | `inverse/inverse_matched4b.json`; `date_probe/date_probe.json`; `base_instruct/base_vs_instruct.json` | `fig_staleness.py` |
| `figD_crossover.pdf` | App. D (crossover plane and age axis) | full, 7.00 × 2.20 | `crossover/crossover.json` (+ closed forms) | `fig_crossover.py` |
| `figE_vocab.pdf` | App. E (vocabulary sweep) | full, 7.00 × 2.05 | `vocabulary/vocab.json` | `fig_vocab.py` |
| `figM_frequency.pdf` | App. M (frequency test) | col, 3.03 × 2.20 | `frequency/frequency_results.json` | `fig_frequency.py` |
| `figO_inverse.pdf` | App. O (inverse result, direction only) | full, 7.00 × 2.45 | `inverse/inverse_disjoint.json`; `inverse/inverse_disjoint4b.json` | `fig_inverse.py` |

Each PDF has a 300-dpi PNG of the same name, for the site.

**Not drawn, on purpose:**
- **ECE/Brier "table-figures":** none exist. Tables 1 and 2 stay as LaTeX tables, with their numbers in `ece_check/ece_levels.json` and `table2/table2_full.json`.
- **Vintage ladder:** the section map replaces it with base vs instruct, which is in Figure 4.
- **Sep 2 Figure 1 (Nike), Figure 2 (verifier chain), Figure 4 (TSCT pipeline):** these are Overleaf figures with no data source here.

## Caption drafts

Every number in these captions is in `docs/orientation/RESULTS_FINAL.md`; the row is given in brackets. Values printed *inside* a figure are read from its source file when the figure is built.

**Figure 3.** ECE rewards a constant once volatile-fact accuracy falls below 0.314. **(A)** TemporalDelta, exact: the ECE of the TVT-label oracle and of the constant [UNKNOWN] as volatile-fact accuracy varies, with slow-class accuracy and class mix held at the test set's values. Left of the crossover at 0.314 the constant wins. The test set sits at 0.026, where the oracle's ECE is 6.20× the constant's (0.4504 vs 0.0727; 95% CI [5.85, 6.62]) [row 1]. **(B)** FreshQA: each model's answers re-scored against answer keys of increasing date; above zero the constant wins. The three strongest models (Qwen3.5-27B, Qwen3.6-27B, Qwen3.6-35B-A3B) cross from oracle-wins to constant-wins as the key moves from 2024 to 2026, while their fast-changing accuracy falls from 0.19–0.24 to 0.09–0.11 [row 6]. Grey: the other five models. Dashed: the two TSCT adapters. For the hedge-trained adapters the constant beats the oracle by 7.2–9.5× under both keys [row 5].

**Figure 4.** All nine models' point estimates favour the expired value. Staleness AUROC (current vs expired office-holder, form-matched set; below 0.5 the expired value is less surprising) on the fixed 4-bit ladder, with 95% entity-bootstrap CIs; the mean over the nine is 0.442 [0.404, 0.477] [row 9]. Diamonds: the same models with the prefix "As of 2026," (mean 0.454 [0.417, 0.489]). A stated past date moves all nine further toward the expired value ("As of 2020," minus "As of 2026," −0.019 [−0.027, −0.012]) [row 11]. Square: Qwen2.5-7B base, at 0.428 against 0.426 for its instruct checkpoint (Δ +0.003 [−0.022, +0.028]), so the preference is present before post-training [row 10].

**Figure D.** The crossover plane and the age axis on TemporalDelta. These curves are exact, because neither policy looks at which items are correct. **(A)** The sweep of Figure 3A with the best constant in the four-level vocabulary added (dashed); the crossover is at 0.314 [row 1]. **(B)** The plane of volatile and slow-class accuracy: where constant [UNKNOWN] beats the TVT-label oracle, where the oracle beats [UNKNOWN] only, and where it beats every constant (around the asserted levels, star). The test set lies in the first region. **(C)** An answerer that knows the world as of a given year, scaled by a knowledge rate k and scored against the key's own validity intervals; above zero the constant wins.

**Figure E.** The constant's margin belongs to the confidence vocabulary, not to the model. **(A)** Lowering the vocabulary's floor from 0.10 to 0.01 raises the oracle/constant ECE ratio from 6.2× to 26× on the released key (65× on the repaired key) [rows 1, 4]. **(B–D)** TSCT seed 0: a recalibrated log-probability score (AUROC 0.82–0.88 across runs) ties a base-rate constant on ECE (about 0.01); only AUROC and resolution separate them [row 4]. * = 5-fold cross-fitted.

**Figure M.** The pre-registered frequency test finds no relation. Per entity (n = 106): the log ratio of expired to current entity–holder co-occurrence in Dolma v1.7, against the expired-value preference (PMI, z-scored within model, mean of the nine models). The line is the primary slope, +0.038 (permutation p = 0.12). With the pre-registered controls (token length, relation, entity frequency, time in office, prominence) the coefficient is −0.02 [row 22].

**Figure O.** The inverse result is a direction, not a finding. Regime AUROC against truth AUROC for the nine admitted models (disjoint false-founder build), with 95% entity-bootstrap CIs and an OLS fit. **(A)** Mixed quantization: slope −1.37, permutation p = 0.013. **(B)** Fixed-quantization ladder: slope −0.27, p = 0.40. The slope is negative in every variant, and across the fixed-quantization variants it runs from −0.27 to −0.73 (p = 0.08–0.40) [row 14].

## Style (`scripts/style.py`, shared by every figure)

- **Size:** column 3.03 in, full width 7.0 in. PDFs are saved at exactly the figure size, with no tight-bbox re-cropping, so they drop in at `\linewidth` without rescaling.
- **Fonts:** Times New Roman, with STIX for the few math glyphs; STIXGeneral is the fallback on machines without Times. Text is 8 pt, ticks and legends 7 pt, the smallest labels 6.5 pt, and nothing is below 6 pt. Fonts are embedded as TrueType (`pdf.fonttype 42`).
- **Palette:** Okabe–Ito, assigned by role and kept the same in every figure:

| role | line colour | text colour |
|---|---|---|
| TVT-label oracle | blue `#0072B2` | `#005A8C` |
| constant policies | vermillion `#D55E00` | `#B04A00` |
| informative scores (log-prob) | bluish green `#009E73` | — |
| TSCT | reddish purple `#CC79A7` | `#9A4A76` |
| a second condition ("As of 2026,") | orange `#E69F00` | — |

  Text in a role colour uses the darker shade, which is at least 4.5:1 on white and on the tinted bands.
- **Output:** PDF (vector) plus a 300-dpi PNG.

## QA (visual-artifact-builder loop, adapted to print figures)

- **Automatic audit, every build** (`style.audit`): every drawn text item is checked at the real page size for size under 6 pt, overlap with another text item, and running past the page edge. `make_all.py` adds the `pdffonts` check: every font embedded and TrueType.
- **By eye:** each PDF was rendered to PNG with `pdftoppm` at 200–250 dpi and checked at print size, up to three rounds per figure. The QA renders stay in the session scratchpad, not in git.
- **Fixed during QA:**
  - `bbox_inches="tight"` had cropped the full-width figures to about 5.9 in and the staleness figure to 3.26 in. They are now exact.
  - Colliding end-labels in 3B are now spread, each with a marker key.
  - Region colours in D-B were inverted by a mapping bug; the regions are now correct and labelled directly.
  - Labels in O now clear the points and bars, with a white halo where they cross the fit line or a CI bar.
  - Legends that covered data moved.
  - Low-contrast role-coloured text now uses the darker shades above.
- **After Edward's PDF read-through (2026-09-29):** every leader line and arrow is gone; "test set" labels sit beside their markers; Figures 3 and O have plain panel titles (`style.panel_title`); O's model names are placed by `style.place_labels` (greedy, avoids points, the fit line and each other; near-coincident points share one label); Figure 4 has a plain-words legend, the base checkpoint in its own ruled row, a labelled chance line, and is 0.40 in taller (3.15 in).
- **Final result:** 0 audit problems across 7 PDFs; all fonts embedded TrueType.
