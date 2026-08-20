# Figures

## `results/` — real figures

Generated from actual prediction files by the scripts in `src/figures/`.
Every number traces to a prediction file; nothing is hardcoded.

| file | what it shows |
|---|---|
| `results_fig1_seed_split.png` | ECE by benchmark, SFT vs TSCT, seed batches shown separately |
| `results_fig2_hedge_distributions.png` | Hedge distribution per condition against the gold target |
| `results_fig3_metric_degeneracy.png` | ECE of each constant hedging policy vs true accuracy |
| `results_fig4_informativeness.png` | AUROC of confidence vs correctness; stable/volatile confidence gap |
| `fig1_ece_vs_refusal.png` | ECE per condition against the oracle and constant-refusal references |
| `fig2_hedge_distribution.png` | Ablation hedge distributions |
| `fig3_confidence_direction.png` | Mean confidence on slow facts minus fast facts |
| `fig4_ece_vs_discrimination.png` | ECE against volatility discrimination |

Regenerate:

```bash
python3 src/figures/make_results_figs.py
python3 src/figures/make_ablation_figs.py
```

## Everything else in this directory is PLACEHOLDER output — do not use

`benchmark_comparison.png`, `calibration_curves.png`, `ece_bar_chart.png`,
`hedge_distribution.png`, `temporal_generalization.png` and
`volatility_confusion.png` were produced by `src/evaluation/paper_plots.py`,
which hardcodes fabricated numbers when no results file is present. Its
placeholder block asserts a TSCT ECE of 0.08 and includes RAG, SFT+TCL+DPO and
Oracle conditions that were never run. `volatility_confusion.png` depicts a
confusion matrix that cannot be computed at all, because no inference run has
ever emitted the `predicted_volatility` field it requires.

These files are not results. They should be deleted from the repository; they
are retained here only until someone confirms nothing external links to them.
