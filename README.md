# TSCT: Temporal Self-Consistency Training

Training language models to natively distinguish time-stable from time-volatile facts and express calibrated temporal uncertainty without retrieval at inference time.

## Overview

Language models answer questions about a changing world using frozen weights. They confidently assert facts that may have changed since training (current CEOs, political leaders, prices) with no signal of temporal uncertainty. **Temporal Self-Consistency Training (TSCT)** is a fine-tuning method that teaches a model to emit one of four hedge tokens with every answer, calibrated to the fact's volatility:

| Hedge token | Confidence | Use case |
|---|---|---|
| `[CONFIDENT]` | 0.95 | Immutable facts (constants, history) |
| `[COND_CONFIDENT]` | 0.75 | Slow-changing (capitals, org structure) |
| `[TEMPORAL_HEDGE]` | 0.45 | Fast-changing (leadership, prices) |
| `[UNKNOWN]` | 0.10 | Beyond reliable knowledge horizon |

**Research question:** Can a language model be trained to reliably distinguish time-stable from time-volatile factual claims and express calibrated uncertainty about the latter, without access to external retrieval at inference time?

This repository contains the **evaluation and data-pipeline** side of the project. Training and inference are maintained separately by the training lead.

## What's new (September 2026 revision)

The evaluation was extended into a measurement-validity audit. Every result sits in `docs/orientation/results_sep10/<experiment>/` with the script that produced it, its logs and a README; `docs/orientation/RESULTS_FINAL.md` lists each final number with its source file. Pre-registered analysis plans are `docs/orientation/PLAN_*.md` and `results_sep10/item5/ENDPOINTS.md`. Per-record model outputs and the derived inputs are now committed under `data/prep/`; see **Data** below for what is not committed and how to get it.

| Experiment | Directory |
|---|---|
| ECE computed per level vs binned; Brier / Murphy decomposition | `ece_check/` |
| Where a constant policy beats the volatility-label oracle (exact crossover, 0.314) | `crossover/` |
| Confidence vocabulary: a 0.01 level and continuous confidence | `vocabulary/` |
| Repaired answer key (Table 2) and why repair raises ECE | `table2/`, `repair_mechanism/` |
| Validation-estimated constant and confidence mappings | `constant_baseline/` |
| Second aged benchmark: FreshQA, with the threshold curve | `item1/` |
| Model audit (revisions, quantization, tokenizers, known-answer checks) and the Gemma-4 exclusion | `MODELS.md`, `model_audit/`, `gemma4/` |
| Stale-value preference: estimands, prominence, time since change | `a5_prominence/` |
| Disjoint false-founder pool and the no-model form screen | `a6_disjoint/`, `screen_rule/` |
| Fixed 4-bit quantization ladder with tokenizer, prompt and length controls | `item4/` |
| Base vs instruct (Qwen2.5-7B) and a date-conditioning probe | `base_instruct/`, `date_probe/` |
| Vintage ladder (Qwen3.5 / 3.6 / 3.8, 27B) | `vintage/` |
| Inverse truth-discrimination result, capability proxy and null simulation | `inverse/`, `capability/` |
| TSCT seeds: 5 paired, compute-matched seeds on Qwen2.5-7B and on Meta-Llama-3-8B | `item5/`, `item5_llama3/` |
| Evergreen (Pletenev et al.) replication rebuilt on answers | `evergreen/` |
| Reproduction check of earlier reported numbers | `provenance/` |
| In-context evidence control, knowledge probe (partial), frequency pre-registration | `context_control/`, `knowledge_probe/`, `frequency/` |

### Data

- **TemporalDelta splits** are the Hugging Face dataset [`jasontae/temporal-delta`](https://huggingface.co/datasets/jasontae/temporal-delta) (`temporal_delta_{train,val,test}.jsonl`). The training and generation code reads a local copy from `data/prep/temporal_delta/` when present (download the files there) and falls back to `datasets.load_dataset("jasontae/temporal-delta")` otherwise.
- **Wikidata API cache** (`data/prep/wikidata_cache/`, about 200 MB of raw entity JSON) is not committed. Regenerate it with
  `python3 src/evaluation/verify_gold_currency.py --cache-dir data/prep/wikidata_cache --out /tmp/gold_currency_audit.new.json`.
  The committed `data/prep/gold_currency_audit.json` was queried on 2026-08-11; live Wikidata changes, so write a re-query to a new `--out` rather than over it.
- **FreshQA** snapshots come from [freshllms/freshqa](https://github.com/freshllms/freshqa) (Apache-2.0); the joined set used here is `data/prep/freshqa/freshqa_built.jsonl`.
- **Model checkpoints** are MLX conversions listed with revisions in `docs/orientation/results_sep10/MODELS.md`; scripts look for them under `$TSCT_MODELS_DIR` (default `~/.oMLX/models`).

## Repository structure

```
temporal-self-consistency/
├── data/
│   ├── prep/               # (gitignored) large generated datasets
│   ├── stress_tests/       # adversarial + mixed-paragraph stress sets
│   └── samples/            # small samples for inspection
├── src/
│   ├── data_pipeline/      # dataset construction + stress-test generation
│   │   ├── prep_mmlu.py            # MMLU stable subset (regression check)
│   │   ├── prep_stress_horizon.py  # 18-36 month post-cutoff facts
│   │   ├── prep_stress_stable.py   # over-hedging detector set
│   │   └── prep_stress_mixed.py    # mixed stable/volatile paragraphs
│   └── evaluation/         # the metrics pipeline
│       ├── eval_pipeline.py        # ECE, EM/F1, volatility, Bonferroni, etc.
│       ├── adapt_predictions.py    # normalize team output formats
│       ├── full_analysis.py        # run all prediction files
│       ├── generate_results_table.py  # MD/LaTeX/CSV results tables
│       ├── hedge_quality_rubric.py # human-eval rubric + auto scoring
│       └── paper_plots.py          # all 6 paper figures
├── scripts/                # convenience runners
├── figures/                # generated paper figures
├── paper/                  # paper draft
├── docs/                   # proposal, status, design notes
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Build evaluation datasets
python src/data_pipeline/prep_mmlu.py
python src/data_pipeline/prep_stress_stable.py
python src/data_pipeline/prep_stress_mixed.py
# prep_stress_horizon.py needs all_triples.jsonl in the working dir

# 2. Normalize any team prediction file to the canonical format
python src/evaluation/adapt_predictions.py raw_preds.jsonl clean_preds.jsonl

# 3. Run the full evaluation across all prediction files
TSCT_PREDICTIONS_DIR=./predictions python src/evaluation/full_analysis.py

# 4. Generate figures and the results table
python src/evaluation/paper_plots.py
python src/evaluation/generate_results_table.py
```

Predictions are produced by the training lead's checkpoints and dropped into `predictions/` (gitignored). This repo consumes those files; it does not train or run inference.

The large dataset files (`all_triples.jsonl` and the train/val/test splits) are gitignored here and hosted on HuggingFace instead: **[jasontae/temporal-delta](https://huggingface.co/datasets/jasontae/temporal-delta)**. Download them with:

```bash
pip install huggingface_hub
hf download jasontae/temporal-delta --repo-type dataset --local-dir ./data/prep
```

## Prediction format

Every prediction the eval pipeline consumes must be a JSON object with:

```json
{
  "predicted_answer": "John Donahoe",
  "gold_answer": "John Donahoe",
  "predicted_hedge": "[TEMPORAL_HEDGE]",
  "correct": true,
  "volatility": "fast",
  "change_year": 2024
}
```

`adapt_predictions.py` converts the known team output variants into this format.

## Evaluation metrics

1. **ECE** — Expected Calibration Error, equal-frequency binning
2. **Accuracy** — exact match + token-level F1
3. **Volatility breakdown** — ECE/accuracy split by fast/slow/immutable
4. **Bonferroni comparison** — TSCT vs 7 baselines, corrected alpha approximately 0.007
5. **Volatility discrimination** — confusion matrix vs ground-truth volatility
6. **Temporal generalization gap** — performance vs months past cutoff

## Datasets

| Dataset | Role |
|---|---|
| [TemporalDelta (ours)](https://huggingface.co/datasets/jasontae/temporal-delta) | Training + test — Wikidata 11-property SPARQL extraction |
| PAT-Questions | Training contrastive pairs + eval |
| FreshQA | Primary eval — fast-changing facts |
| TLQA | Eval — list-based temporal QA |
| TDBench | Eval — time-accuracy metric |
| MMLU stable subset | Regression check — accuracy must not degrade |

## Status

See `docs/STATUS.md` for the live task tracker. As of the latest update, the evaluation infrastructure is complete and verified; model checkpoints are being debugged on the training side (see `docs/tcl_debugging.md`).

## Team

Jason Tae (project lead + evaluation), Tanvi Varangaonkar (training), Logan Kim (baselines), Aarav Vignesh (data).
