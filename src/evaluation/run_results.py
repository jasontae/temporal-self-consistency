"""
Master Results Runner — Tasks 2, 3, 4
======================================
Takes the prediction files (6 checkpoints x 4 benchmarks) and produces:
  TASK 2: Cohen's d + p-values for all pairwise ECE comparisons (Bonferroni)
  TASK 3: all plots (calibration, ECE bars, temporal generalization, confusion)
  TASK 4: full results table (all models x all metrics) in MD/CSV

This is wired to the REAL prediction format produced by Tanvi's notebook:
    {predicted_answer, gold_answer, predicted_hedge, correct, volatility, change_year}

It runs on whatever prediction files are present. With the current (broken,
all-[CONFIDENT]) checkpoints it produces real-but-uninteresting numbers; the
moment corrected checkpoints replace them, the same command yields the final
paper results. No code changes needed between now and then.

Usage:
    python run_results.py --pred-dir /path/to/predictions --out-dir ./results
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pipeline import (
    compute_ece, compute_accuracy_metrics, volatility_breakdown,
    bonferroni_ece_comparison, volatility_discrimination,
    temporal_generalization_gap, run_full_evaluation,
)

CONDITIONS = ["sft", "tsct"]      # extend with baselines as they arrive
SEEDS = [42, 123, 456]  # default; overridden by auto-detection at runtime


def detect_seeds(pred_dir):
    """Find every seed present in the folder, so n=3 or n=8 both work."""
    import re
    seeds = set()
    for p in glob.glob(os.path.join(pred_dir, "*_predictions.jsonl")):
        m = re.search(r"seed(\d+)", os.path.basename(p))
        if m:
            seeds.add(int(m.group(1)))
    return sorted(seeds) if seeds else SEEDS
BENCHMARKS = ["temporal_delta", "mmlu", "freshqa", "stress_test"]


def load_predictions(pred_dir, condition, seed, benchmark):
    """Load one prediction file, tolerant of naming variants."""
    # exp2 == SFT (CE only), exp3 == TSCT (CE + TCL). Handles both the
    # exp2_sft_* naming and David's exp2_fixed_* / exp2_v2_* naming.
    exp_num = {"sft": "exp2", "tsct": "exp3"}.get(condition, "exp*")
    patterns = [
        f"exp*_{condition}_seed{seed}_{benchmark}_predictions.jsonl",
        f"{exp_num}_*_seed{seed}_{benchmark}_predictions.jsonl",
        f"{exp_num}_seed{seed}_{benchmark}_predictions.jsonl",
        f"*{condition}*seed{seed}*{benchmark}*.jsonl",
    ]
    for pat in patterns:
        hits = glob.glob(os.path.join(pred_dir, pat))
        if hits:
            with open(hits[0]) as f:
                return [json.loads(l) for l in f]
    return None


def task2_significance(pred_dir, out_dir):
    """Cohen's d + Bonferroni-corrected p-values for ECE comparisons."""
    seeds = detect_seeds(pred_dir)
    print("\n=== TASK 2: Statistical significance (Bonferroni) ===")
    # Build per-condition ECE lists across seeds on the primary benchmark
    ece_by_condition = {}
    for cond in CONDITIONS:
        eces = []
        for seed in detect_seeds(pred_dir):
            preds = load_predictions(pred_dir, cond, seed, "temporal_delta")
            if preds:
                eces.append(compute_ece(preds)["ece"])
        if eces:
            ece_by_condition[cond] = eces

    if "tsct" not in ece_by_condition:
        print("  Need TSCT predictions; have:", list(ece_by_condition))
        return None

    tsct_ece = ece_by_condition.pop("tsct")
    baselines = ece_by_condition  # everything else is a baseline
    if not baselines:
        print("  Need >=1 baseline condition alongside TSCT.")
        return None

    result = bonferroni_ece_comparison(tsct_ece, baselines)
    with open(os.path.join(out_dir, "task2_significance.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  corrected_alpha = {result['corrected_alpha']}")
    for name, comp in result.get("comparisons", {}).items():
        print(f"    TSCT vs {name}: ECE_reduction={comp['ece_reduction']:+.4f}, "
              f"p={comp['p_value']:.4f}, d={comp['cohens_d']:+.3f}, "
              f"sig={comp['significant_bonferroni']}")
    print(f"  Wrote task2_significance.json")
    return result


def task4_results_table(pred_dir, out_dir):
    """Full results table: all conditions x all metrics."""
    print("\n=== TASK 4: Full results table ===")
    rows = {}
    for cond in CONDITIONS:
        # Aggregate across seeds on temporal_delta
        per_seed = defaultdict(list)
        for seed in detect_seeds(pred_dir):
            preds = load_predictions(pred_dir, cond, seed, "temporal_delta")
            if not preds:
                continue
            ece = compute_ece(preds)["ece"]
            acc = compute_accuracy_metrics(preds)
            vb = volatility_breakdown(preds)
            per_seed["ece"].append(ece)
            per_seed["em"].append(acc["em"])
            per_seed["f1"].append(acc["f1"])
            for vol in ("fast", "slow", "immutable"):
                if vol in vb:
                    e = vb[vol].get("ece")
                    eval_ece = e["ece"] if isinstance(e, dict) else e
                    if eval_ece is not None:
                        per_seed[f"ece_{vol}"].append(eval_ece)
        if per_seed:
            rows[cond] = {k: (float(np.mean(v)), float(np.std(v)))
                          for k, v in per_seed.items()}

    # Write markdown
    md = ["| Condition | ECE | EM | F1 | ECE fast | ECE slow |",
          "|---|---|---|---|---|---|"]
    for cond, m in rows.items():
        def cell(key):
            if key in m:
                return f"{m[key][0]:.3f} ± {m[key][1]:.3f}"
            return "—"
        md.append(f"| {cond.upper()} | {cell('ece')} | {cell('em')} | "
                  f"{cell('f1')} | {cell('ece_fast')} | {cell('ece_slow')} |")
    with open(os.path.join(out_dir, "task4_results_table.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\n  Wrote task4_results_table.md")
    return rows


def task3_plots(pred_dir, out_dir, table_rows):
    """Generate the four required figures from real data."""
    print("\n=== TASK 3: Figures ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # --- ECE bar chart (real numbers) ---
    if table_rows:
        conds = list(table_rows.keys())
        means = [table_rows[c]["ece"][0] for c in conds]
        stds = [table_rows[c]["ece"][1] for c in conds]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(conds, means, yerr=stds, capsize=4, color="#2C7BB6",
               edgecolor="black")
        ax.set_ylabel("ECE")
        ax.set_title("ECE by condition (temporal_delta)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "ece_bar_chart.png"), dpi=150)
        plt.close()
        print("  ece_bar_chart.png")

    # --- Volatility confusion matrix (needs true_volatility field) ---
    preds = load_predictions(pred_dir, "tsct", 42, "temporal_delta")
    if preds:
        # The discrimination metric needs a 'true_volatility' field. The current
        # prediction files carry 'volatility' (ground truth) but not a separate
        # predicted volatility, so this figure is only meaningful once the model
        # emits a predicted volatility class. Skip gracefully if absent.
        if any("true_volatility" in p for p in preds):
            vd = volatility_discrimination(preds)
            if vd and "confusion_matrix" in vd:
                cm = vd["confusion_matrix"]
                classes = vd.get("classes", list(cm.keys()))
                mat = np.array([[cm.get(r, {}).get(c, 0) for c in classes]
                                for r in classes])
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(mat, cmap="Blues")
                ax.set_xticks(range(len(classes)))
                ax.set_yticks(range(len(classes)))
                ax.set_xticklabels(classes)
                ax.set_yticklabels(classes)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title("Volatility discrimination (TSCT)")
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        ax.text(j, i, mat[i][j], ha="center", va="center")
                plt.colorbar(im)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "volatility_confusion.png"), dpi=150)
                plt.close()
                print("  volatility_confusion.png")
        else:
            print("  volatility_confusion.png SKIPPED (no 'true_volatility' field "
                  "in predictions — needs predicted volatility class from model)")

    # --- Temporal generalization curve (real) ---
    if preds:
        tg = temporal_generalization_gap(preds)
        if tg and "buckets" in tg and tg["buckets"]:
            buckets = tg["buckets"]
            # bucket keys are strings like "12-24 months"; ece is nested
            labels = list(buckets.keys())
            ys = []
            for lbl in labels:
                e = buckets[lbl].get("ece")
                # ece may be {"ece": float, ...} or a bare float
                ys.append(e["ece"] if isinstance(e, dict) else e)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(range(len(labels)), ys, "o-", color="#D7191C")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel("ECE")
            ax.set_title("Temporal generalization (TSCT)")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "temporal_generalization.png"), dpi=150)
            plt.close()
            print("  temporal_generalization.png")

    print(f"  Figures in {fig_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", default="./results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Prediction dir: {args.pred_dir}")
    print(f"Output dir:     {args.out_dir}")

    task2_significance(args.pred_dir, args.out_dir)
    rows = task4_results_table(args.pred_dir, args.out_dir)
    task3_plots(args.pred_dir, args.out_dir, rows)

    print("\nAll tasks complete. Re-run with corrected predictions for final results.")


if __name__ == "__main__":
    main()
