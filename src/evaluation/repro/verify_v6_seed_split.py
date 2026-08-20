"""
Reproduce the two findings in section 0 of TCL_V6_DIAGNOSIS_AND_PROPOSAL.md
==========================================================================

  0.1  The v6 numbers reported in docs/RESULTS_LOG.md as "n=8" reproduce
       exactly on seeds 42/123/456 only. Pooled over all 8 seeds the
       headline ECE reduction disappears (and reverses on temporal_delta).

  0.2  Seeds 501-505 show a complete inversion of the SFT/TSCT roles on
       all four benchmarks.

This script imports the project's own eval_pipeline.py and changes nothing
about how ECE / EM are computed. It only changes which seeds get averaged.

No GPU. CPU only, ~20 seconds. Needs: numpy, scipy.

Usage
-----
    python3 verify_v6_seed_split.py --pred-dir /path/to/predictions_v6

Get predictions_v6 from HuggingFace if you don't have it locally:

    hf download DavidS64/llama3-hedge-sft --include "predictions_v6/*" \
        --repo-type model --local-dir ~/Downloads/david_v6

Expected: 64 files (8 seeds x 4 benchmarks x 2 conditions).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

# Import the project's real metric code from ../eval_toolkit
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from eval_pipeline import (  # noqa: E402
    compute_ece, compute_accuracy_metrics, bonferroni_ece_comparison,
)

OLD_SEEDS = [42, 123, 456]           # the three original seeds
NEW_SEEDS = [501, 502, 503, 504, 505]  # the five David added for v6
ALL_SEEDS = OLD_SEEDS + NEW_SEEDS
BENCHMARKS = ["temporal_delta", "mmlu", "freshqa", "stress_test"]

# What docs/RESULTS_LOG.md "Run 3 - predictions_v6 (n=8)" reports.
DOC_ECE = {
    "temporal_delta": (0.854, 0.419),
    "mmlu":           (0.754, 0.375),
    "freshqa":        (0.756, 0.354),
    "stress_test":    (0.847, 0.416),
}
DOC_EM = {"mmlu": (0.067, 0.030), "freshqa": (0.097, 0.049)}
DOC_SIG = {"reduction": 0.435, "p": 0.0099, "d": 3.767}


def load(pred_dir, cond, seed, bench):
    """Same exp2=SFT / exp3=TSCT convention the eval_toolkit scripts use."""
    exp = "exp2" if cond == "sft" else "exp3"
    hits = glob.glob(os.path.join(
        pred_dir, f"{exp}_*_seed{seed}_{bench}_predictions.jsonl"))
    if not hits:
        return None
    with open(hits[0]) as f:
        return [json.loads(line) for line in f]


def mean_over(pred_dir, seeds, bench, cond, fn):
    vals = []
    for s in seeds:
        recs = load(pred_dir, cond, s, bench)
        if recs:
            vals.append(fn(recs))
    return (float(np.mean(vals)) if vals else float("nan")), vals


ece_of = lambda r: compute_ece(r)["ece"]           # noqa: E731
em_of  = lambda r: compute_accuracy_metrics(r)["em"]  # noqa: E731


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True,
                    help="folder containing the 64 predictions_v6 jsonl files")
    args = ap.parse_args()

    n_files = len(glob.glob(os.path.join(args.pred_dir, "*_predictions.jsonl")))
    print(f"Found {n_files} prediction files in {args.pred_dir}")
    if n_files != 64:
        print("  WARNING: expected 64 (8 seeds x 4 benchmarks x 2 conditions).")
    print()

    # ---------------------------------------------------------------- 0.1
    print("=" * 94)
    print("0.1  Documented 'n=8' numbers vs seeds 42/123/456 vs all 8 seeds")
    print("=" * 94)
    print(f"{'benchmark':<16}{'doc SFT':>9}{'doc TSCT':>10}"
          f"{'n3 SFT':>9}{'n3 TSCT':>9}{'n8 SFT':>9}{'n8 TSCT':>9}")
    for b in BENCHMARKS:
        d_sft, d_tsct = DOC_ECE[b]
        n3_s, _ = mean_over(args.pred_dir, OLD_SEEDS, b, "sft", ece_of)
        n3_t, _ = mean_over(args.pred_dir, OLD_SEEDS, b, "tsct", ece_of)
        n8_s, _ = mean_over(args.pred_dir, ALL_SEEDS, b, "sft", ece_of)
        n8_t, _ = mean_over(args.pred_dir, ALL_SEEDS, b, "tsct", ece_of)
        print(f"{b:<16}{d_sft:>9.3f}{d_tsct:>10.3f}"
              f"{n3_s:>9.3f}{n3_t:>9.3f}{n8_s:>9.3f}{n8_t:>9.3f}")

    print("\nAccuracy (EM):")
    for b in ("mmlu", "freshqa"):
        d_sft, d_tsct = DOC_EM[b]
        n3_s, _ = mean_over(args.pred_dir, OLD_SEEDS, b, "sft", em_of)
        n3_t, _ = mean_over(args.pred_dir, OLD_SEEDS, b, "tsct", em_of)
        n8_s, _ = mean_over(args.pred_dir, ALL_SEEDS, b, "sft", em_of)
        n8_t, _ = mean_over(args.pred_dir, ALL_SEEDS, b, "tsct", em_of)
        print(f"  {b:<10} doc {d_sft:.3f}->{d_tsct:.3f}   "
              f"n=3 {n3_s:.4f}->{n3_t:.4f}   n=8 {n8_s:.4f}->{n8_t:.4f}")

    # ------------------------------------------------------- significance
    print("\n" + "=" * 94)
    print(f"Task 2 significance on temporal_delta "
          f"(docs report reduction={DOC_SIG['reduction']:+.3f}, "
          f"p={DOC_SIG['p']}, d={DOC_SIG['d']})")
    print("=" * 94)
    for label, seeds in (("seeds 42/123/456 (n=3)", OLD_SEEDS),
                         ("all 8 seeds     (n=8)", ALL_SEEDS),
                         ("seeds 501-505   (n=5)", NEW_SEEDS)):
        _, tsct = mean_over(args.pred_dir, seeds, "temporal_delta", "tsct", ece_of)
        _, sft  = mean_over(args.pred_dir, seeds, "temporal_delta", "sft", ece_of)
        if len(tsct) < 2 or len(sft) < 2:
            continue
        comp = bonferroni_ece_comparison(tsct, {"SFT": sft})["comparisons"]["SFT"]
        print(f"  {label:<24} reduction={comp['ece_reduction']:+.4f}  "
              f"p={comp['p_value']:.4f}  d={comp['cohens_d']:+.3f}  "
              f"sig_bonferroni={comp['significant_bonferroni']}")

    # ---------------------------------------------------------------- 0.2
    print("\n" + "=" * 94)
    print("0.2  Seed-group inversion: SFT/TSCT roles swap between the two batches")
    print("=" * 94)
    print(f"{'benchmark':<16}{'seed group':<13}{'SFT ECE':>9}{'TSCT ECE':>10}"
          f"{'SFT EM':>9}{'TSCT EM':>9}   direction")
    for b in BENCHMARKS:
        for label, seeds in (("42/123/456", OLD_SEEDS), ("501-505", NEW_SEEDS)):
            se, _ = mean_over(args.pred_dir, seeds, b, "sft", ece_of)
            te, _ = mean_over(args.pred_dir, seeds, b, "tsct", ece_of)
            sm, _ = mean_over(args.pred_dir, seeds, b, "sft", em_of)
            tm, _ = mean_over(args.pred_dir, seeds, b, "tsct", em_of)
            print(f"{b:<16}{label:<13}{se:>9.3f}{te:>10.3f}{sm:>9.4f}{tm:>9.4f}"
                  f"   ECE {'TSCT better' if te < se else 'TSCT WORSE '}"
                  f", EM {'TSCT better' if tm > sm else 'TSCT WORSE'}")
        print()

    print("If the two seed groups disagree in direction on all four benchmarks,")
    print("they are not exchangeable and must not be averaged as 'n=8'.")


if __name__ == "__main__":
    main()
