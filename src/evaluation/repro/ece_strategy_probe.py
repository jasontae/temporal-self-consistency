"""
Reproduce section 1.5 of TCL_V6_DIAGNOSIS_AND_PROPOSAL.md
========================================================

Question: under eval_pipeline.compute_ece, which hedge token minimises ECE
at a given true accuracy?

eval_pipeline maps each hedge token to a FIXED confidence scalar
(0.95 / 0.75 / 0.45 / 0.10), so ECE is a pure function of the hedge
distribution and the correctness rate. That means the metric has a
well-defined optimal strategy for any accuracy level -- and at the accuracy
levels actually measured on MMLU (EM 0.03-0.07), that strategy is to emit
[UNKNOWN] on everything.

This is why "restore [CONFIDENT] on stable facts" and "keep ECE low" pull in
opposite directions until answer accuracy improves. It is a property of the
metric, not a bug in the model.

Uses the project's own compute_ece. No GPU, no prediction files needed.

Usage
-----
    python3 ece_strategy_probe.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from eval_pipeline import compute_ece, HEDGE_TO_CONFIDENCE  # noqa: E402

HEDGES = ["[CONFIDENT]", "[COND_CONFIDENT]", "[TEMPORAL_HEDGE]", "[UNKNOWN]"]


def synth(dist, accuracy, n=20000, seed=0):
    """Synthetic predictions with a given hedge distribution and accuracy.

    Correctness is drawn independently of the hedge token, which is the
    neutral assumption: it isolates the metric's own preference.
    """
    rng = np.random.default_rng(seed)
    hedges = rng.choice(HEDGES, size=n, p=dist)
    correct = rng.random(n) < accuracy
    return [{"predicted_hedge": h, "correct": bool(c)}
            for h, c in zip(hedges, correct)]


def main():
    print("Hedge -> confidence scalars used by eval_pipeline.compute_ece:")
    for h in HEDGES:
        print(f"    {h:<20} {HEDGE_TO_CONFIDENCE[h]}")

    print("\n" + "=" * 88)
    print("ECE of each pure strategy, as a function of true accuracy")
    print("=" * 88)
    print(f"{'accuracy':>9}" + "".join(f"{h:>19}" for h in HEDGES)
          + "   ECE-optimal")
    print("-" * 88)

    for acc in [0.03, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85, 0.90, 0.95]:
        row = []
        for i in range(len(HEDGES)):
            dist = [0.0] * len(HEDGES)
            dist[i] = 1.0
            row.append(compute_ece(synth(dist, acc))["ece"])
        best = HEDGES[int(np.argmin(row))]
        print(f"{acc:>9.2f}" + "".join(f"{v:>19.4f}" for v in row)
              + f"   {best}")

    print("\n" + "=" * 88)
    print("The v6 MMLU TSCT hedge distribution (seeds 42/123/456), for reference")
    print("=" * 88)
    # 10.9% CONFIDENT, 61.6% TEMPORAL_HEDGE, 22.7% UNKNOWN -> remainder COND
    v6_mmlu = [0.109, 0.048, 0.616, 0.227]
    for label, acc in (("TSCT MMLU EM = 0.030", 0.030),
                       ("SFT  MMLU EM = 0.067", 0.067)):
        e_actual = compute_ece(synth(v6_mmlu, acc))["ece"]
        e_allconf = compute_ece(synth([1, 0, 0, 0], acc))["ece"]
        e_allunk = compute_ece(synth([0, 0, 0, 1], acc))["ece"]
        print(f"  at {label}:")
        print(f"      v6 actual distribution   ECE = {e_actual:.4f}")
        print(f"      all [CONFIDENT]          ECE = {e_allconf:.4f}")
        print(f"      all [UNKNOWN]            ECE = {e_allunk:.4f}")

    print("\n" + "=" * 88)
    print("Analytic check (no sampling)")
    print("=" * 88)
    print("  For a pure strategy with confidence c at true accuracy a, ECE = |a - c|.")
    print("  So the optimal token switches at the midpoints between the scalars:")
    print("      a < 0.275  -> [UNKNOWN]          (midpoint of 0.10 and 0.45)")
    print("      a < 0.60   -> [TEMPORAL_HEDGE]   (midpoint of 0.45 and 0.75)")
    print("      a < 0.85   -> [COND_CONFIDENT]   (midpoint of 0.75 and 0.95)")
    print("      a > 0.85   -> [CONFIDENT]")
    print()
    print("Takeaway: [CONFIDENT] only becomes ECE-optimal above ~85% accuracy.")
    print("Measured MMLU accuracy is 0.03-0.07, an order of magnitude below that.")
    print("Any loss term that restores confidence on stable facts will raise ECE")
    print("unless answer accuracy rises first -- see section 3 of the proposal.")


if __name__ == "__main__":
    main()
