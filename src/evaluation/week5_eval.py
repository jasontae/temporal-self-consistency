"""
Week 5 Evaluation Runner — Three Buildable Tasks
================================================
1. Temporal generalization gap (12-month splits)   — WORKS on current data
2. Hedge taxonomy appropriateness scoring           — WORKS on current data
   (automated rubric, separate from ECE and from human eval)
3. Volatility discrimination confusion matrix       — SCAFFOLD; needs a
   predicted-volatility field to produce real output

Run:
    python week5_eval.py --pred-dir <folder> --out-dir ./week5_results
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pipeline import temporal_generalization_gap, volatility_discrimination
from hedge_quality_rubric import automated_hedge_score, GOLD_HEDGE_FOR_VOLATILITY


def load(pred_dir, condition, seed, benchmark):
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


# ---------------------------------------------------------------------------
# TASK: Temporal generalization gap
# ---------------------------------------------------------------------------
def task_temporal_gap(pred_dir, out_dir):
    print("\n=== Temporal generalization gap (12-month splits) ===")
    out = {}
    for cond in ("sft", "tsct"):
        preds = load(pred_dir, cond, 42, "temporal_delta")
        if not preds:
            continue
        tg = temporal_generalization_gap(preds, cutoff_year=2022, split_months=12)
        # flatten the nested ece for readability
        buckets = {}
        for label, info in tg.get("buckets", {}).items():
            e = info.get("ece")
            buckets[label] = {
                "count": info.get("count"),
                "ece": e["ece"] if isinstance(e, dict) else e,
            }
        out[cond] = {"buckets": buckets,
                     "generalization_gap": tg.get("generalization_gap")}
        print(f"  {cond.upper()}:")
        for label, info in buckets.items():
            print(f"    {label}: ECE={info['ece']:.4f} (n={info['count']})")
        print(f"    gap (closest->farthest): {out[cond]['generalization_gap']}")

    with open(os.path.join(out_dir, "temporal_generalization_gap.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("  -> temporal_generalization_gap.json")
    return out


# ---------------------------------------------------------------------------
# TASK: Hedge taxonomy appropriateness scoring (automated rubric)
# ---------------------------------------------------------------------------
def task_hedge_scoring(pred_dir, out_dir):
    print("\n=== Hedge taxonomy appropriateness (automated rubric, 1-5) ===")
    out = {}
    for cond in ("sft", "tsct"):
        # aggregate across seeds on temporal_delta
        all_scores = []
        per_class_scores = defaultdict(list)
        hedge_dist = Counter()
        for seed in (42, 123, 456):
            preds = load(pred_dir, cond, seed, "temporal_delta")
            if not preds:
                continue
            for p in preds:
                vol = p.get("volatility")
                hedge = p.get("predicted_hedge")
                if vol is None or hedge is None:
                    continue
                score = automated_hedge_score(hedge, vol)
                all_scores.append(score)
                per_class_scores[vol].append(score)
                hedge_dist[hedge] += 1

        if not all_scores:
            continue
        out[cond] = {
            "mean_appropriateness": round(sum(all_scores) / len(all_scores), 3),
            "n_scored": len(all_scores),
            "per_volatility_mean": {
                v: round(sum(s) / len(s), 3) for v, s in per_class_scores.items()
            },
            "hedge_distribution": dict(hedge_dist),
            "score_histogram": dict(Counter(all_scores)),
        }
        print(f"  {cond.upper()}: mean={out[cond]['mean_appropriateness']}/5 "
              f"(n={out[cond]['n_scored']})")
        print(f"    per-volatility: {out[cond]['per_volatility_mean']}")
        print(f"    hedge dist: {out[cond]['hedge_distribution']}")

    with open(os.path.join(out_dir, "hedge_appropriateness.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("  -> hedge_appropriateness.json")
    return out


# ---------------------------------------------------------------------------
# TASK: Volatility discrimination confusion matrix (scaffold)
# ---------------------------------------------------------------------------
def task_volatility_discrimination(pred_dir, out_dir):
    print("\n=== Volatility discrimination confusion matrix ===")
    preds = load(pred_dir, "tsct", 42, "temporal_delta")
    if not preds:
        print("  no predictions found")
        return None

    has_pred_vol = any("predicted_volatility" in p for p in preds)
    if not has_pred_vol:
        print("  SCAFFOLD ONLY: predictions have ground-truth 'volatility' but no")
        print("  'predicted_volatility' field. The model must emit a predicted")
        print("  volatility class for a real confusion matrix. Writing a stub that")
        print("  activates automatically once that field is present.")
        # Show what the ground-truth distribution looks like, for reference
        gt = Counter(p.get("volatility") for p in preds)
        stub = {
            "status": "awaiting_predicted_volatility_field",
            "ground_truth_distribution": dict(gt),
            "note": ("Add 'predicted_volatility' to each prediction record; this "
                     "task then produces a real confusion matrix with no code change."),
        }
        with open(os.path.join(out_dir, "volatility_discrimination.json"), "w") as f:
            json.dump(stub, f, indent=2, default=str)
        print(f"    ground-truth volatility distribution: {dict(gt)}")
        print("  -> volatility_discrimination.json (stub)")
        return stub

    # Real path (runs when predicted_volatility exists)
    vd = volatility_discrimination(preds)
    with open(os.path.join(out_dir, "volatility_discrimination.json"), "w") as f:
        json.dump(vd, f, indent=2, default=str)
    print("  -> volatility_discrimination.json (real)")
    return vd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", default="./week5_results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    task_temporal_gap(args.pred_dir, args.out_dir)
    task_hedge_scoring(args.pred_dir, args.out_dir)
    task_volatility_discrimination(args.pred_dir, args.out_dir)

    print("\nDone. Two tasks produce real output now; volatility discrimination")
    print("activates once predictions include a predicted volatility class.")


if __name__ == "__main__":
    main()
