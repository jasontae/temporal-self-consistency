"""
Per-Benchmark Breakdown
=======================
The main results table aggregates temporal_delta only. This breaks out every
benchmark separately so you can check the proposal's second success criterion:

    "No accuracy regression: MMLU and time-stable FreshQA accuracy within
     1-2% of base model."

MMLU is the regression check (entirely stable math/science facts). If TSCT's
MMLU accuracy drops more than ~2 points vs SFT, the no-regression criterion
fails and that has to be reported.

Also reports the hedge distribution per benchmark, which matters because MMLU
is 100% stable facts — the model should be emitting [CONFIDENT] there, not
hedging. Over-hedging on MMLU is the failure mode the adversarial-stable stress
test was designed to catch.

Usage:
    python per_benchmark_breakdown.py --pred-dir <folder> --out-dir ./results
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pipeline import compute_ece, compute_accuracy_metrics

CONDITIONS = ["sft", "tsct"]
SEEDS = [42, 123, 456]  # default; auto-detected at runtime


def detect_seeds(pred_dir):
    import re
    seeds = set()
    for p in glob.glob(os.path.join(pred_dir, "*_predictions.jsonl")):
        m = re.search(r"seed(\d+)", os.path.basename(p))
        if m:
            seeds.add(int(m.group(1)))
    return sorted(seeds) if seeds else SEEDS
BENCHMARKS = ["temporal_delta", "mmlu", "freshqa", "stress_test"]


def load(pred_dir, condition, seed, benchmark):
    exp_num = {"sft": "exp2", "tsct": "exp3"}.get(condition, "exp*")
    patterns = [
        f"{exp_num}_*_seed{seed}_{benchmark}_predictions.jsonl",
        f"exp*_{condition}_seed{seed}_{benchmark}_predictions.jsonl",
        f"{exp_num}_*_seed{seed}_{benchmark}_*_predictions.jsonl",
    ]
    for pat in patterns:
        hits = glob.glob(os.path.join(pred_dir, pat))
        if hits:
            with open(hits[0]) as f:
                return [json.loads(l) for l in f]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", default="./results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = defaultdict(dict)

    for bench in BENCHMARKS:
        for cond in CONDITIONS:
            eces, ems, f1s = [], [], []
            hedges = Counter()
            n_records = 0
            for seed in detect_seeds(args.pred_dir):
                preds = load(args.pred_dir, cond, seed, bench)
                if not preds:
                    continue
                eces.append(compute_ece(preds)["ece"])
                acc = compute_accuracy_metrics(preds)
                ems.append(acc["em"])
                f1s.append(acc["f1"])
                for p in preds:
                    hedges[p.get("predicted_hedge", "?")] += 1
                n_records = len(preds)
            if not eces:
                continue
            total_h = sum(hedges.values()) or 1
            results[bench][cond] = {
                "ece": [round(float(np.mean(eces)), 4), round(float(np.std(eces)), 4)],
                "em": [round(float(np.mean(ems)), 4), round(float(np.std(ems)), 4)],
                "f1": [round(float(np.mean(f1s)), 4), round(float(np.std(f1s)), 4)],
                "n_per_seed": n_records,
                "hedge_pct": {k: round(100 * v / total_h, 1)
                              for k, v in hedges.most_common()},
            }

    # ---- print tables ----
    print("\n" + "=" * 78)
    print("PER-BENCHMARK RESULTS")
    print("=" * 78)
    print(f"{'Benchmark':<16}{'Cond':<7}{'ECE':>16}{'EM':>16}{'F1':>16}")
    print("-" * 78)
    for bench in BENCHMARKS:
        if bench not in results:
            continue
        for cond in CONDITIONS:
            if cond not in results[bench]:
                continue
            r = results[bench][cond]
            print(f"{bench:<16}{cond.upper():<7}"
                  f"{r['ece'][0]:>9.4f}±{r['ece'][1]:<6.3f}"
                  f"{r['em'][0]:>9.4f}±{r['em'][1]:<6.3f}"
                  f"{r['f1'][0]:>9.4f}±{r['f1'][1]:<6.3f}")
        print()

    # ---- regression check ----
    print("=" * 78)
    print("ACCURACY REGRESSION CHECK (proposal: within 1-2% on stable facts)")
    print("=" * 78)
    for bench in ("mmlu", "freshqa"):
        if bench not in results or "sft" not in results[bench] or "tsct" not in results[bench]:
            continue
        sft_em = results[bench]["sft"]["em"][0]
        tsct_em = results[bench]["tsct"]["em"][0]
        delta_pp = (tsct_em - sft_em) * 100
        rel = (100 * (tsct_em - sft_em) / sft_em) if sft_em else 0
        verdict = "PASS" if abs(delta_pp) <= 2.0 else "FAIL"
        print(f"  {bench.upper():<12} SFT EM={sft_em:.4f}  TSCT EM={tsct_em:.4f}  "
              f"delta={delta_pp:+.2f}pp ({rel:+.1f}% rel)  -> {verdict}")
    print("\n  Criterion is on absolute percentage points (1-2pp), per the proposal.")

    # ---- over-hedging check on stable benchmarks ----
    print("\n" + "=" * 78)
    print("OVER-HEDGING CHECK (MMLU is 100% stable -> should be [CONFIDENT])")
    print("=" * 78)
    for bench in ("mmlu",):
        if bench not in results:
            continue
        for cond in CONDITIONS:
            if cond not in results[bench]:
                continue
            h = results[bench][cond]["hedge_pct"]
            conf = h.get("[CONFIDENT]", 0.0)
            hedged = 100.0 - conf
            print(f"  {cond.upper():<6} [CONFIDENT]={conf:.1f}%   hedged={hedged:.1f}%")
            print(f"         full: {h}")
    print("\n  High hedged% on MMLU = over-hedging on stable facts.")

    with open(os.path.join(args.out_dir, "per_benchmark_breakdown.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {os.path.join(args.out_dir, 'per_benchmark_breakdown.json')}")


if __name__ == "__main__":
    main()
