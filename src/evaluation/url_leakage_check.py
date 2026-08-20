"""
URL Leakage Diagnostic
======================
David flagged that the TemporalDelta extraction leaked raw Wikidata URLs into
answers, tanking exact-match accuracy. This quantifies the damage and tells you
whether EM/F1 are trustworthy.

Checks BOTH sides:
  - gold_answer containing a URL  -> the eval data is corrupted (scoring bug,
    fixable by post-processing, no retrain needed)
  - predicted_answer containing a URL -> the MODEL learned to emit URLs
    (training data corrupted; needs a retrain to fix properly)

Also reports what EM/F1 would look like if URL-contaminated rows were excluded,
so you can see how much of the accuracy hit is the bug vs. genuine model error.

Usage:
    python url_leakage_check.py --pred-dir <folder>
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

URL_RE = re.compile(r"(https?://|wikidata\.org|/entity/Q\d+|^Q\d+$)", re.IGNORECASE)


def has_url(text):
    if not isinstance(text, str):
        return False
    return bool(URL_RE.search(text))


def check_file(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f]

    n = len(rows)
    gold_url = sum(1 for r in rows if has_url(r.get("gold_answer", "")))
    pred_url = sum(1 for r in rows if has_url(r.get("predicted_answer", "")))
    either = sum(1 for r in rows
                 if has_url(r.get("gold_answer", "")) or has_url(r.get("predicted_answer", "")))

    clean = [r for r in rows
             if not has_url(r.get("gold_answer", ""))
             and not has_url(r.get("predicted_answer", ""))]

    em_all = sum(1 for r in rows if r.get("correct")) / n if n else 0
    em_clean = (sum(1 for r in clean if r.get("correct")) / len(clean)) if clean else 0

    return {
        "n": n,
        "gold_with_url": gold_url,
        "gold_pct": round(100 * gold_url / n, 1) if n else 0,
        "pred_with_url": pred_url,
        "pred_pct": round(100 * pred_url / n, 1) if n else 0,
        "either_pct": round(100 * either / n, 1) if n else 0,
        "n_clean": len(clean),
        "em_all": round(em_all, 4),
        "em_clean_only": round(em_clean, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.pred_dir, "*_predictions.jsonl")))
    if not files:
        print(f"No prediction files found in {args.pred_dir}")
        return

    print(f"Checking {len(files)} prediction files for Wikidata URL leakage\n")
    print(f"{'file':<52} {'gold%':>7} {'pred%':>7} {'EM all':>8} {'EM clean':>9}")
    print("-" * 88)

    worst_pred = 0
    for path in files:
        r = check_file(path)
        name = os.path.basename(path).replace("_predictions.jsonl", "")
        worst_pred = max(worst_pred, r["pred_pct"])
        print(f"{name:<52} {r['gold_pct']:>6.1f}% {r['pred_pct']:>6.1f}% "
              f"{r['em_all']:>8.4f} {r['em_clean_only']:>9.4f}")

    print("\nInterpretation:")
    print("  gold%  > 0  -> eval data corrupted; fixable by post-processing (no retrain)")
    print("  pred%  > 0  -> MODEL emits URLs; training data corrupted (needs retrain)")
    print("  EM clean vs EM all -> how much of the accuracy hit is the bug")
    if worst_pred > 5:
        print(f"\n  WARNING: up to {worst_pred}% of predictions contain URLs. The model")
        print("  learned this from corrupted training answers. EM/F1 are NOT trustworthy")
        print("  until the extraction is fixed and the models retrained.")
    else:
        print("\n  Predictions look clean of URLs; leakage is likely gold-side only,")
        print("  which post-processing can fix without retraining.")


if __name__ == "__main__":
    main()
