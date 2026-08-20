"""
Backfill change_year into prediction files
==========================================
The temporal generalization gap needs a `change_year` field on each prediction
(the year the fact changed, from the test set's t_end). David's inference script
doesn't emit it, so the metric returns empty buckets.

This joins change_year back in from the source test set by matching on the
question text, then rewrites the prediction files. No retrain needed.

Usage:
    python backfill_change_year.py \
        --pred-dir ~/Downloads/david_preds/predictions \
        --test-set ~/path/to/temporal_delta_test_cleaned.jsonl \
        --out-dir ~/Downloads/david_preds/predictions_with_year
"""
import argparse
import glob
import json
import os


def build_lookup(test_path):
    """Map question -> change_year from the source test set."""
    lookup = {}
    with open(test_path) as f:
        for line in f:
            r = json.loads(line)
            q = r.get("question")
            t_end = r.get("t_end") or r.get("validity_end")
            if q and t_end:
                try:
                    lookup[q.strip()] = int(t_end)
                except (ValueError, TypeError):
                    pass
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--test-set", required=True,
                    help="temporal_delta_test_cleaned.jsonl")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lookup = build_lookup(args.test_set)
    print(f"Loaded {len(lookup)} question->change_year mappings from test set\n")

    # Only temporal_delta predictions have a meaningful change_year
    files = sorted(glob.glob(os.path.join(args.pred_dir, "*temporal_delta*.jsonl")))
    if not files:
        print(f"No temporal_delta prediction files in {args.pred_dir}")
        return

    for path in files:
        with open(path) as f:
            rows = [json.loads(l) for l in f]

        matched = 0
        for r in rows:
            if "change_year" in r and r["change_year"] is not None:
                matched += 1
                continue
            q = (r.get("question") or "").strip()
            if q and q in lookup:
                r["change_year"] = lookup[q]
                matched += 1

        out_path = os.path.join(args.out_dir, os.path.basename(path))
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        name = os.path.basename(path)
        pct = 100 * matched / len(rows) if rows else 0
        print(f"{name:<55} {matched}/{len(rows)} matched ({pct:.1f}%)")

    # Copy the non-temporal_delta files through unchanged so the out-dir is complete
    others = [p for p in glob.glob(os.path.join(args.pred_dir, "*_predictions.jsonl"))
              if "temporal_delta" not in p]
    for path in others:
        dest = os.path.join(args.out_dir, os.path.basename(path))
        if not os.path.exists(dest):
            with open(path) as src, open(dest, "w") as dst:
                dst.write(src.read())

    print(f"\nWrote to {args.out_dir}")
    print("If match rate is low, the prediction files may not carry a 'question'")
    print("field — in that case David needs to add change_year at inference time.")


if __name__ == "__main__":
    main()
