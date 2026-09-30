"""Read the EverGreenQA uncertainty runs three ways, and adjudicate the trend.

Pletenev et al. (EMNLP 2025) Table 3 reports |Pearson r| of 0.17-0.35 between
evergreen-ness and two uncertainty measures, with the note that "larger models
correlate more strongly with evergreen-ness, possibly indicating a greater
internal reliance on temporal cues."

Our matched within-entity result (B11-Y) points the other way: the regime effect
is significant in all four weakest models and absent in all four strongest. The
two are measured differently, so this puts them on one axis.

Three readings per model:

  pooled r       their statistic, so the numbers are comparable to the paper
  pooled AUROC   the same signal on the ledger's scale
  within-form    the stricter control

The third is the point. Our screen (`evergreen_surface_screen`) showed a
past-tense marker alone separates their classes at AUROC 0.6640 with no model
consulted. An uncertainty signal that merely tracks that marker is not evidence
of an internal temporal representation, so we stratify on it and ask what the
signal adds once question form is held fixed. This is the same move that took
B5 from 0.6700 to 0.4997 and that B11 failed to survive.

Sign convention: scores are oriented so that HIGHER predicts EVERGREEN. Since a
stable fact should be answered more confidently, that means negated perplexity
and negated entropy.

Usage:
    python3 -m src.evaluation.analyze_evergreen_uncertainty
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from .b11_surface_control import auroc
from .evergreen_surface_screen import PAST

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "data" / "prep" / "predictions_7b"

# rough parameter counts, for the size-trend question
SIZE = {
    "qwen3_4b_it": 4, "qwen25_7b": 7, "gptoss_20b": 20,
    "qwen35_27b": 27, "qwen36_27b": 27, "qwen38_27b": 27, "qwen36_35b": 35,
}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx_, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx_) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx_) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def within_form(rows, score):
    """AUROC averaged over question-form strata, weighted by stratum size.

    Stratum = whether the question carries a past-tense marker, the single
    surface feature that most separates their classes with no model involved.
    """
    strata = defaultdict(list)
    for r in rows:
        strata[bool(PAST.search(r["question"]))].append(r)
    scored, tot = [], 0
    for g in strata.values():
        a = auroc([(score(r), bool(r["is_evergreen"])) for r in g])
        if a is not None:
            scored.append((a, len(g)))
            tot += len(g)
    if not tot:
        return float("nan"), 0
    return sum(a * n for a, n in scored) / tot, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="eg_",
                    help="eg_ = the Aug runs; eg_ao_ = the answer-only rerun (results_sep10/MODELS.md)")
    args = ap.parse_args()
    files = sorted(PRED_DIR.glob(f"{args.prefix}*.jsonl"))
    # eg_* also matches eg_ao_*; keep only files whose remainder is a model tag
    files = [f for f in files if f.stem != "eg_driver"
             and not (args.prefix == "eg_" and f.stem.startswith("eg_ao_"))]
    if not files:
        raise SystemExit("no eg_*.jsonl yet")

    print("## EverGreenQA uncertainty, replicated and re-read")
    print("   scores oriented so HIGHER predicts EVERGREEN (negated PPL / entropy)")
    print("   within-form stratifies on the past-tense marker, which alone scores")
    print("   AUROC 0.6640 on these questions with no model consulted.\n")
    print(f"   {'model':14s} {'n':>4s} | {'r(PPL)':>7s} {'r(ent)':>7s} | "
          f"{'AUROC PPL':>9s} {'AUROC ent':>9s} | {'within PPL':>10s} {'within ent':>10s}")

    trend = []
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        rows = [r for r in rows if r.get("ppl") and r.get("entropy") is not None]
        if not rows:
            continue
        label = f.stem[len(args.prefix):]
        y = [float(r["is_evergreen"]) for r in rows]

        r_ppl = pearson([-r["ppl"] for r in rows], y)
        r_ent = pearson([-r["entropy"] for r in rows], y)
        a_ppl = auroc([(-r["ppl"], bool(r["is_evergreen"])) for r in rows])
        a_ent = auroc([(-r["entropy"], bool(r["is_evergreen"])) for r in rows])
        w_ppl, n1 = within_form(rows, lambda r: -r["ppl"])
        w_ent, _ = within_form(rows, lambda r: -r["entropy"])

        print(f"   {label:14s} {len(rows):4d} | {r_ppl:+7.3f} {r_ent:+7.3f} | "
              f"{a_ppl:9.4f} {a_ent:9.4f} | {w_ppl:10.4f} {w_ent:10.4f}")
        if label in SIZE:
            trend.append((SIZE[label], label, abs(r_ppl), a_ppl, w_ppl))

    if len(trend) >= 3:
        trend.sort()
        print("\n## the size trend, which is the disputed claim")
        print("   Pletenev et al.: larger models correlate MORE strongly.")
        print(f"\n   {'params':>7s} {'model':14s} {'|r| PPL':>8s} {'AUROC':>8s} {'within-form':>12s}")
        for p, lab, ar, a, w in trend:
            print(f"   {p:6d}B {lab:14s} {ar:8.3f} {a:8.4f} {w:12.4f}")
        small = [t for t in trend if t[0] <= 10]
        large = [t for t in trend if t[0] >= 20]
        if small and large:
            ms = sum(t[4] for t in small) / len(small)
            ml = sum(t[4] for t in large) / len(large)
            print(f"\n   mean within-form AUROC, <=10B: {ms:.4f}  ({len(small)} models)")
            print(f"   mean within-form AUROC, >=20B: {ml:.4f}  ({len(large)} models)")
            print(f"   difference (large - small):    {ml-ms:+.4f}")
            print("   positive supports their trend; negative supports B11-Y.")


if __name__ == "__main__":
    main()
