"""B11 re-analysis on distinct claims, with the proper-noun confound held out.

`stress_mixed_paragraphs.jsonl` presents 233 claims across 60 passages, but
only **22 of those claim texts are distinct** -- "Germany's chancellor is Olaf
Scholz." appears 17 times. Because AUROC counts positive-negative *pairs*, a
text repeated 17 times contributes 17x its weight, and the reported n=153
overstates the evidence by roughly an order of magnitude. Every B11 figure so
far, including the headline 0.9183, is computed on that inflated sample.

This module recomputes the claim on distinct texts, scored once each, and adds
the two things the inflated version cannot support:

  - a **bootstrap interval** over distinct claims, which is what the 22-item
    sample can actually justify;
  - a **within-form** contrast restricted to person-name predicates, since
    100% of volatile claims carry a person name against 34-41% of stable ones,
    making "is the predicate a proper noun" an alternative explanation with no
    model in it.

Usage:
    python3 -m src.evaluation.b11_dedup
"""
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from .b11_surface_control import auroc, has_person_name

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "data" / "prep" / "predictions_7b"
N_BOOT = 2000
SEED = 20260812


def distinct_claims(rows):
    """One row per distinct claim text; identical texts score identically."""
    seen = {}
    for r in rows:
        seen.setdefault(r["text"], r)
    return list(seen.values())


def boot_ci(pairs, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap over claims. Returns (lo, hi) or None."""
    rng = random.Random(seed)
    n = len(pairs)
    vals = []
    for _ in range(n_boot):
        samp = [pairs[rng.randrange(n)] for _ in range(n)]
        a = auroc(samp)
        if a is not None:
            vals.append(a)
    if len(vals) < 100:
        return None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def main():
    files = sorted(PRED_DIR.glob("mixed_*.jsonl"))
    ref = [json.loads(l) for l in open(files[0])]
    uniq = distinct_claims(ref)

    print(f"## sample size, as presented vs as distinct")
    print(f"   rows in file            {len(ref)}")
    print(f"   distinct claim texts    {len(uniq)}")
    by_v = defaultdict(int)
    for r in uniq:
        by_v[r["volatility"]] += 1
    print(f"   distinct by class       "
          + ", ".join(f"{v}={n}" for v, n in sorted(by_v.items())))
    named = [r for r in uniq if has_person_name(r["text"])]
    print(f"   distinct w/ person name {len(named)}")
    nv = sum(1 for r in named if r["is_volatile"])
    print(f"      of which volatile    {nv}   stable {len(named)-nv}")

    print(f"\n## AUROC(logprob -> claim is STABLE) on DISTINCT claims")
    print(f"   bootstrap 95% CI over claims, {N_BOOT} resamples, seed {SEED}\n")
    print(f"   {'model':16s} {'inflated':>9s} {'distinct':>9s} {'95% CI':>18s} "
          f"{'names only':>11s} {'95% CI':>18s}")

    for f in files:
        rows = [json.loads(l) for l in open(f)]
        u = distinct_claims(rows)

        # inflated: the figure B11 currently reports, pooled over all rows
        infl = auroc([(r["mean_logprob"], not r["is_volatile"]) for r in rows])

        pairs = [(r["mean_logprob"], not r["is_volatile"]) for r in u]
        d = auroc(pairs)
        ci = boot_ci(pairs)
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"

        nm = [r for r in u if has_person_name(r["text"])]
        npairs = [(r["mean_logprob"], not r["is_volatile"]) for r in nm]
        na = auroc(npairs)
        nci = boot_ci(npairs)
        na_s = f"{na:.4f}" if na is not None else "n/a"
        nci_s = f"[{nci[0]:.3f}, {nci[1]:.3f}]" if nci else "n/a"

        label = f.stem.replace("mixed_", "")
        print(f"   {label:16s} {infl:9.4f} {d:9.4f} {ci_s:>18s} "
              f"{na_s:>11s} {nci_s:>18s}")

    print(f"\n## surface baselines on DISTINCT claims (no model consulted)")
    for name, fn in (
        ("NOT has_person_name", lambda r: float(not has_person_name(r["text"]))),
        ("has_digit", lambda r: float(any(c.isdigit() for c in r["text"]))),
    ):
        pairs = [(fn(r), not r["is_volatile"]) for r in uniq]
        a = auroc(pairs)
        ci = boot_ci(pairs)
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
        print(f"   {name:22s} {a:.4f}  {ci_s}")


if __name__ == "__main__":
    main()
