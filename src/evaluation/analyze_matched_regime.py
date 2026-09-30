"""Analyse the matched within-entity regime set (B11-X).

Every entity contributes all four arms in one frame, so each contrast can be
taken **paired within entity**. That is stronger than the AUROC B11 reported:
entity popularity, topic and tokenization are differenced out exactly, and the
sign test needs no distributional assumption.

Three contrasts, each answering a question B11's set could not separate:

  regime     volatile_current vs immutable_true
             Both true, both a person, same frame. Any gap is volatility.
             **This is the headline test.**

  truth      immutable_true vs immutable_false
             Volatility held out. Establishes what a truth effect looks like,
             so the regime effect can be read against it rather than in a
             vacuum.

  staleness  volatile_current vs volatile_stale
             Does the model prefer the value holding now, or the one the
             dataset froze? This is C9's question measured per claim.

Each is reported twice:

  raw  mean per-token log-probability
  pmi  logprob(claim) - logprob("<person> is a person.")

`pmi` nets out the marginal frequency of the name. Founders skew historically
famous and sitting office-holders can be obscure, so a gap present in `raw` and
absent in `pmi` is a name-frequency effect, not a regime effect. Both are
printed side by side and neither is reported alone.

Usage:
    python3 -m src.evaluation.analyze_matched_regime
"""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .b11_surface_control import auroc

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "data" / "prep" / "predictions_7b"
N_BOOT = 2000
SEED = 20260812

CONTRASTS = [
    ("regime", "immutable_true", "volatile_current"),
    ("truth", "immutable_true", "immutable_false"),
    ("staleness", "volatile_current", "volatile_stale"),
]


def sign_test(diffs):
    """Two-sided exact binomial on sign(diff), ties dropped. Returns (k, n, p)."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    k = sum(1 for d in nz if d > 0)
    if n == 0:
        return 0, 0, 1.0
    # exact two-sided p under Binomial(n, 0.5)
    def c(n, r):
        return math.comb(n, r)
    tail = min(k, n - k)
    p = 2 * sum(c(n, i) for i in range(tail + 1)) / (2 ** n)
    return k, n, min(1.0, p)


def boot_mean_ci(vals, n_boot=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    n = len(vals)
    if n < 2:
        return None
    out = []
    for _ in range(n_boot):
        out.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    out.sort()
    return out[int(0.025 * n_boot)], out[int(0.975 * n_boot)]


def load(path):
    """Index rows by entity, then arm."""
    by_ent = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        by_ent[r["entity_id"]][r["arm"]] = r
    # keep only entities with all four arms intact
    return {e: a for e, a in by_ent.items() if len(a) == 4}


def report(path):
    ents = load(path)
    label = path.stem.replace("matched_", "")
    print(f"\n### {label}   ({len(ents)} entities, all four arms)")
    print(f"    {'contrast':11s} {'field':4s} {'mean diff':>10s} {'95% CI':>18s} "
          f"{'sign k/n':>10s} {'p':>9s} {'AUROC':>7s}")

    for name, hi_arm, lo_arm in CONTRASTS:
        for field in ("mean_logprob", "pmi"):
            diffs, pairs = [], []
            for a in ents.values():
                x, y = a[hi_arm].get(field), a[lo_arm].get(field)
                if x is None or y is None:
                    continue
                diffs.append(x - y)
                pairs.append((x, True))
                pairs.append((y, False))
            if not diffs:
                continue
            md = sum(diffs) / len(diffs)
            ci = boot_mean_ci(diffs)
            ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "n/a"
            k, n, p = sign_test(diffs)
            a = auroc(pairs)
            p_s = f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"
            fld = "raw" if field == "mean_logprob" else "pmi"
            print(f"    {name:11s} {fld:4s} {md:+10.4f} {ci_s:>18s} "
                  f"{k:4d}/{n:<5d} {p_s:>9s} {a:7.4f}")


def main():
    files = sorted(PRED_DIR.glob("matched_*.jsonl"))
    if not files:
        raise SystemExit("no matched_*.jsonl yet")

    print("## B11-X: matched within-entity regime set")
    print("   diff is (first arm - second arm) per entity; positive means the")
    print("   first arm is LESS surprising. 'regime' is the headline test:")
    print("   immutable_true vs volatile_current, both TRUE, both a person name.")

    for f in files:
        report(f)

    print("\n## how to read this")
    print("   regime raw >0 with pmi ~0  -> name frequency, not volatility")
    print("   regime pmi >0, p small     -> genuine regime signal")
    print("   regime effect << truth     -> logprob tracks truth, not volatility")


if __name__ == "__main__":
    main()
