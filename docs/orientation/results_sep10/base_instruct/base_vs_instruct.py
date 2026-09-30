"""Directive 3, Worker A item 5: Qwen2.5-7B base vs Qwen2.5-7B-Instruct.

Same pretraining; only post-training differs. Both are mlx-community 4-bit g64
conversions made with mlx-lm 0.18.1 (MODELS.md). If the stale-value preference
is a pretraining property, base and instruct should agree on it; a difference
localizes it to post-training. Contrasts (PMI, paired within entity): truth,
regime, staleness, on the original and the disjoint claim sets, with an
entity-paired bootstrap on every base - instruct difference.

Run from the repo root:
    python3 docs/orientation/results_sep10/base_instruct/base_vs_instruct.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent
C = {"truth": ("immutable_true", "immutable_false"), "regime": ("immutable_true", "volatile_current"),
     "staleness": ("volatile_current", "volatile_stale")}


def load(p):
    by = defaultdict(dict)
    for r in map(json.loads, open(p)):
        by[r["entity_id"]][r["arm"]] = r["pmi"]
    return {e: a for e, a in by.items() if len(a) == 4}


def auroc(pos, neg):
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    rng = np.random.default_rng(20260928)
    res, lines = {}, ["| claim set | contrast | base | instruct | base − instruct [95% CI] | p | sign test, base (paired within entity) |", "|---|---|---|---|---|---|---|"]
    for cs in ("matched", "disjoint"):
        fb, fi = P / f"{cs}_qwen25_7b_base.jsonl", P / f"{cs}_qwen25_7b.jsonl"
        if not fb.exists():
            raise SystemExit(f"{fb} missing: FETCH_EXTRA.sh step 1, then the queue's base-vs-instruct step")
        B, I = load(fb), load(fi)
        ents = sorted(set(B) & set(I))
        boots = rng.integers(0, len(ents), (4000, len(ents)))
        for c, (h, l) in C.items():
            bh, bl = np.array([B[e][h] for e in ents]), np.array([B[e][l] for e in ents])
            ih, il = np.array([I[e][h] for e in ents]), np.array([I[e][l] for e in ents])
            ab, ai = auroc(bh, bl), auroc(ih, il)
            d = np.array([auroc(bh[x], bl[x]) - auroc(ih[x], il[x]) for x in boots])
            p = float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))
            dd = bh - bl
            dd = dd[dd != 0]
            sp = float(stats.binomtest(int((dd > 0).sum()), len(dd)).pvalue)
            res[f"{cs}:{c}"] = {"base": ab, "instruct": ai, "diff": ab - ai, "ci95": np.percentile(d, [2.5, 97.5]).tolist(),
                                "p": p, "base_sign_p": sp, "n": len(ents)}
            lines.append(f"| {cs} | {c} | {ab:.3f} | {ai:.3f} | {ab - ai:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] | {p:.3f} | {sp:.4f} |")
    json.dump(res, open(OUT / "base_vs_instruct.json", "w"), indent=2)
    (OUT / "base_vs_instruct.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
