"""Item 6: the 27B vintage ladder, Qwen3.5 -> Qwen3.6 -> Qwen3.8.

Same publisher, parameter count and quantization (4-bit affine g64); release
vintage differs. (Converter versions differ: mlx-vlm 0.3.12 / 0.4.4 / 0.6.8;
MODELS.md.) The Sep 2 paper has one pair (3.5 vs 3.6: staleness 0.4151 vs
0.4421) and calls it suggestive, not localizing. This adds the third point and
puts entity-bootstrap CIs on every AUROC and on every between-vintage
difference (the same entity resample for all models, so differences are paired).

The gemma-3-4B / gemma-4-E4B pair is no longer usable: Gemma-4 was dropped
(MODELS.md finding 1).

Run from the repo root:
    python3 docs/orientation/results_sep10/vintage/vintage_ladder.py [--prefix matched_]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
PRED = ROOT / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent
SEED, N_BOOT = 20260928, 4000
LADDER = [("qwen35_27b", "Qwen3.5-27B"), ("qwen36_27b_stock", "Qwen3.6-27B"), ("qwen38_27b", "Qwen3.8-27B")]
ARMS = ("immutable_true", "immutable_false", "volatile_current", "volatile_stale")
CONTRASTS = {"truth": (0, 1), "regime": (0, 2), "staleness": (2, 3)}


def load(path):
    by = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        by[r["entity_id"]][r["arm"]] = r["pmi"]
    return {e: a for e, a in by.items() if len(a) == 4}


def auroc(pos, neg):
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="matched_")
    args = ap.parse_args()
    tag = args.prefix.rstrip("_")
    data = {k: load(PRED / f"{args.prefix}{k}.jsonl") for k, _ in LADDER
            if (PRED / f"{args.prefix}{k}.jsonl").exists()}
    keys = [k for k, _ in LADDER if k in data]
    ents = sorted(set.intersection(*(set(data[k]) for k in keys)))
    S = {k: np.array([[data[k][e][a] for a in ARMS] for e in ents]) for k in keys}
    rng = np.random.default_rng(SEED)
    boots = rng.integers(0, len(ents), (N_BOOT, len(ents)))

    res = {"prefix": args.prefix, "n_entities": len(ents), "models": {}, "differences": {}}
    bt = {}
    for k in keys:
        res["models"][k] = {}
        for c, (i, j) in CONTRASTS.items():
            pt = auroc(S[k][:, i], S[k][:, j])
            b = np.array([auroc(S[k][ix, i], S[k][ix, j]) for ix in boots])
            bt[(k, c)] = b
            res["models"][k][c] = {"auroc": float(pt), "ci95": np.percentile(b, [2.5, 97.5]).tolist()}
        d = S[k][:, 2] - S[k][:, 3]
        d = d[d != 0]
        res["models"][k]["staleness_sign_p"] = float(stats.binomtest(int((d > 0).sum()), len(d)).pvalue)
    for a, b in zip(keys, keys[1:]):
        for c in CONTRASTS:
            diff = bt[(b, c)] - bt[(a, c)]
            pt = res["models"][b][c]["auroc"] - res["models"][a][c]["auroc"]
            res["differences"][f"{b} - {a}: {c}"] = {
                "diff": float(pt), "ci95": np.percentile(diff, [2.5, 97.5]).tolist(),
                "p_two_sided": float(min(1, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))}
    if len(keys) == 3:
        for c in CONTRASTS:
            diff = bt[(keys[2], c)] - bt[(keys[0], c)]
            pt = res["models"][keys[2]][c]["auroc"] - res["models"][keys[0]][c]["auroc"]
            res["differences"][f"{keys[2]} - {keys[0]}: {c}"] = {
                "diff": float(pt), "ci95": np.percentile(diff, [2.5, 97.5]).tolist(),
                "p_two_sided": float(min(1, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))}
    json.dump(res, open(OUT / f"vintage_{tag}.json", "w"), indent=2)

    lab = dict(LADDER)
    lines = [f"### Vintage ladder, 27B, 4-bit affine g64 ({args.prefix}*, {len(ents)} entities)\n",
             "| model | truth AUROC [95% CI] | regime AUROC [95% CI] | staleness AUROC [95% CI] | staleness sign p |",
             "|---|---|---|---|---|"]
    for k in keys:
        m = res["models"][k]
        f = lambda c: f"{m[c]['auroc']:.3f} [{m[c]['ci95'][0]:.3f}, {m[c]['ci95'][1]:.3f}]"
        lines.append(f"| {lab[k]} | {f('truth')} | {f('regime')} | {f('staleness')} | {m['staleness_sign_p']:.4f} |")
    lines += ["", "| difference (entity-paired bootstrap) | Δ AUROC | 95% CI | p |", "|---|---|---|---|"]
    for k, v in res["differences"].items():
        lines.append(f"| {k} | {v['diff']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['p_two_sided']:.3f} |")
    (OUT / f"vintage_table_{tag}.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for c, col in (("truth", "#1f4e79"), ("regime", "#7f7f7f"), ("staleness", "#c0504d")):
        y = [res["models"][k][c]["auroc"] for k in keys]
        lo = [y[i] - res["models"][k][c]["ci95"][0] for i, k in enumerate(keys)]
        hi = [res["models"][k][c]["ci95"][1] - y[i] for i, k in enumerate(keys)]
        ax.errorbar(range(len(keys)), y, yerr=[lo, hi], marker="o", color=col, capsize=3, label=c)
    ax.axhline(0.5, color="k", lw=0.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([lab[k] for k in keys])
    ax.set_ylabel("AUROC (PMI, 95% entity-bootstrap CI)")
    ax.set_title("27B vintage ladder, same publisher and 4-bit recipe", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"vintage_{tag}.png", dpi=180)
    fig.savefig(OUT / f"vintage_{tag}.pdf")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
