"""Item 3: does the failure disappear with a 0.01 level or continuous confidence?

"The failure" is that a policy with no item-level information (a constant)
gets better ECE than the volatility-label oracle. Three tests:

  (i)   Floor sweep, exact. Replace the lowest level 0.10 with f in [0, 0.45].
        The oracle's ECE does not involve the floor; the best constant's is
        |abar - f|. Evaluated on the released key and, from the ledger's
        accuracies, on the repaired key.
  (ii)  Free discrete vocabulary. Let each gold class assert its own realized
        accuracy (fit on held-out folds), i.e. a recalibrated oracle, and
        compare with a constant at the base rate fit the same way.
  (iii) Continuous confidence from the model's own answer log-probability
        (`mean_logprob` in *_test_lp.jsonl), mapped to a probability by
        isotonic regression and by Platt scaling, 5-fold cross-fitted so no
        record is scored by a map fitted on itself.

Each confidence is reported with ECE (pipeline estimator: 10 equal-frequency
bins; and 15 equal-width bins), Brier, a binned Murphy decomposition, and
AUROC for correctness. Resolution and AUROC are what distinguish an
informative confidence from a constant; ECE alone does not.

Run from the repo root:
    python3 docs/orientation/results_sep10/vocabulary/vocab_sweep.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[4]
PRED = ROOT / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent
H = {"[CONFIDENT]": 0.95, "[COND_CONFIDENT]": 0.75, "[TEMPORAL_HEDGE]": 0.45, "[UNKNOWN]": 0.10}
FILES = {"tsct_test_lp": "TSCT seed 0", "tsctS1_test_lp": "TSCT seed 1",
         "sftS1_test_lp": "SFT seed 1", "base_test_lp": "base Qwen2.5-7B"}
SEED, N_BOOT = 20260928, 2000


def ece_eqfreq(c, y, n_bins=10):
    """Same estimator as eval_pipeline.compute_ece: sort, cut equal-count bins."""
    o = np.argsort(c, kind="stable")
    c, y = c[o], y[o]
    size = max(1, len(c) // n_bins)
    return sum(len(c[i:i + size]) / len(c) * abs(c[i:i + size].mean() - y[i:i + size].mean())
               for i in range(0, len(c), size))


def ece_width(c, y, n_bins=15):
    idx = np.minimum((c * n_bins).astype(int), n_bins - 1)
    return sum((idx == b).mean() * abs(c[idx == b].mean() - y[idx == b].mean())
               for b in range(n_bins) if (idx == b).any())


def murphy_binned(c, y, n_bins=15):
    """Brier decomposition on 15 equal-width bins (exact when c takes <=15 values
    that fall in distinct bins, as for the four hedge levels)."""
    idx = np.minimum((c * n_bins).astype(int), n_bins - 1)
    ob = y.mean()
    rel = sum((idx == b).mean() * (c[idx == b].mean() - y[idx == b].mean()) ** 2
              for b in range(n_bins) if (idx == b).any())
    res = sum((idx == b).mean() * (y[idx == b].mean() - ob) ** 2
              for b in range(n_bins) if (idx == b).any())
    return rel, res, ob * (1 - ob)


def score(c, y):
    rel, res, unc = murphy_binned(c, y)
    auc = roc_auc_score(y, c) if np.ptp(c) > 0 else 0.5
    return {"ece_eqfreq10": float(ece_eqfreq(c, y)), "ece_width15": float(ece_width(c, y)),
            "brier": float(((c - y) ** 2).mean()), "reliability": float(rel),
            "resolution": float(res), "uncertainty": float(unc), "auroc": float(auc)}


def crossfit(x, y, groups_fn):
    """5-fold cross-fitted predictions from a map fitted on the other folds."""
    out = np.zeros(len(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(x.reshape(-1, 1), y):
        out[te] = groups_fn(x[tr], y[tr], x[te])
    return out


def iso(xtr, ytr, xte):
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(xtr, ytr).predict(xte)


def platt(xtr, ytr, xte):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(xtr.reshape(-1, 1), ytr)
    return m.predict_proba(xte.reshape(-1, 1))[:, 1]


def const_rate(xtr, ytr, xte):
    return np.full(len(xte), ytr.mean())


def class_rate(gtr, ytr, gte):
    rates = {g: ytr[gtr == g].mean() for g in np.unique(gtr)}
    return np.array([rates.get(g, ytr.mean()) for g in gte])


def main():
    results = {}
    rng = np.random.default_rng(SEED)
    for fname, label in FILES.items():
        recs = [json.loads(l) for l in open(PRED / f"{fname}.jsonl")]
        y = np.array([float(r["correct"]) for r in recs])
        lp = np.array([r["mean_logprob"] for r in recs])
        gold = np.array([H[r["gold_hedge"]] for r in recs])
        hedge = np.array([H[r["predicted_hedge"]] for r in recs])
        pols = {
            "emitted hedge (fixed levels)": hedge,
            "oracle (fixed levels)": gold,
            "constant [UNKNOWN] = 0.10": np.full(len(y), 0.10),
            "constant 0.01": np.full(len(y), 0.01),
            "oracle, levels refit per class (cross-fit)": crossfit(gold, y, class_rate),
            "constant at base rate (cross-fit)": crossfit(lp, y, const_rate),
            "exp(mean logprob), raw": np.exp(lp),
            "logprob, isotonic (cross-fit)": crossfit(lp, y, iso),
            "logprob, Platt (cross-fit)": crossfit(lp, y, platt),
        }
        rows = {k: score(v, y) for k, v in pols.items()}
        # bootstrap: Brier of isotonic logprob minus Brier of base-rate constant
        c_iso, c_k = pols["logprob, isotonic (cross-fit)"], pols["constant at base rate (cross-fit)"]
        d = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            d.append(((c_iso[i] - y[i]) ** 2).mean() - ((c_k[i] - y[i]) ** 2).mean())
        results[label] = {"n": len(y), "accuracy": float(y.mean()), "policies": rows,
                          "brier_iso_minus_const_ci95": np.percentile(d, [2.5, 97.5]).tolist()}

    # floor sweep (exact). Released key from tsct seed 0; repaired key accuracy
    # from specs/CLAIMS-LEDGER.md F-1 (TSCT 0.0026), oracle ECE 0.4788 there.
    floors = np.linspace(0, 0.45, 451)
    rel = results["TSCT seed 0"]
    a_rel = rel["accuracy"]
    o_rel = rel["policies"]["oracle (fixed levels)"]["ece_eqfreq10"]
    a_rep, o_rep = 0.0026, 0.4788
    sweep = {"floor": floors.tolist(),
             "released_const": np.abs(a_rel - floors).tolist(),
             "repaired_const": np.abs(a_rep - floors).tolist()}
    points = {}
    for f in (0.10, 0.05, 0.01, 0.0):
        points[f"{f:.2f}"] = {"released_const_ece": abs(a_rel - f), "released_ratio": o_rel / max(abs(a_rel - f), 1e-12),
                              "repaired_const_ece": abs(a_rep - f), "repaired_ratio": o_rep / max(abs(a_rep - f), 1e-12)}
    results["floor_sweep_points"] = points
    json.dump({"results": results, "floor_sweep": sweep}, open(OUT / "vocab.json", "w"), indent=2)

    # table
    with open(OUT / "vocab_table.md", "w") as f:
        f.write("### Floor sweep (exact; oracle ECE unchanged by the floor)\n\n")
        f.write("| lowest level | best-constant ECE, released key | oracle / constant | best-constant ECE, repaired key | oracle / constant |\n|---|---|---|---|---|\n")
        for k, v in points.items():
            f.write(f"| {k} | {v['released_const_ece']:.4f} | {v['released_ratio']:.1f}x | "
                    f"{v['repaired_const_ece']:.4f} | {v['repaired_ratio']:.1f}x |\n")
        f.write(f"\nReleased key: accuracy {a_rel:.4f}, oracle ECE {o_rel:.4f}. Repaired key: accuracy 0.0026, oracle ECE 0.4788 "
                "(ledger F-1; the repaired set is not rebuilt here).\n")
        for label in FILES.values():
            r = results[label]
            f.write(f"\n### {label} (n = {r['n']}, accuracy {r['accuracy']:.4f})\n\n")
            f.write("| confidence | ECE (10 eq-freq) | ECE (15 eq-width) | Brier | reliability | resolution | AUROC |\n|---|---|---|---|---|---|---|\n")
            for k, v in r["policies"].items():
                f.write(f"| {k} | {v['ece_eqfreq10']:.4f} | {v['ece_width15']:.4f} | {v['brier']:.4f} | "
                        f"{v['reliability']:.4f} | {v['resolution']:.5f} | {v['auroc']:.3f} |\n")
            lo, hi = r["brier_iso_minus_const_ci95"]
            f.write(f"\nBrier(isotonic logprob) - Brier(base-rate constant): 95% CI [{lo:+.4f}, {hi:+.4f}]\n")

    # figure: floor sweep, and ECE vs resolution for tsct seed 0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(floors, sweep["released_const"], color="#c0504d", lw=2, label="best constant, released key")
    ax.plot(floors, sweep["repaired_const"], color="#c0504d", lw=1.4, ls="--", label="best constant, repaired key")
    ax.axhline(o_rel, color="#1f4e79", lw=2, label="oracle, released key")
    ax.axhline(o_rep, color="#1f4e79", lw=1.4, ls="--", label="oracle, repaired key")
    for f_ in (0.10, 0.01):
        ax.axvline(f_, color="k", lw=0.6, ls=":")
        ax.text(f_ + 0.004, 0.3, f"floor {f_:.2f}", fontsize=8)
    ax.set_xlabel("lowest confidence level in the vocabulary")
    ax.set_ylabel("ECE")
    ax.set_title("lowering the floor widens the constant's win", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    r = results["TSCT seed 0"]["policies"]
    style = {  # marker, colour
        "emitted hedge (fixed levels)": ("s", "#7f7f7f"),
        "oracle (fixed levels)": ("D", "#7f7f7f"),
        "constant [UNKNOWN] = 0.10": ("o", "#c0504d"),
        "constant 0.01": ("v", "#c0504d"),
        "oracle, levels refit per class (cross-fit)": ("P", "#7f7f7f"),
        "constant at base rate (cross-fit)": ("X", "#c0504d"),
        "exp(mean logprob), raw": ("^", "#1f4e79"),
        "logprob, isotonic (cross-fit)": ("o", "#1f4e79"),
        "logprob, Platt (cross-fit)": ("s", "#1f4e79"),
    }
    for k, v in r.items():
        m, col = style[k]
        ax.scatter(v["ece_eqfreq10"], v["auroc"], s=45, marker=m, color=col, alpha=0.85,
                   label=k.replace(" (cross-fit)", "*"))
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.set_xlabel("ECE (pipeline estimator)")
    ax.set_ylabel("AUROC for correctness")
    ax.set_title("TSCT seed 0: ECE cannot tell a constant from an informative score", fontsize=9)
    ax.axhline(0.5, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "vocab.png", dpi=180)
    fig.savefig(OUT / "vocab.pdf")
    print(open(OUT / "vocab_table.md").read())


if __name__ == "__main__":
    main()
