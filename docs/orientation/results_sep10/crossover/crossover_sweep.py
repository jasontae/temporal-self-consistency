"""Item 2: where does constant [UNKNOWN] start beating the volatility-label oracle?

Neither the oracle nor a constant policy looks at which items are correct, so
their ECE depends only on per-class accuracy. With classes k (gold hedge) of
share w_k, asserted level c_k and accuracy a_k, and overall accuracy
abar = sum_k w_k a_k:

    ECE(oracle)     = sum_k w_k |a_k - c_k|
    ECE(constant c) = |abar - c|

So the sweep over accuracy is exact, not simulated. Three views:

  1. 1-D: volatile accuracy a_v swept, slow-class accuracy held at its
     observed value, test-set mixture fixed. Closed-form crossover.
  2. 2-D: (a_v, a_s) plane, regions where the oracle beats constant
     [UNKNOWN] and where it beats the best constant in the vocabulary.
  3. Age: an ideal answerer that knows the world at date Y is right iff the
     key's value is valid at Y (t_start <= Y < t_end, from the benchmark's own
     Wikidata intervals). Scaled by a knowledge rate k, since real models know
     far fewer office-holders than an ideal one.

Also a record-level bootstrap CI on the 6.2x ratio.

Run from the repo root:
    python3 docs/orientation/results_sep10/crossover/crossover_sweep.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PRED = ROOT / "data" / "prep" / "predictions_7b"
AUDIT = ROOT / "data" / "prep" / "gold_currency_audit.json"
OUT = Path(__file__).resolve().parent

LEVELS = {"[CONFIDENT]": 0.95, "[COND_CONFIDENT]": 0.75, "[TEMPORAL_HEDGE]": 0.45, "[UNKNOWN]": 0.10}
C_V, C_S, C_UNK = 0.45, 0.75, 0.10
N_BOOT, SEED = 10000, 20260928


def oracle_ece(a_v, a_s, w_v):
    return w_v * np.abs(a_v - C_V) + (1 - w_v) * np.abs(a_s - C_S)


def const_ece(a_v, a_s, w_v, c):
    return np.abs(w_v * a_v + (1 - w_v) * a_s - c)


def best_const_ece(a_v, a_s, w_v):
    return np.min([const_ece(a_v, a_s, w_v, c) for c in LEVELS.values()], axis=0)


def crossover_av(a_s, w_v):
    """a_v at which ECE(const [UNKNOWN]) = ECE(oracle), for abar >= 0.10 and
    a_v <= 0.45, a_s <= 0.75 (the branch the data sits on):
        w_v a_v + w_s a_s - 0.10 = w_v (0.45 - a_v) + w_s (0.75 - a_s)
    """
    w_s = 1 - w_v
    return (C_UNK + C_V * w_v + C_S * w_s - 2 * w_s * a_s) / (2 * w_v)


def main():
    recs = [json.loads(l) for l in open(PRED / "tsct_test.jsonl")]
    y = np.array([float(r["correct"]) for r in recs])
    vol = np.array([r["gold_hedge"] == "[TEMPORAL_HEDGE]" for r in recs])
    w_v = vol.mean()
    a_v, a_s = y[vol].mean(), y[~vol].mean()
    obs_oracle = float(oracle_ece(a_v, a_s, w_v))
    obs_unk = float(const_ece(a_v, a_s, w_v, C_UNK))

    # bootstrap the 6.2x ratio over records (classes resampled jointly)
    rng = np.random.default_rng(SEED)
    ratios, gaps = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        yy, vv = y[i], vol[i]
        o = oracle_ece(yy[vv].mean(), yy[~vv].mean(), vv.mean())
        u = abs(yy.mean() - C_UNK)
        ratios.append(o / u)
        gaps.append(o - u)
    ratio_ci = np.percentile(ratios, [2.5, 97.5])
    gap_ci = np.percentile(gaps, [2.5, 97.5])

    # 1-D crossover at the observed slow-class accuracy, and a few alternatives
    x_obs = crossover_av(a_s, w_v)
    alt = {f"a_s={s:.2f}": float(crossover_av(s, w_v)) for s in (0.0, a_s, 0.25, 0.5, 0.75)}
    # best constant vs oracle along the same slice (numerical, fine grid)
    grid = np.linspace(0, 1, 100001)
    diff_best = oracle_ece(grid, a_s, w_v) - best_const_ece(grid, a_s, w_v)
    oracle_wins_best = grid[diff_best < 0]
    # Sanity: the closed form matches the grid.
    diff_unk = oracle_ece(grid, a_s, w_v) - const_ece(grid, a_s, w_v, C_UNK)
    x_grid = grid[np.argmax(diff_unk < 0)]

    # age axis from the benchmark's own validity intervals
    audit = {(r["question"], r["dataset_gold"]): r for r in json.load(open(AUDIT))["rows"]}
    ts, te, cls = [], [], []
    for r in recs:
        a = audit.get((r["question"], r["gold_answer"]))
        if a is None:
            continue
        ts.append(int(a["t_start"]) if a["t_start"] else -10**6)
        te.append(int(a["t_end"]) if a["t_end"] else 10**6)
        cls.append(r["gold_hedge"] == "[TEMPORAL_HEDGE]")
    ts, te, cls = map(np.array, (ts, te, cls))
    years = np.arange(2000, 2027)
    age_rows = []
    for k in (1.0, 0.5, 0.2, 0.05):
        for Y in years:
            valid = (ts <= Y) & (Y < te)
            av, as_ = k * valid[cls].mean(), k * valid[~cls].mean()
            age_rows.append({"k": k, "year": int(Y), "a_v": float(av), "a_s": float(as_),
                             "oracle": float(oracle_ece(av, as_, cls.mean())),
                             "const_unknown": float(const_ece(av, as_, cls.mean(), C_UNK)),
                             "best_const": float(best_const_ece(av, as_, cls.mean()))})
    cross_years = {}
    for k in (1.0, 0.5, 0.2, 0.05):
        rows = [r for r in age_rows if r["k"] == k]
        unk_wins = [r["year"] for r in rows if r["const_unknown"] < r["oracle"]]
        cross_years[str(k)] = unk_wins

    summary = {
        "n": len(y), "w_volatile": float(w_v), "acc_volatile": float(a_v), "acc_slow": float(a_s),
        "ece_oracle": obs_oracle, "ece_const_unknown": obs_unk,
        "ratio": obs_oracle / obs_unk, "ratio_ci95": ratio_ci.tolist(),
        "gap_ci95": gap_ci.tolist(),
        "crossover_av_closed_form": float(x_obs), "crossover_av_grid": float(x_grid),
        "crossover_av_by_slow_acc": alt,
        "oracle_beats_best_constant_av_range": ([float(oracle_wins_best.min()), float(oracle_wins_best.max())]
                                                 if len(oracle_wins_best) else None),
        "age_years_where_const_unknown_beats_oracle": cross_years,
        "age_n_matched": int(len(ts)),
    }
    json.dump({"summary": summary, "age": age_rows}, open(OUT / "crossover.json", "w"), indent=2)

    # table
    with open(OUT / "crossover_table.md", "w") as f:
        f.write("| slow-class accuracy a_s | constant [UNKNOWN] beats oracle when volatile accuracy a_v < |\n|---|---|\n")
        for kk, v in alt.items():
            f.write(f"| {kk[4:]} | {v:.4f} |\n")
        f.write(f"\nObserved: a_v = {a_v:.4f}, a_s = {a_s:.4f}, w_v = {w_v:.4f}. "
                f"Oracle {obs_oracle:.4f}, constant [UNKNOWN] {obs_unk:.4f}, ratio {obs_oracle/obs_unk:.2f} "
                f"(95% bootstrap CI {ratio_ci[0]:.2f}-{ratio_ci[1]:.2f}, {N_BOOT} record resamples).\n")
        f.write("\n| k (knowledge rate) | years in 2000-2026 where constant [UNKNOWN] beats oracle |\n|---|---|\n")
        for k, ys in cross_years.items():
            f.write(f"| {k} | {_ranges(ys)} |\n")

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot(grid, oracle_ece(grid, a_s, w_v), color="#1f4e79", lw=2, label="volatility-label oracle")
    ax.plot(grid, const_ece(grid, a_s, w_v, C_UNK), color="#c0504d", lw=2, label="constant [UNKNOWN]")
    ax.plot(grid, best_const_ece(grid, a_s, w_v), color="#7f7f7f", lw=1.2, ls="--", label="best constant in vocabulary")
    ax.axvline(x_obs, color="k", lw=0.8, ls=":")
    ax.annotate(f"crossover\na_v = {x_obs:.3f}", (x_obs, 0.55), xytext=(x_obs + 0.04, 0.6), fontsize=9)
    ax.scatter([a_v], [obs_oracle], color="#1f4e79", zorder=5)
    ax.scatter([a_v], [obs_unk], color="#c0504d", zorder=5)
    ax.annotate("released test set\n(a_v = 0.026)", (a_v, obs_unk), xytext=(0.06, 0.18), fontsize=9,
                arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_xlabel("volatile-class accuracy a_v (slow class held at 0.045)")
    ax.set_ylabel("ECE")
    ax.set_xlim(0, 1)
    ax.set_title("1-D sweep at the test-set mixture", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    A_v, A_s = np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401))
    d_unk = oracle_ece(A_v, A_s, w_v) - const_ece(A_v, A_s, w_v, C_UNK)
    d_best = oracle_ece(A_v, A_s, w_v) - best_const_ece(A_v, A_s, w_v)
    ax.contourf(A_v, A_s, (d_unk < 0).astype(float) + (d_best < 0), levels=[-0.5, 0.5, 1.5, 2.5],
                colors=["#f2dcdb", "#dce6f1", "#9dc3e6"])
    ax.contour(A_v, A_s, d_unk, levels=[0], colors="#c0504d", linewidths=1.5)
    ax.contour(A_v, A_s, d_best, levels=[0], colors="#1f4e79", linewidths=1.5)
    ax.scatter([a_v], [a_s], color="k", zorder=5)
    ax.annotate("released test set", (a_v, a_s), xytext=(0.08, 0.1), fontsize=9)
    ax.scatter([C_V], [C_S], marker="*", s=90, color="k")
    ax.annotate("accuracy = asserted level", (C_V, C_S), xytext=(0.5, 0.82), fontsize=9)
    ax.set_xlabel("volatile-class accuracy a_v")
    ax.set_ylabel("slow-class accuracy a_s")
    ax.set_title("pink: constant [UNKNOWN] beats oracle\nlight blue: oracle beats [UNKNOWN] only; dark: beats every constant",
                 fontsize=9)

    ax = axes[2]
    colors = {1.0: "#1f4e79", 0.5: "#2e75b6", 0.2: "#9dc3e6", 0.05: "#bfbfbf"}
    for k in (1.0, 0.5, 0.2, 0.05):
        rows = [r for r in age_rows if r["k"] == k]
        ax.plot([r["year"] for r in rows], [r["oracle"] - r["const_unknown"] for r in rows],
                color=colors[k], lw=1.8, label=f"k = {k}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("date Y at which an answerer knows the world")
    ax.set_ylabel("ECE(oracle) - ECE(constant [UNKNOWN])\n> 0: constant wins")
    ax.set_title("age axis from the key's own validity intervals", fontsize=10)
    ax.legend(fontsize=8, frameon=False, title="knowledge rate", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "crossover.png", dpi=180)
    fig.savefig(OUT / "crossover.pdf")

    print(json.dumps(summary, indent=2))


def _ranges(ys):
    if not ys:
        return "none"
    out, start, prev = [], ys[0], ys[0]
    for y in ys[1:] + [None]:
        if y is None or y != prev + 1:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            if y is not None:
                start = y
        if y is not None:
            prev = y
    return ", ".join(out)


if __name__ == "__main__":
    main()
