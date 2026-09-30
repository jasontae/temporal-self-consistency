"""Item 1 main figure: the threshold curve.

ECE(constant [UNKNOWN]) vs ECE(volatility-label oracle) against volatile-fact
accuracy, across FreshQA answer-key ages, with the TemporalDelta crossover
(0.314, results_sep10/crossover/) and the TemporalDelta test set (0.026) marked.

Neither policy consults which items are correct, so any model's answers serve:
every model's FreshQA generations (data/prep/predictions_7b/fqgen_*.jsonl, from
generate_predictions --source freshqa) are scored against each dated key
(freshqa_built.jsonl `answers_by_date`). One point per model x key date.
Correctness: containment of any accepted answer (normalized); exact match as a
sensitivity. Levels: never 0.95, slow 0.75, fast 0.45; constant 0.10.

Also reported per point: Brier reliability and resolution (exact, by level) and
AUROC(oracle confidence -> correct), and FreshQA's own analytic crossover (fast
accuracy at which the constant ties the oracle, holding the point's never/slow
accuracies).

Run from the repo root:
    python3 docs/orientation/results_sep10/item1/freshqa_threshold.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
sys.path.insert(0, str(ROOT / "docs" / "orientation" / "results_sep10" / "crossover"))
from eval_pipeline import exact_match, normalize_answer  # noqa: E402
from crossover_sweep import oracle_ece as td_oracle, const_ece as td_const  # noqa: E402

PRED = ROOT / "data" / "prep" / "predictions_7b"
BUILT = ROOT / "data" / "prep" / "freshqa" / "freshqa_built.jsonl"
OUT = Path(__file__).resolve().parent
LEV = {"never-changing": 0.95, "slow-changing": 0.75, "fast-changing": 0.45}
EVAL_DATE = dt.date(2026, 9, 28)
TD_W_V, TD_A_S, TD_A_V, TD_CROSS = 0.9075096631695196, 0.04477611940298507, 0.0256, 0.314


def ok(pred, answers, mode):
    if mode == "exact":
        return any(exact_match(pred, a) for a in answers)
    p = normalize_answer(pred)
    return any(normalize_answer(a) and normalize_answer(a) in p for a in answers)


def point(recs, built, date, mode):
    rows = [(r, built[r["freshqa_id"]]) for r in recs if r.get("freshqa_id") in built
            and date in built[r["freshqa_id"]]["answers_by_date"]]
    y = np.array([float(ok(r["predicted_answer"], b["answers_by_date"][date], mode)) for r, b in rows])
    ft = np.array([b["fact_type"] for _, b in rows])
    c = np.array([LEV[f] for f in ft])
    acc = {f: float(y[ft == f].mean()) for f in LEV if (ft == f).any()}
    w = {f: float((ft == f).mean()) for f in LEV}
    ob = y.mean()
    e_or = sum(w[f] * abs(acc[f] - LEV[f]) for f in acc)
    e_un = abs(ob - 0.10)
    rel = sum(w[f] * (LEV[f] - acc[f]) ** 2 for f in acc)
    res = sum(w[f] * (acc[f] - ob) ** 2 for f in acc)
    auc = 0.5
    if 0 < y.sum() < len(y):
        from scipy import stats
        auc = float(stats.mannwhitneyu(c[y == 1], c[y == 0]).statistic / (y.sum() * (len(y) - y.sum())))
    # FreshQA's own crossover in fast accuracy, other classes held
    grid = np.linspace(0, 1, 2001)
    o = sum(w[f] * (np.abs(grid - LEV[f]) if f == "fast-changing" else abs(acc[f] - LEV[f])) for f in acc)
    ab = sum(w[f] * (grid if f == "fast-changing" else acc[f]) for f in acc)
    wins = grid[np.abs(ab - 0.10) < o]
    cross = float(wins.max()) if len(wins) and wins.min() == 0 else None
    age = (EVAL_DATE - dt.date.fromisoformat(date)).days / 365.25
    return {"n": len(rows), "key_date": date, "key_age_years": age, "accuracy": float(ob), "acc_by_type": acc,
            "ece_oracle": float(e_or), "ece_const_unknown": float(e_un), "constant_wins": bool(e_un < e_or),
            "oracle_reliability": float(rel), "oracle_resolution": float(res), "oracle_auroc": auc,
            "freshqa_crossover_fast_acc": cross}


def main():
    built = {r["id"]: r for r in map(json.loads, open(BUILT))}
    dates = sorted({d for r in built.values() for d in r["answers_by_date"]})
    files = sorted(PRED.glob("fqgen_*.jsonl")) + sorted(PRED.glob("freshqa_*.jsonl"))
    if not files:
        raise SystemExit("no fqgen_*.jsonl: run the item 1 generation step")
    res = {}
    for f in files:
        recs = [json.loads(l) for l in open(f)]
        res[f.stem.replace("fqgen_", "").replace("freshqa_", "adapter_")] = {mode: [point(recs, built, d, mode) for d in dates]
                                            for mode in ("contains", "exact")}
    json.dump(res, open(OUT / "threshold.json", "w"), indent=2)

    lines = ["| model | key date (age, y) | fast acc | never / slow acc | ECE oracle | ECE constant | constant wins | oracle reliability | resolution | AUROC | FreshQA crossover (fast acc) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, v in res.items():
        for p in v["contains"]:
            a = p["acc_by_type"]
            lines.append(f"| {m} | {p['key_date']} ({p['key_age_years']:.1f}) | {a.get('fast-changing', float('nan')):.3f} | "
                         f"{a.get('never-changing', float('nan')):.3f} / {a.get('slow-changing', float('nan')):.3f} | "
                         f"{p['ece_oracle']:.4f} | {p['ece_const_unknown']:.4f} | {'yes' if p['constant_wins'] else 'no'} | "
                         f"{p['oracle_reliability']:.4f} | {p['oracle_resolution']:.4f} | {p['oracle_auroc']:.3f} | "
                         f"{p['freshqa_crossover_fast_acc'] if p['freshqa_crossover_fast_acc'] is not None else '—'} |")
    (OUT / "threshold_table.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    # A: the analytic threshold on TemporalDelta's own mixture
    ax = axes[0]
    grid = np.linspace(0, 1, 1001)
    ax.plot(grid, td_oracle(grid, TD_A_S, TD_W_V), color="#1f4e79", lw=1.8, label="volatility-label oracle")
    ax.plot(grid, td_const(grid, TD_A_S, TD_W_V, 0.10), color="#c0504d", lw=1.8, label="constant [UNKNOWN]")
    ax.axvline(TD_CROSS, color="k", lw=0.8, ls=":")
    ax.text(TD_CROSS + 0.01, 0.62, "crossover 0.314", fontsize=8)
    ax.scatter([TD_A_V], [td_oracle(TD_A_V, TD_A_S, TD_W_V)], color="#1f4e79", zorder=5)
    ax.scatter([TD_A_V], [td_const(TD_A_V, TD_A_S, TD_W_V, 0.10)], color="#c0504d", zorder=5)
    ax.annotate("TemporalDelta test\n(0.026)", (TD_A_V, 0.07), xytext=(0.07, 0.2), fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_xlabel("volatile-fact accuracy")
    ax.set_ylabel("ECE")
    ax.set_xlim(0, 1)
    ax.set_title("A. TemporalDelta (analytic, exact)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    # B: FreshQA, the same model's answers scored against keys of different age
    ax = axes[1]
    for m, v in res.items():
        pts = sorted(v["contains"], key=lambda p: p["key_date"])
        xs = [dt.date.fromisoformat(p["key_date"]).year + (dt.date.fromisoformat(p["key_date"]).timetuple().tm_yday - 1) / 365.25
              for p in pts]
        ys = [p["ece_oracle"] - p["ece_const_unknown"] for p in pts]
        adapter = m.startswith("adapter_")
        ax.plot(xs, ys, marker="o", ms=3, lw=1, alpha=0.8, color="#bfbfbf" if adapter else None,
                label=None if adapter else m)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("date of the FreshQA answer key (the same answers, re-scored)")
    ax.set_ylabel("ECE(oracle) - ECE(constant [UNKNOWN])\n> 0: constant wins")
    ax.set_title("B. FreshQA: the constant overtakes the oracle as the key moves past\nthe model's knowledge (grey: hedge-trained Qwen2.5-7B adapters)", fontsize=9)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    # C: every point against its own crossover (identity; illustrates the rule)
    ax = axes[2]
    for m, v in res.items():
        for p in v["contains"]:
            c = p["freshqa_crossover_fast_acc"]
            if not c:
                continue
            ax.scatter(p["acc_by_type"]["fast-changing"] / c, p["ece_oracle"] - p["ece_const_unknown"],
                       s=16, color="#bfbfbf" if m.startswith("adapter_") else "#1f4e79", alpha=0.8)
    td_gap = float(td_oracle(TD_A_V, TD_A_S, TD_W_V) - td_const(TD_A_V, TD_A_S, TD_W_V, 0.10))
    ax.scatter([TD_A_V / TD_CROSS], [td_gap], marker="*", s=120, color="#c0504d", zorder=5, label="TemporalDelta test")
    ax.axvline(1, color="k", lw=0.8, ls=":")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("volatile accuracy / that benchmark's crossover")
    ax.set_ylabel("ECE(oracle) - ECE(constant)")
    ax.set_title("C. All points against their own crossover\n(an identity: illustrates the threshold rule)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "threshold.png", dpi=200)
    fig.savefig(OUT / "threshold.pdf")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
