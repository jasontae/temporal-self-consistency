"""Check the Table 1 caption: "ECE is computed exactly on the four discrete
stated confidence levels without binning."

The shared pipeline's `compute_ece` (src/evaluation/eval_pipeline.py:34) sorts
by confidence and cuts 10 equal-frequency bins, so tied levels are split across
bins and adjacent levels can share one. This script recomputes every Table 1 row
both ways, reports where they differ, and recomputes the Brier score and its
Murphy decomposition, grouped by level (exact for a discrete forecast).

Why the two ECEs agree here: if every bin has mean accuracy <= mean confidence,
binned ECE = sum_b w_b (conf_b - acc_b) = mean(conf) - mean(acc), and if every
level also has accuracy <= its confidence, per-level ECE equals the same
quantity. Both conditions are checked below rather than assumed.

Run from the repo root:
    python3 docs/orientation/results_sep10/ece_check/ece_levels.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from eval_pipeline import HEDGE_TO_CONFIDENCE as H, compute_ece  # noqa: E402

PRED = ROOT / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent

# Values printed in the Sep 2 PDF, Table 1 (ECE, Brier, Resolution).
PAPER_T1 = {
    "oracle": (0.4504, 0.218, 0.002),
    "trained model": (0.4506, 0.219, 0.001),
    "cross-entropy baseline": (0.4552, 0.224, 0.001),
    "constant [TEMPORAL_HEDGE]": (0.4227, 0.187, 0.000),
    "constant [UNKNOWN]": (0.0727, 0.031, 0.000),
}


def load(name):
    return [json.loads(line) for line in open(PRED / f"{name}.jsonl")]


def arrays(recs, policy):
    y = np.array([float(r["correct"]) for r in recs])
    if policy == "model":
        c = np.array([H[r["predicted_hedge"]] for r in recs])
    elif policy == "oracle":
        c = np.array([H[r["gold_hedge"]] for r in recs])
    else:
        c = np.full(len(recs), H[policy])
    return c, y


def ece_by_level(c, y):
    return sum((c == k).mean() * abs(y[c == k].mean() - k) for k in np.unique(c))


def murphy(c, y):
    """Brier = reliability - resolution + uncertainty, grouped by level (exact)."""
    ob = y.mean()
    rel = sum((c == k).mean() * (k - y[c == k].mean()) ** 2 for k in np.unique(c))
    res = sum((c == k).mean() * (y[c == k].mean() - ob) ** 2 for k in np.unique(c))
    return {"brier": float(((c - y) ** 2).mean()), "reliability": float(rel),
            "resolution": float(res), "uncertainty": float(ob * (1 - ob))}


def bins_all_overconfident(recs, policy):
    """True if every equal-frequency bin of compute_ece has acc <= conf."""
    tagged = [dict(r, predicted_hedge=r["gold_hedge"]) if policy == "oracle"
              else r if policy == "model" else dict(r, predicted_hedge=policy)
              for r in recs]
    return all(b["accuracy"] <= b["avg_confidence"] for b in compute_ece(tagged)["bins"])


def main():
    rows = []
    arms = {"tsct_test": "trained model", "sft_test": "cross-entropy baseline"}
    for fname, label in arms.items():
        recs = load(fname)
        policies = [("model", label), ("oracle", "oracle"),
                    ("[TEMPORAL_HEDGE]", "constant [TEMPORAL_HEDGE]"),
                    ("[UNKNOWN]", "constant [UNKNOWN]")]
        for pol, name in policies:
            c, y = arrays(recs, pol)
            tagged = [dict(r, predicted_hedge=r["gold_hedge"]) if pol == "oracle"
                      else r if pol == "model" else dict(r, predicted_hedge=pol)
                      for r in recs]
            binned = compute_ece(tagged)["ece"]
            exact = ece_by_level(c, y)
            m = murphy(c, y)
            rows.append({
                "correctness_from": fname, "policy": name, "n": len(recs),
                "accuracy": float(y.mean()),
                "ece_binned_10": binned, "ece_by_level": round(float(exact), 6),
                "abs_diff": round(abs(binned - exact), 8),
                "all_bins_overconfident": bins_all_overconfident(recs, pol),
                **{k: round(v, 6) for k, v in m.items()},
                "paper": PAPER_T1.get(name) if (fname == "tsct_test" or name == label) else None,
            })

    # A case where the two differ, to show the agreement above is a property of
    # this data and not of the code: the untuned base model, constant [UNKNOWN].
    base = load("base_test")
    c, y = arrays(base, "[UNKNOWN]")
    counter = {"policy": "base model correctness, constant [UNKNOWN]",
               "accuracy": float(y.mean()),
               "ece_binned_10": compute_ece([dict(r, predicted_hedge="[UNKNOWN]") for r in base])["ece"],
               "ece_by_level": round(float(ece_by_level(c, y)), 6)}

    # Table 2 (repaired set, 1,953 items). The rebuild needs the train split,
    # which is not on disk. The equality still follows from counts alone:
    # accuracy <= 0.0046 means at most 9 correct items, and a bin holds
    # floor(1953/10) = 195 items, so the most accurate possible bin has
    # 9/195 = 0.046 < 0.10, the lowest level. Every bin is overconfident.
    t2 = {"n": 1953, "max_accuracy": 0.0046, "max_correct": int(np.ceil(0.0046 * 1953)),
          "bin_size": 1953 // 10}
    t2["max_bin_accuracy"] = round(t2["max_correct"] / t2["bin_size"], 4)
    t2["binned_equals_by_level"] = t2["max_bin_accuracy"] < 0.10

    json.dump({"table1": rows, "counterexample": counter, "table2_bound": t2},
              open(OUT / "ece_levels.json", "w"), indent=2)

    print(f"{'correctness':11s} {'policy':28s} {'acc':>6s} {'binned':>8s} {'level':>8s} "
          f"{'bins<=c':>7s} {'Brier':>7s} {'Rel':>7s} {'Res':>9s} | paper ECE/Brier/Res")
    for r in rows:
        p = r["paper"]
        ps = f"{p[0]:.4f}/{p[1]:.3f}/{p[2]:.3f}" if p else ""
        print(f"{r['correctness_from'][:11]:11s} {r['policy']:28s} {r['accuracy']:6.4f} "
              f"{r['ece_binned_10']:8.4f} {r['ece_by_level']:8.4f} {str(r['all_bins_overconfident']):>7s} "
              f"{r['brier']:7.4f} {r['reliability']:7.4f} {r['resolution']:9.6f} | {ps}")
    print("\ncounterexample:", counter)
    print("table 2 bound:", t2)


if __name__ == "__main__":
    main()
