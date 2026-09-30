"""Appendix E: the constant's win belongs to the confidence vocabulary.

A. Floor sweep (exact): ECE of a constant at the vocabulary's lowest level,
   against the TVT-label oracle, on the released and repaired keys.
B-D. TSCT seed 0, released key: ECE, AUROC for correctness and Murphy
   resolution for the oracle, two constants and two recalibrated log-prob
   scores (* = 5-fold cross-fitted). ECE ties the informative score with a
   constant; AUROC and resolution separate them.

Source: results_sep10/vocabulary/vocab.json (vocab_sweep.py).

    python3 figures/paper/scripts/fig_vocab.py
"""
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from style import C, FULL, RES, SIZE, apply, panel_label, save

POL = [  # key in vocab.json, display label, colour
    ("oracle (fixed levels)", "TVT-label oracle", C["oracle"]),
    ("constant [UNKNOWN] = 0.10", "constant [UNKNOWN]", C["constant"]),
    ("constant at base rate (cross-fit)", "constant, base rate*", C["constant"]),
    ("logprob, isotonic (cross-fit)", "log-prob, isotonic*", C["score"]),
    ("logprob, Platt (cross-fit)", "log-prob, Platt*", C["score"]),
]
A_REP, O_REP = 0.0026, 0.4788   # repaired key, as in vocab_sweep.py (ledger F-1)


def main():
    apply()
    v = json.load(open(RES / "vocabulary" / "vocab.json"))
    sw, pts = v["floor_sweep"], v["results"]["floor_sweep_points"]
    r = v["results"]["TSCT seed 0"]["policies"]
    o_rel = r["oracle (fixed levels)"]["ece_eqfreq10"]

    fig = plt.figure(figsize=(FULL, 2.05))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 0.75, 0.75, 0.75])
    ax = fig.add_subplot(gs[0])
    f = np.array(sw["floor"])
    ax.plot(f, sw["released_const"], color=C["constant"], label="constant at floor, released key")
    ax.plot(f, sw["repaired_const"], color=C["constant"], ls=(0, (3, 1.5)), lw=1.0, label="same, repaired key")
    ax.axhline(o_rel, color=C["oracle"], label="oracle, released key")
    ax.axhline(O_REP, color=C["oracle"], ls=(0, (3, 1.5)), lw=1.0, label="oracle, repaired key")
    # the two floors in the text: each point labelled directly with its ratio ECE(oracle) / ECE(constant)
    place = {"0.10": ((3, -3), "left", "top"), "0.01": ((0, 5), "center", "bottom")}
    for fl, (off, ha, va) in place.items():
        x = float(fl)
        yv = pts[fl]["released_const_ece"]
        ratio = pts[fl]["released_ratio"]
        ax.plot([x], [yv], "o", color=C["constant"], ms=3, zorder=5)
        ax.annotate(f"{ratio:.1f}×" if ratio < 10 else f"{ratio:.0f}×", (x, yv), xytext=off,
                    textcoords="offset points", fontsize=SIZE["tiny"], ha=ha, va=va)
    ax.set_xlim(0, 0.15)
    ax.set_xticks([0, 0.05, 0.10, 0.15])
    ax.set_xticklabels(["0", "0.05", "0.10", "0.15"])
    ax.set_ylim(0, 0.52)
    ax.set_xlabel("lowest confidence level in the vocabulary")
    ax.set_ylabel("ECE")
    h, _ = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=C["constant"], marker="o", ms=3, lw=0,
                    label="oracle\u2019s ECE \u00f7 constant\u2019s,\nat that floor (released key)"))
    ax.legend(handles=h, loc="center right", fontsize=SIZE["tiny"], bbox_to_anchor=(1.0, 0.55), labelspacing=0.35)
    panel_label(ax, "A", x=-0.14)

    yy = np.arange(len(POL))[::-1]
    metrics = [("ece_eqfreq10", "ECE (log scale)", "B"), ("auroc", "AUROC for correctness", "C"),
               ("resolution", "resolution (×10$^{-3}$)", "D")]
    for k, (key, xl, letter) in enumerate(metrics):
        a = fig.add_subplot(gs[k + 1])
        vals = [r[p][key] * (1e3 if key == "resolution" else 1) for p, _, _ in POL]
        for yv, val, (_, _, col) in zip(yy, vals, POL):
            a.plot([val], [yv], "o", color=col, ms=4, zorder=4)
            if key != "ece_eqfreq10":
                a.plot([0.5 if key == "auroc" else 0, val], [yv, yv], color=col, lw=0.8)
        a.set_yticks(yy)
        a.set_yticklabels([lab for _, lab, _ in POL] if k == 0 else [])
        a.tick_params(axis="y", length=0)
        a.spines["left"].set_visible(False)
        a.set_ylim(-0.6, len(POL) - 0.4)
        a.grid(axis="y", color=C["fill"], lw=0.6)
        a.set_axisbelow(True)
        if key == "ece_eqfreq10":
            a.set_xscale("log")
            a.set_xlim(0.004, 0.7)
            a.set_xticks([0.01, 0.1])
            a.set_xticklabels(["0.01", "0.1"])
        elif key == "auroc":
            a.set_xlim(0.45, 0.9)
            a.axvline(0.5, color=C["muted"], lw=0.6, ls=(0, (2, 2)))
            a.set_xticks([0.5, 0.7, 0.9])
        else:
            a.set_xlim(0, 1.5)
            a.set_xticks([0, 0.5, 1.0, 1.5])
            a.set_xticklabels(["0", "0.5", "1", "1.5"])
        a.set_xlabel(xl)
        if k == 0:
            a.set_title("TSCT seed 0, released key", fontsize=SIZE["small"])
        panel_label(a, letter, x=0.0 if k else -0.02)
    print("\n".join(map(str, save(fig, "figE_vocab"))))


if __name__ == "__main__":
    main()
