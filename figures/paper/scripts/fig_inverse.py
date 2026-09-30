"""Appendix O: the inverse result, secondary and direction-only.

Regime AUROC (immutable true vs volatile current) against truth AUROC (true
vs false founder) for the nine admitted models, disjoint false-founder build.
A: mixed quantization (as scored Aug 13). B: the fixed-quantization ladder
(item 4). Bars: 95% entity-bootstrap CIs. Line: OLS fit over the nine.

Sources: results_sep10/inverse/inverse_disjoint.json (A),
results_sep10/inverse/inverse_disjoint4b.json (B).

    python3 figures/paper/scripts/fig_inverse.py
"""
import json

import numpy as np
from matplotlib import pyplot as plt

from style import C, FULL, RES, SIZE, apply, panel_title, place_labels, save, signed

SHORT = {
    "g3_4b": "gemma-3-4B", "qwen3_4b_th": "Qwen3-4B-Thinking", "tsct": "Qwen2.5-7B + TSCT",
    "qwen3_4b_it": "Qwen3-4B-Instruct", "qwen25_7b": "Qwen2.5-7B", "gptoss_20b": "gpt-oss-20B",
    "qwen36_35b": "Qwen3.6-35B-A3B", "qwen35_27b": "Qwen3.5-27B", "qwen36_27b_stock": "Qwen3.6-27B",
}
XLIM, YLIM = (0.62, 0.92), (0.40, 0.76)


FIXED = {"A": {"Qwen2.5-7B + TSCT": (7, 6, "center", "bottom"), "Qwen2.5-7B": (5, 0, "left", "center")}, "B": {}}


def panel(ax, d, letter):
    pm = d["per_model"]
    ml = d["model_level"]
    ms = d["admitted"]
    soft = []
    for m in ms:
        p = pm[m]
        tsct = m == "tsct"
        ax.plot(p["truth_ci"], [p["regime"]] * 2, color="#E4C3D6" if tsct else "#E2E2E2", lw=0.5, zorder=1)
        ax.plot([p["truth"]] * 2, p["regime_ci"], color="#E4C3D6" if tsct else "#E2E2E2", lw=0.5, zorder=1)
        ax.plot(p["truth"], p["regime"], "o", color=C["tsct"] if tsct else C["ink"], ms=3.2, zorder=4)
        soft += [(p["truth_ci"], [p["regime"]] * 2), ([p["truth"]] * 2, p["regime_ci"])]
    xs = np.array([pm[m]["truth"] for m in ms])
    g = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 10)
    fit = ml["ols_intercept"] + ml["ols_slope"] * g
    ax.plot(g, fit, color=C["oracle"], lw=1.0, zorder=3)
    ax.axhline(0.5, color=C["muted"], lw=0.6, ls=(0, (2, 2)), zorder=2)
    ax.text(XLIM[0] + 0.004, 0.497, "chance", fontsize=SIZE["tiny"], color=C["muted"], va="top")
    lo, hi = ml["slope_entity_bootstrap_ci"]
    ax.text(0.98, 0.97, f"fit slope {signed(ml['ols_slope'])} [{signed(lo)}, {signed(hi)}]\n"
            f"permutation p = {ml['exact_permutation_p']:.3f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=SIZE["small"], color=C["oracle_text"])
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    place_labels(ax, [(pm[m]["truth"], pm[m]["regime"]) for m in ms], [SHORT[m] for m in ms],
                 colors=[C["tsct_text"] if m == "tsct" else C["ink"] for m in ms],
                 avoid=[(g, fit)], soft=soft + [([XLIM[0], XLIM[1]], [0.5, 0.5])],
                 keep_out=[(0.80, 0.69, 0.92, 0.76), (XLIM[0], 0.475, XLIM[0] + 0.03, 0.50)], fixed=FIXED[letter])


def main():
    apply()
    mixed = json.load(open(RES / "inverse" / "inverse_disjoint.json"))
    fixed = json.load(open(RES / "inverse" / "inverse_disjoint4b.json"))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.45), sharey=True)
    for ax, letter, title in ((axes[0], "A", "mixed quantization (as downloaded)"),
                              (axes[1], "B", "all models at 4-bit")):
        ax.set_xlabel("truth AUROC (true vs false founder)")
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        panel_title(ax, letter, title, x=-0.13 if letter == "A" else -0.02)
    axes[0].set_ylabel("regime AUROC (volatile vs immutable)")
    fig.canvas.draw()   # fix the layout before placing labels in display space
    fig.set_layout_engine("none")
    panel(axes[0], mixed, "A")
    panel(axes[1], fixed, "B")
    print("\n".join(map(str, save(fig, "figO_inverse"))))


if __name__ == "__main__":
    main()
