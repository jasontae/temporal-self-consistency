"""Appendix D: the crossover plane and the age axis (TemporalDelta).

A. 1-D sweep at the test-set mixture: oracle, constant [UNKNOWN], and the best
   constant in the four-level vocabulary.
B. The (volatile, slow) accuracy plane: where constant [UNKNOWN] beats the
   oracle, and where the oracle beats every constant.
C. Age axis: an answerer that knows the world as of year Y, scaled by a
   knowledge rate k, scored against the key's own Wikidata validity intervals.

Source: results_sep10/crossover/crossover.json; closed forms imported from
crossover_sweep.py (neither policy looks at which items are correct, so the
curves are exact).

    python3 figures/paper/scripts/fig_crossover.py
"""
import json
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from style import C, FULL, RES, SIZE, apply, panel_label, save

sys.path.insert(0, str(RES / "crossover"))
from crossover_sweep import C_S, C_UNK, C_V, best_const_ece, const_ece, oracle_ece  # noqa: E402

K_STYLE = {1.0: (C["ink"], "-"), 0.5: (C["ink"], (0, (4, 1.5))), 0.2: (C["muted"], (0, (1.5, 1.2))),
           0.05: (C["light"], "-")}


def main():
    apply()
    d = json.load(open(RES / "crossover" / "crossover.json"))
    s, age = d["summary"], d["age"]
    w_v, a_s, a_v, x0 = s["w_volatile"], s["acc_slow"], s["acc_volatile"], s["crossover_av_closed_form"]

    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw={"width_ratios": [1, 0.9, 1]})
    ax = axes[0]
    g = np.linspace(0, 1, 2001)
    ax.plot(g, oracle_ece(g, a_s, w_v), color=C["oracle"], label="TVT-label oracle")
    ax.plot(g, const_ece(g, a_s, w_v, C_UNK), color=C["constant"], label="constant [UNKNOWN]")
    ax.plot(g, best_const_ece(g, a_s, w_v), color=C["ink"], lw=0.9, ls=(0, (3, 1.5)), label="best constant")
    ax.axvline(x0, color=C["muted"], lw=0.6, ls=(0, (2, 2)))
    ax.text(x0 + 0.02, 0.42, f"{x0:.3f}", fontsize=SIZE["small"], va="bottom")
    ax.plot([a_v], [s["ece_oracle"]], "o", color=C["oracle"], ms=3, zorder=5)
    ax.plot([a_v], [s["ece_const_unknown"]], "o", color=C["constant"], ms=3, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.88)
    ax.set_xlabel(f"volatile accuracy (slow held at {a_s:.3f})")
    ax.set_ylabel("ECE")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.03), fontsize=SIZE["tiny"], frameon=True, facecolor="white",
              edgecolor="none", framealpha=1)
    panel_label(ax, "A", x=-0.2)

    ax = axes[1]
    A_v, A_s = np.meshgrid(np.linspace(0, 1, 601), np.linspace(0, 1, 601))
    d_unk = oracle_ece(A_v, A_s, w_v) - const_ece(A_v, A_s, w_v, C_UNK)
    d_best = oracle_ece(A_v, A_s, w_v) - best_const_ece(A_v, A_s, w_v)
    region = np.where(d_unk >= 0, 0, np.where(d_best < 0, 2, 1))   # 0 [UNKNOWN] wins, 1 oracle beats it only, 2 beats all
    cols = ["#F6DCCB", "#D6E6F2", "#8DB8D9"]   # vermillion / blue tints
    ax.contourf(A_v, A_s, region, levels=[-0.5, 0.5, 1.5, 2.5], colors=cols)
    ax.plot([a_v], [a_s], "o", color=C["ink"], ms=3, zorder=5, clip_on=False)
    # the test set sits in the corner: its label goes directly beside it, inside the [UNKNOWN] region
    ax.annotate("test set", (a_v, a_s), xytext=(4, 0), textcoords="offset points", fontsize=SIZE["small"],
                ha="left", va="center")
    ax.plot([C_V], [C_S], "*", color=C["ink"], ms=6, zorder=5)
    ax.legend(handles=[Line2D([], [], color=C["ink"], marker="*", ms=6, lw=0, label="oracle\u2019s asserted levels")],
              loc="lower right", fontsize=SIZE["tiny"], handletextpad=0.3, borderaxespad=0.5, handlelength=1.0,
              frameon=True, facecolor="white", edgecolor=C["light"], framealpha=1, fancybox=False, borderpad=0.35)
    ax.set_xlabel("volatile accuracy")
    ax.set_ylabel("slow-class accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    # direct region labels
    ax.text(0.13, 0.55, "[UNKNOWN] beats oracle", rotation=90, fontsize=SIZE["tiny"], color=C["constant_text"],
            ha="center", va="center")
    ax.text(0.70, 0.36, "oracle beats\n[UNKNOWN]", fontsize=SIZE["tiny"], color=C["oracle_text"], ha="center", va="center")
    ax.text(0.63, 0.80, "oracle beats\nevery constant", fontsize=SIZE["tiny"], color=C["ink"], ha="left", va="center")
    panel_label(ax, "B", x=-0.2)

    ax = axes[2]
    for k, (col, ls) in K_STYLE.items():
        rows = [r for r in age if r["k"] == k]
        ax.plot([r["year"] for r in rows], [r["oracle"] - r["const_unknown"] for r in rows], color=col, ls=ls,
                lw=1.0, label=f"k = {k:g}")
    ax.axhline(0, color=C["ink"], lw=0.6)
    ax.set_xlabel("year of the answerer\u2019s knowledge")
    ax.set_ylabel("ECE(oracle) $-$ ECE([UNKNOWN])")
    ax.set_xticks([2000, 2010, 2020])
    ax.legend(loc="lower left", fontsize=SIZE["tiny"], title="knowledge rate", title_fontsize=SIZE["tiny"])
    panel_label(ax, "C", x=-0.2)
    print("\n".join(map(str, save(fig, "figD_crossover"))))


if __name__ == "__main__":
    main()
