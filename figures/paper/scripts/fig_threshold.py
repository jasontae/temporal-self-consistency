"""Figure 3 (main, §5): the threshold at which ECE starts rewarding a constant.

A. TemporalDelta, exact: ECE of the volatility-label (TVT) oracle and of the
   constant [UNKNOWN] as volatile-fact accuracy varies, slow-class accuracy and
   class mix held at the test set's values. Crossover 0.314; test set at 0.026.
B. FreshQA: the same answers re-scored against answer keys of increasing date.
   y = ECE(oracle) - ECE(constant [UNKNOWN]); above 0 the constant wins.

Sources: results_sep10/crossover/crossover.json (summary; the curve is the
closed form in crossover_sweep.py), results_sep10/item1/threshold.json.

    python3 figures/paper/scripts/fig_threshold.py
"""
import datetime as dt
import json
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from style import C, COL, FULL, RES, SIZE, apply, panel_title, save

sys.path.insert(0, str(RES / "crossover"))
from crossover_sweep import C_UNK, const_ece, oracle_ece  # noqa: E402

LABEL = {
    "g3_4b": "gemma-3-4B", "gptoss_20b": "gpt-oss-20B", "qwen25_7b": "Qwen2.5-7B",
    "qwen3_4b_it": "Qwen3-4B-Instruct", "qwen3_4b_th": "Qwen3-4B-Thinking",
    "qwen35_27b": "Qwen3.5-27B", "qwen36_27b_stock": "Qwen3.6-27B", "qwen36_35b": "Qwen3.6-35B-A3B",
}
STRONG = {"qwen35_27b": "o", "qwen36_27b_stock": "s", "qwen36_35b": "^"}   # the three that cross; markers, one ink colour
TSCT = ("adapter_tsct_seed0_paired", "adapter_tsct_seed1")
WORDS = {2: "two", 3: "three", 5: "five", 8: "eight"}


def decimal_year(s):
    d = dt.date.fromisoformat(s)
    return d.year + (d.timetuple().tm_yday - 1) / 365.25


def panel_td(ax, s, small):
    w_v, a_s, a_v, x0 = s["w_volatile"], s["acc_slow"], s["acc_volatile"], s["crossover_av_closed_form"]
    g = np.linspace(0, 1, 2001)
    o, u = oracle_ece(g, a_s, w_v), const_ece(g, a_s, w_v, C_UNK)
    ax.axvspan(0, x0, color=C["constant"], alpha=0.08, lw=0)
    ax.plot(g, o, color=C["oracle"])
    ax.plot(g, u, color=C["constant"])
    ax.text(0.70, 0.17, "TVT-label oracle", fontsize=SIZE["small"], color=C["oracle_text"], va="top")
    ax.text(0.66, 0.62, "constant [UNKNOWN]", fontsize=SIZE["small"], color=C["constant_text"], ha="right", va="bottom")
    ax.axvline(x0, color=C["muted"], lw=0.6, ls=(0, (2, 2)))
    ax.text(x0 + 0.02, 0.83, f"crossover\n{x0:.3f}", fontsize=SIZE["small"], va="top")
    ax.text(0.015, 0.83, "constant\nwins", fontsize=SIZE["small"], color=C["constant_text"], va="top")
    # the test set: both policies at the observed volatile accuracy, joined by a drop line (the 6.2x gap)
    yo, yc = s["ece_oracle"], s["ece_const_unknown"]
    ax.plot([a_v, a_v], [yc, yo], color=C["ink"], lw=0.6, ls=(0, (1, 1.2)), zorder=4)
    ax.plot([a_v], [yo], "o", color=C["oracle"], ms=3.5, zorder=5)
    ax.plot([a_v], [yc], "o", color=C["constant"], ms=3.5, zorder=5)
    ax.annotate(f"test set\n({a_v:.3f})", (a_v, 0.5 * (yo + yc) - 0.03), xytext=(4, 0),
                textcoords="offset points", fontsize=SIZE["small"], ha="left", va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("accuracy on volatile facts")
    ax.set_ylabel("expected calibration error (ECE)")


def spread(ys, gap):
    """Nudge label y positions apart so no two are closer than gap (order kept)."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        out[i] = max(out[i], out[j] + gap)
    return out - (out.mean() - np.mean(ys))   # keep the group centred on the line ends


TOP = 0.98   # headroom above the highest line (0.63) for the key


def panel_fq(ax, res, small):
    ends = {}
    for m in LABEL:
        pts = sorted(res[m]["contains"], key=lambda p: p["key_date"])
        xs = [decimal_year(p["key_date"]) for p in pts]
        ys = [p["ece_oracle"] - p["ece_const_unknown"] for p in pts]
        if m in STRONG:
            ax.plot(xs, ys, "-", marker=STRONG[m], color=C["ink"], lw=1.0, ms=3, mfc="white", mew=0.8, zorder=4)
            ends[m] = (xs[-1], ys[-1])
        else:
            ax.plot(xs, ys, "-", color=C["light"], lw=0.8, zorder=2)
    names = list(ends)
    ly = spread([ends[m][1] for m in names], 0.085 if small else 0.075)
    for m, y in zip(names, ly):
        ax.plot([ends[m][0] + 0.16], [y], marker=STRONG[m], color=C["ink"], ms=3, mfc="white", mew=0.8,
                clip_on=False)
        ax.text(ends[m][0] + 0.26, y, LABEL[m], fontsize=SIZE["tiny"], color=C["ink"], va="center")
    for m in TSCT:
        pts = sorted(res[m]["contains"], key=lambda p: p["key_date"])
        ax.plot([decimal_year(p["key_date"]) for p in pts], [p["ece_oracle"] - p["ece_const_unknown"] for p in pts],
                color=C["tsct"], lw=0.9, ls=(0, (3, 1.5)), zorder=3)
    ax.axhline(0, color=C["ink"], lw=0.6)
    ax.axhspan(0, TOP, color=C["constant"], alpha=0.08, lw=0)
    ax.text(2024.08, TOP - 0.03, "constant wins", fontsize=SIZE["small"], color=C["constant_text"], va="top")
    ax.text(2026.45, -0.19, "oracle wins", fontsize=SIZE["small"], color=C["oracle_text"], ha="right", va="bottom")
    n_other, n_tsct = len(LABEL) - len(STRONG), len(TSCT)
    words = WORDS
    key = [Line2D([], [], color=C["ink"], marker="o", ms=3, mfc="white", mew=0.8, lw=1.0,
                  label=f"{words[len(STRONG)]} strongest models (named at right)"),
           Line2D([], [], color=C["light"], lw=0.8, label=f"other {words[n_other]} models"),
           Line2D([], [], color=C["tsct"], lw=0.9, ls=(0, (3, 1.5)), label=f"{words[n_tsct]} TSCT-trained adapters")]
    ax.legend(handles=key, loc="upper right", bbox_to_anchor=(1.0 if small else 0.80, 1.0), fontsize=SIZE["tiny"],
              handlelength=2.0, borderaxespad=0.2, labelspacing=0.25)
    ax.set_ylim(-0.2, TOP)
    ax.set_xlim(2024.0, 2027.35 if not small else 2027.55)
    ax.set_xticks([2024, 2025, 2026])
    ax.spines["bottom"].set_bounds(2024.0, 2026.5)
    ax.set_xlabel("date of the FreshQA answer key")
    ax.set_ylabel("ECE(oracle) $-$ ECE(constant)")


def main():
    apply()
    s = json.load(open(RES / "crossover" / "crossover.json"))["summary"]
    res = json.load(open(RES / "item1" / "threshold.json"))
    out = []
    # full width
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.35), gridspec_kw={"width_ratios": [1, 1.15]})
    panel_td(a, s, False)
    panel_fq(b, res, False)
    panel_title(a, "A", "TemporalDelta (our benchmark), exact:\nECE of each policy as volatile-fact accuracy varies",
                x=-0.1)
    panel_title(b, "B", f"FreshQA: {WORDS[len(LABEL)]} off-the-shelf models and {WORDS[len(TSCT)]} TSCT-trained "
                "adapters,\nre-scored against answer keys of later dates", x=-0.1)
    out += save(fig, "fig3_threshold")
    # single column
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL, 4.3), )
    panel_td(a, s, True)
    panel_fq(b, res, True)
    panel_title(a, "A", "TemporalDelta (our benchmark), exact:\nECE as volatile-fact accuracy varies", x=-0.155)
    panel_title(b, "B", f"FreshQA: {WORDS[len(LABEL)]} off-the-shelf models and\n{WORDS[len(TSCT)]} TSCT-trained "
                "adapters, by answer-key date", x=-0.155)
    out += save(fig, "fig3_threshold_col")
    print("\n".join(map(str, out)))


if __name__ == "__main__":
    main()
