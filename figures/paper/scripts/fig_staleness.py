"""Figure 4 (main, §7): the preference for expired values, per model.

Staleness AUROC (current vs expired office-holder; < 0.5 means the expired
value is less surprising) for the nine admitted models on the fixed 4-bit
ladder, with 95% entity-bootstrap CIs. Beside each model, its score under the
"As of 2026," prefix. Qwen2.5-7B base sits under its instruct checkpoint.
The bottom row is the mean over the nine models.

Sources: results_sep10/inverse/inverse_matched4b.json (per model, 4-bit),
results_sep10/date_probe/date_probe.json ("As of 2026," per model and pooled),
results_sep10/base_instruct/base_vs_instruct.json (base vs instruct).

    python3 figures/paper/scripts/fig_staleness.py
"""
import json

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from style import C, COL, RES, SIZE, apply, save

LABEL = {
    "g3_4b": "gemma-3-4B", "gptoss_20b": "gpt-oss-20B", "qwen3_4b_th": "Qwen3-4B-Thinking",
    "qwen3_4b_it": "Qwen3-4B-Instruct", "qwen35_27b": "Qwen3.5-27B", "qwen36_27b_stock": "Qwen3.6-27B",
    "qwen36_35b": "Qwen3.6-35B-A3B", "qwen25_7b": "Qwen2.5-7B-Instruct", "tsct": "Qwen2.5-7B + TSCT",
}
# top to bottom: the nine ladder models (six others by size, then Qwen2.5-7B instruct and + TSCT),
# a rule, the Qwen2.5-7B base checkpoint (not one of the nine), a rule, the mean of the nine
ORDER = ["g3_4b", "qwen3_4b_th", "qwen3_4b_it", "gptoss_20b", "qwen35_27b", "qwen36_27b_stock",
         "qwen36_35b", "qwen25_7b", "tsct"]
DIAMOND = dict(marker="D", color=C["accent"], mec="#8C5F00", mew=0.5, lw=0)
UP = 0.3   # the "As of 2026," diamond sits this far above its model's row


def main():
    apply()
    inv = json.load(open(RES / "inverse" / "inverse_matched4b.json"))
    dp = json.load(open(RES / "date_probe" / "date_probe.json"))
    bi = json.load(open(RES / "base_instruct" / "base_vs_instruct.json"))["matched:staleness"]
    pm = inv["per_model"]
    # the base/instruct run and the 4-bit ladder score the same instruct checkpoint
    assert abs(bi["instruct"] - pm["qwen25_7b"]["staleness"]) < 1e-9
    assert sorted(ORDER) == sorted(inv["admitted"])

    fig, ax = plt.subplots(figsize=(COL, 3.15))
    rows = ORDER + ["RULE1", "BASE", "RULE2", "MEAN"]
    step = {"RULE1": 0.8, "BASE": 0.8, "RULE2": 0.8, "MEAN": 0.8}
    y, cur = {}, 0.0
    for r in reversed(rows):   # bottom up; rules and the rows next to them get a little less space
        y[r] = cur
        cur += step.get(r, 1.0)
    for r in ORDER:
        yy = y[r]
        m = pm[r]
        col = C["tsct"] if r == "tsct" else C["ink"]
        lo, hi = m["staleness_ci"]
        ax.plot([lo, hi], [yy, yy], color=col, lw=0.9, solid_capstyle="butt")
        ax.plot(m["staleness"], yy, "o", color=col, ms=3.4, zorder=4)
        ax.plot(dp["per_model"][r]["asof2026"], yy + UP, ms=3.4, zorder=5, **DIAMOND)
    ax.plot(bi["base"], y["BASE"], "s", color=C["ink"], mfc="white", mew=0.9, ms=3.6, zorder=4)
    # pooled mean over the nine
    pooled = dp["pooled"]
    m0, (l0, h0) = pooled["mean staleness, none"]["mean"], pooled["mean staleness, none"]["ci95"]
    m6, (l6, h6) = pooled["mean staleness, asof2026"]["mean"], pooled["mean staleness, asof2026"]["ci95"]
    assert abs(m0 - sum(pm[k]["staleness"] for k in inv["admitted"]) / len(inv["admitted"])) < 1e-9
    yy = y["MEAN"]
    ax.plot([l0, h0], [yy, yy], color=C["ink"], lw=1.6, solid_capstyle="butt")
    ax.plot(m0, yy, "o", color=C["ink"], ms=4.2, zorder=4)
    ax.plot([l6, h6], [yy + UP] * 2, color=C["accent"], lw=1.2, solid_capstyle="butt")
    ax.plot(m6, yy + UP, ms=3.8, zorder=5, **DIAMOND)

    ax.axvline(0.5, color=C["muted"], lw=0.6, ls=(0, (2, 2)))
    top = y[ORDER[0]] + 0.95
    ax.text(0.503, top - 0.02, "chance", fontsize=SIZE["tiny"], color=C["muted"], va="top", ha="left")
    for r in ("RULE1", "RULE2"):
        ax.axhline(y[r], color=C["light"], lw=0.6)
    labels = {**LABEL, "BASE": "Qwen2.5-7B base\n(before post-training)", "MEAN": "mean of the nine"}
    ticks = [r for r in rows if not r.startswith("RULE")]
    ax.set_yticks([y[r] for r in ticks])
    ax.set_yticklabels([labels[r] for r in ticks])
    for t, r in zip(ax.get_yticklabels(), ticks):
        if r == "tsct":
            t.set_color(C["tsct_text"])
        if r == "MEAN":
            t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.34, 0.54)
    ax.set_ylim(y["MEAN"] - 0.45, top)
    ax.set_xticks([0.35, 0.40, 0.45, 0.50])
    ax.set_xlabel("staleness AUROC (below 0.5: expired value more familiar)", loc="right")
    handles = [Line2D([], [], color=C["ink"], marker="o", ms=3.4, lw=0.9, label="model at fixed 4-bit quantization, 95% CI"),
               Line2D([], [], ms=3.8, label="same prompt prefixed \u201cAs of 2026,\u201d (just above each row)",
                      **DIAMOND),
               Line2D([], [], color=C["ink"], marker="s", mfc="white", mew=0.9, ms=3.6, lw=0,
                      label="base model, before post-training")]
    fig.legend(handles=handles, loc="outside upper left", ncol=1, fontsize=SIZE["small"], handlelength=1.4,
               handletextpad=0.5, labelspacing=0.3, borderaxespad=0.1)
    print("\n".join(map(str, save(fig, "fig4_staleness"))))


if __name__ == "__main__":
    main()
