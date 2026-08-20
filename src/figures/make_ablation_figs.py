"""Figures for the TSCT ablation results. Real data only — nothing from paper_plots.py."""
import json, os, sys, glob
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

TK = "/private/tmp/claude-501/-Users-atae/77395a14-31c7-47ff-9160-46cffd570e2b/scratchpad/tsct_context/eval_toolkit"
sys.path.insert(0, TK)
from eval_pipeline import compute_ece, compute_accuracy_metrics, HEDGE_TO_CONFIDENCE
from hedge_quality_rubric import GOLD_HEDGE_FOR_VOLATILITY

D = "/Users/atae/Downloads/tsct_ablation_results/predictions/with_year"
FIG = "/Users/atae/Downloads/tsct_ablation_results/figures"
os.makedirs(FIG, exist_ok=True)

# ── assigned palette (teal sequential ramp) ───────────────────────────────────
# Ordinal 4-step selection validated: monotone L, all dL >= 0.06,
# light-end 2.66:1 vs white, hue spread 1deg.
T_DARK, T_MID, T_LMID, T_LIGHT = "#015d67", "#3b747c", "#618c93", "#85a4a9"
SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

# hedge tokens are ORDINAL (confidence 0.95 -> 0.10): more confidence = darker
HEDGE_ORDER = ["[CONFIDENT]", "[COND_CONFIDENT]", "[TEMPORAL_HEDGE]", "[UNKNOWN]"]
HEDGE_COLOR = dict(zip(HEDGE_ORDER, [T_DARK, T_MID, T_LMID, T_LIGHT]))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "savefig.facecolor": SURFACE, "savefig.bbox": "tight",
})

CONDS = [("SFT", "exp2_v2"), ("TSCT v6\n(full TCL)", "exp3_v6"),
         ("ablation:\nno_volatility", "exp3_ablNoVol"),
         ("ablation:\nno_contrastive", "exp3_ablNoCon"),
         ("ablation:\nfixed_deploy", "exp3_ablFixDep")]
load = lambda p: [json.loads(l) for l in open(glob.glob(
    os.path.join(D, f"{p}_seed42_temporal_delta_predictions.jsonl"))[0])]


def tidy(ax, xgrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS); ax.spines["bottom"].set_color(AXIS)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8)
    ax.tick_params(length=0)


# ── gather ────────────────────────────────────────────────────────────────────
data = {}
for lab, pfx in CONDS:
    R = load(pfx); n = len(R)
    h = Counter(r["predicted_hedge"] for r in R)
    conf = {c: np.mean([HEDGE_TO_CONFIDENCE.get(r["predicted_hedge"], .5)
                        for r in R if r.get("volatility") == c]) for c in ("fast", "slow")}
    f = Counter(r["predicted_hedge"] for r in R if r.get("volatility") == "fast")
    s = Counter(r["predicted_hedge"] for r in R if r.get("volatility") == "slow")
    nf, ns = sum(f.values()), sum(s.values())
    tv = 0.5 * sum(abs(f.get(k, 0)/nf - s.get(k, 0)/ns) for k in HEDGE_ORDER)
    data[lab] = dict(ece=compute_ece(R)["ece"], em=compute_accuracy_metrics(R)["em"],
                     hedge={k: 100*h.get(k, 0)/n for k in HEDGE_ORDER},
                     delta=conf["slow"]-conf["fast"], tv=tv, n=n)

base = load("exp2_v2")
ECE_ORACLE = compute_ece([{**r, "predicted_hedge": GOLD_HEDGE_FOR_VOLATILITY.get(
    r.get("volatility"), "[TEMPORAL_HEDGE]")} for r in base])["ece"]
ECE_UNK = compute_ece([{**r, "predicted_hedge": "[UNKNOWN]"} for r in base])["ece"]
gold = Counter(GOLD_HEDGE_FOR_VOLATILITY.get(r.get("volatility")) for r in base)
gn = sum(gold.values())

labels = [l for l, _ in CONDS]

# ── Fig 1: ECE with reference lines ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.4, 3.9))
y = np.arange(len(labels))[::-1]
vals = [data[l]["ece"] for l in labels]
ax.barh(y, vals, height=0.6, color=T_DARK, zorder=3)
for yi, v in zip(y, vals):
    ax.text(v + 0.012, yi, f"{v:.3f}", va="center", ha="left",
            fontsize=9, color=INK, fontweight="medium", zorder=4)
for x, txt, dash in ((ECE_ORACLE, f"oracle {ECE_ORACLE:.3f}", (5, 3)),
                     (ECE_UNK, f"constant [UNKNOWN] {ECE_UNK:.3f}", (2, 2))):
    ax.axvline(x, color=MUTED, linestyle=(0, dash), linewidth=1.4, zorder=2)
    ax.text(x + 0.012, 4.82, txt, va="center", ha="left",
            fontsize=8, color=INK2, zorder=4)
ax.set_ylim(-0.6, 5.15)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlim(0, 1.0); ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.set_xlabel("Expected calibration error (lower is better)")
ax.set_title("Refusing every question scores a better ECE than perfect hedging",
             loc="left", pad=26, fontweight="bold", color=INK)
ax.text(0, 1.055, "TemporalDelta cleaned test set, n=3,456, seed 42",
        transform=ax.transAxes, fontsize=8.5, color=INK2)
tidy(ax)
plt.savefig(f"{FIG}/fig1_ece_vs_refusal.png", dpi=200); plt.close()

# ── Fig 2: hedge distribution (ordinal ramp, stacked) ─────────────────────────
fig, ax = plt.subplots(figsize=(7.4, 4.3))
rows = labels + ["gold (from\nvolatility)"]
y = np.arange(len(rows))[::-1]
for i, row in enumerate(rows):
    dist = ({k: 100*gold.get(k, 0)/gn for k in HEDGE_ORDER} if i == len(labels)
            else data[row]["hedge"])
    left = 0.0
    for k in HEDGE_ORDER:
        w = dist.get(k, 0.0)
        if w <= 0:
            continue
        ax.barh(y[i], w, left=left, height=0.62, color=HEDGE_COLOR[k],
                edgecolor=SURFACE, linewidth=1.6, zorder=3)   # 2px surface gap
        if w >= 9:
            ax.text(left + w/2, y[i], f"{w:.0f}", va="center", ha="center",
                    fontsize=8.5, color=SURFACE if k in (T_DARK, T_MID) else INK, zorder=4)
        left += w
ax.set_yticks(y); ax.set_yticklabels(rows)
ax.set_xlim(0, 100); ax.set_xlabel("share of predictions (%)")
ax.xaxis.set_major_locator(MultipleLocator(25))
handles = [plt.Rectangle((0, 0), 1, 1, color=HEDGE_COLOR[k]) for k in HEDGE_ORDER]
ax.legend(handles, [k.strip("[]").replace("_", " ").title() + f"  ({HEDGE_TO_CONFIDENCE[k]:.2f})"
                    for k in HEDGE_ORDER],
          loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False,
          fontsize=8.5, labelcolor=INK2, handlelength=1.1, columnspacing=1.6)
ax.set_title("Every ablation collapses onto a single hedge token",
             loc="left", pad=22, fontweight="bold", color=INK)
ax.text(0, 1.045, "Darker = higher stated confidence. Bottom row is the correct answer.",
        transform=ax.transAxes, fontsize=8.5, color=INK2)
tidy(ax)
plt.savefig(f"{FIG}/fig2_hedge_distribution.png", dpi=200); plt.close()

# ── Fig 3: confidence direction (emphasis on the inverted case) ───────────────
fig, ax = plt.subplots(figsize=(7.4, 3.6))
y = np.arange(len(labels))[::-1]
deltas = [data[l]["delta"] for l in labels]
worst = int(np.argmin(deltas))
cols = [T_DARK if i == worst else MUTED for i in range(len(labels))]
ax.barh(y, deltas, height=0.6, color=cols, zorder=3)
for yi, v, i in zip(y, deltas, range(len(labels))):
    lbl = "0.000" if abs(v) < 5e-4 else f"{v:+.3f}"
    ax.text(v + (0.006 if v >= 0 else -0.006), yi, lbl, va="center",
            ha="left" if v >= 0 else "right", fontsize=9,
            color=INK if i == worst else INK2, zorder=4)
ax.axvline(0, color=AXIS, linewidth=1.2, zorder=2)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlim(-0.24, 0.12)
ax.set_xlabel("mean confidence on slow facts  −  mean confidence on fast facts")
ax.set_title("Full TCL is more confident about facts that change faster",
             loc="left", pad=26, fontweight="bold", color=INK)
ax.text(0, 1.055, "Slow-changing facts should receive MORE confidence, so correct is positive.",
        transform=ax.transAxes, fontsize=8.5, color=INK2)
tidy(ax)
plt.savefig(f"{FIG}/fig3_confidence_direction.png", dpi=200); plt.close()

# ── Fig 4: ECE vs discrimination ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 4.4))
xs = [data[l]["ece"] for l in labels]; ys = [data[l]["tv"] for l in labels]
ax.scatter(xs, ys, s=110, color=T_DARK, zorder=4,
           edgecolor=SURFACE, linewidth=2)
off = {"SFT": (10, -4), "TSCT v6\n(full TCL)": (10, -2),
       "ablation:\nno_volatility": (12, -2), "ablation:\nno_contrastive": (-12, 10),
       "ablation:\nfixed_deploy": (10, -14)}
for l, x, yv in zip(labels, xs, ys):
    dx = off.get(l, (10, 0))
    ax.annotate(l.replace("\n", " "), (x, yv), textcoords="offset points",
                xytext=dx, fontsize=8.5, color=INK2, va="center",
                ha="right" if dx[0] < 0 else "left")
ax.axvline(ECE_UNK, color=MUTED, linestyle=(0, (2, 2)), linewidth=1.4, zorder=2)
ax.text(ECE_UNK + 0.014, 0.44, "constant [UNKNOWN]", fontsize=8, color=INK2, rotation=90,
        va="top")
ax.set_xlabel("ECE (lower is better)")
ax.set_ylabel("volatility discrimination\nTV distance, fast vs slow  (higher is better)")
ax.set_xlim(0, 0.95); ax.set_ylim(-0.02, 0.45)
ax.set_title("No condition combines low ECE with volatility discrimination",
             loc="left", pad=26, fontweight="bold", color=INK)
ax.text(0, 1.055, "Ideal is the upper-left corner: low ECE and real discrimination. It is empty.",
        transform=ax.transAxes, fontsize=8.5, color=INK2)
tidy(ax); ax.grid(axis="y", color=GRID, linewidth=0.8)
plt.savefig(f"{FIG}/fig4_ece_vs_discrimination.png", dpi=200); plt.close()

print("wrote:")
for f in sorted(os.listdir(FIG)):
    print("  ", f, os.path.getsize(os.path.join(FIG, f)) // 1024, "KB")
print(f"\nreference ECE — oracle {ECE_ORACLE:.4f}, constant [UNKNOWN] {ECE_UNK:.4f}")
