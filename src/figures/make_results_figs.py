"""Results-section figures. Real predictions only."""
import json, os, sys, glob
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

TK = "/private/tmp/claude-501/-Users-atae/77395a14-31c7-47ff-9160-46cffd570e2b/scratchpad/tsct_context/eval_toolkit"
sys.path.insert(0, TK)
from eval_pipeline import compute_ece, compute_accuracy_metrics, HEDGE_TO_CONFIDENCE
from hedge_quality_rubric import GOLD_HEDGE_FOR_VOLATILITY

V6 = "/Users/atae/Downloads/david_v6/predictions_v6"
ABL = "/Users/atae/Downloads/tsct_ablation_results/predictions/with_year"
FIG = "/Users/atae/Downloads/tsct_ablation_results/figures_results"
os.makedirs(FIG, exist_ok=True)

T_DARK, T_MID, T_LMID, T_LIGHT = "#015d67", "#3b747c", "#618c93", "#85a4a9"
SURFACE, INK, INK2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
HED = ["[CONFIDENT]", "[COND_CONFIDENT]", "[TEMPORAL_HEDGE]", "[UNKNOWN]"]
HCOL = dict(zip(HED, [T_DARK, T_MID, T_LMID, T_LIGHT]))

FRAME = "#333333"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Verdana", "Arial"],
    "font.size": 11, "axes.titlesize": 15, "axes.labelsize": 12,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": FRAME, "axes.linewidth": 1.0,
    "text.color": INK, "axes.labelcolor": "#333333",
    "xtick.color": FRAME, "ytick.color": FRAME,
    "xtick.labelcolor": "#333333", "ytick.labelcolor": "#333333",
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10.5, "legend.title_fontsize": 11.5,
    "grid.linestyle": "--", "grid.linewidth": 0.7, "grid.color": "#cfcfcf",
    "savefig.facecolor": SURFACE, "savefig.bbox": "tight",
})
BENCH = ["temporal_delta", "mmlu", "freshqa", "stress_test"]
BLAB = {"temporal_delta": "TemporalDelta", "mmlu": "MMLU (stable)",
        "freshqa": "FreshQA", "stress_test": "Extended horizon"}
A, B = [42, 123, 456], [501, 502, 503, 504, 505]


def load(cond, seed, bench, root=V6):
    p = glob.glob(os.path.join(root, f"{'exp2' if cond=='sft' else 'exp3'}_*_seed{seed}_{bench}_predictions.jsonl"))
    return [json.loads(l) for l in open(p[0])] if p else []


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    o = np.argsort(s); sv = s[o]; ranks = np.empty(len(s)); r = np.arange(1, len(s) + 1, dtype=float); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]:
            j += 1
        ranks[o[i:j + 1]] = r[i:j + 1].mean(); i = j + 1
    npos, nneg = y.sum(), (~y).sum()
    return (ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def tidy(ax, axis="x"):
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(True); ax.spines[sp].set_color(FRAME)
    ax.set_axisbelow(True)
    ax.grid(axis=axis, linestyle="--", linewidth=0.7, color="#cfcfcf")
    ax.tick_params(length=3, color=FRAME)


# ══ Figure 1 — per-benchmark ECE, SFT vs TSCT, split by seed batch ═══════════
fig, axes = plt.subplots(1, 4, figsize=(13.0, 4.2), sharey=True)
for ax, b in zip(axes, BENCH):
    for xi, cond in enumerate(("sft", "tsct")):
        eA = np.mean([compute_ece(load(cond, s, b))["ece"] for s in A])
        eB = np.mean([compute_ece(load(cond, s, b))["ece"] for s in B])
        ax.plot([xi, xi], [eA, eB], color=MUTED, linewidth=1.4, zorder=2)
        ax.scatter([xi], [eA], s=78, color=T_DARK, zorder=4, edgecolor=SURFACE, linewidth=1.6)
        ax.scatter([xi], [eB], s=78, facecolor=SURFACE, edgecolor=T_DARK,
                   linewidth=1.8, zorder=4)
    ax.set_xlim(-0.55, 1.55); ax.set_xticks([0, 1]); ax.set_xticklabels(["SFT", "TSCT"])
    ax.set_title(BLAB[b], color=INK, fontsize=12.5, pad=8)
    tidy(ax, "y")
axes[0].set_ylabel("ECE"); axes[0].set_ylim(0, 1.0)
axes[0].yaxis.set_major_locator(MultipleLocator(0.2))
axes[-1].legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=9,
                                markerfacecolor=T_DARK, markeredgecolor=SURFACE,
                                label="42 / 123 / 456"),
                         Line2D([], [], marker="o", linestyle="", markersize=9,
                                markerfacecolor=SURFACE, markeredgecolor=T_DARK,
                                label="501–505")],
                loc="center left", bbox_to_anchor=(1.06, 0.5), frameon=False,
                title="Seed batch", alignment="left")
fig.suptitle("The two seed batches disagree in direction on every benchmark",
             fontsize=15, y=1.03, color=INK)
plt.savefig(f"{FIG}/results_fig1_seed_split.png", dpi=200); plt.close()

# ══ Figure 2 — hedge distribution ════════════════════════════════════════════
rows = [("SFT  (seeds 42/123/456)", [load("sft", s, "temporal_delta") for s in A]),
        ("TSCT (seeds 42/123/456)", [load("tsct", s, "temporal_delta") for s in A]),
        ("SFT  (seeds 501–505)", [load("sft", s, "temporal_delta") for s in B]),
        ("TSCT (seeds 501–505)", [load("tsct", s, "temporal_delta") for s in B])]
abl = [("ablation: no_volatility", "exp3_ablNoVol"), ("ablation: no_contrastive", "exp3_ablNoCon"),
       ("ablation: fixed_deploy", "exp3_ablFixDep")]
for lab, pfx in abl:
    f = glob.glob(os.path.join(ABL, f"{pfx}_seed42_temporal_delta_predictions.jsonl"))[0]
    rows.append((lab + "  (seed 42)", [[json.loads(l) for l in open(f)]]))
gold_R = load("sft", 42, "temporal_delta")
gold = Counter(GOLD_HEDGE_FOR_VOLATILITY.get(r.get("volatility")) for r in gold_R)
gn = sum(gold.values())
rows.append(("gold (from volatility)", None))

fig, ax = plt.subplots(figsize=(11.0, 5.4))
y = np.arange(len(rows))[::-1]
for i, (lab, groups) in enumerate(rows):
    if groups is None:
        dist = {k: 100 * gold.get(k, 0) / gn for k in HED}
    else:
        c = Counter()
        for g in groups:
            c.update(r["predicted_hedge"] for r in g)
        tot = sum(c.values())
        dist = {k: 100 * c.get(k, 0) / tot for k in HED}
    left = 0.0
    for k in HED:
        w = dist.get(k, 0.0)
        if w <= 0:
            continue
        ax.barh(y[i], w, left=left, height=0.62, color=HCOL[k], edgecolor=SURFACE,
                linewidth=1.6, zorder=3)
        if w >= 9:
            ax.text(left + w / 2, y[i], f"{w:.0f}", va="center", ha="center", fontsize=10.5,
                    color=SURFACE if k in (T_DARK, T_MID) else INK, zorder=4)
        left += w
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=11)
ax.set_xlim(0, 100); ax.set_xlabel("share of predictions (%)")
ax.xaxis.set_major_locator(MultipleLocator(25))
ax.axhline(y[-1] + 0.5, color=AXIS, linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=HCOL[k]) for k in HED],
          [k.strip("[]").replace("_", " ").title() + f"  ({HEDGE_TO_CONFIDENCE[k]:.2f})" for k in HED],
          loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
          title="Hedge token", alignment="left", handlelength=1.2)
ax.set_title("Hedge distributions diverge sharply, and several conditions collapse",
             pad=12, color=INK)
tidy(ax)
plt.savefig(f"{FIG}/results_fig2_hedge_distributions.png", dpi=200); plt.close()

# ══ Figure 3 — ECE is minimised by refusal at low accuracy ═══════════════════
accs = np.linspace(0.01, 0.99, 99)
DASH = {"[CONFIDENT]": (0, ()), "[COND_CONFIDENT]": (0, (7, 2.2)),
        "[TEMPORAL_HEDGE]": (0, (5, 2, 1.2, 2)), "[UNKNOWN]": (0, (1.6, 2.0))}
fig, ax = plt.subplots(figsize=(9.4, 5.0))
for k in HED:
    c = HEDGE_TO_CONFIDENCE[k]
    ax.plot(accs, np.abs(accs - c), color=HCOL[k], linewidth=2.7,
            linestyle=DASH[k], solid_capstyle="round", dash_capstyle="round",
            zorder=3, label=f"always {k.strip('[]').replace('_',' ').title()} ({c:.2f})")
# Derive the reference points from ONE source so the diamonds sit at the
# accuracy their ECE was actually computed on.
_ref = load("sft", 42, "temporal_delta")
ACC_REF = float(np.mean([bool(r.get("correct")) for r in _ref]))
ECE_ORACLE = compute_ece([{**r, "predicted_hedge": GOLD_HEDGE_FOR_VOLATILITY.get(
    r.get("volatility"), "[TEMPORAL_HEDGE]")} for r in _ref])["ece"]
ECE_UNK = compute_ece([{**r, "predicted_hedge": "[UNKNOWN]"} for r in _ref])["ece"]
obs = {}
for _b in ("temporal_delta", "mmlu", "freshqa"):
    _all = [r for _s in A + B for r in load("sft", _s, _b)]
    obs[_b] = float(np.mean([bool(r.get("correct")) for r in _all]))
lo, hi = min(obs.values()), max(obs.values())
ax.axvspan(lo, hi, color=T_LIGHT, alpha=0.18, zorder=1)
ax.text((lo + hi) / 2, 0.24, f"observed\naccuracy\n({100*lo:.0f}–{100*hi:.0f}%)", ha="center", va="center",
        fontsize=10, color="#333333",
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2, alpha=0.9))
ax.scatter([ACC_REF], [ECE_ORACLE], s=80, color=INK, zorder=5, marker="D")
ax.annotate(f"gold-hedge oracle  {ECE_ORACLE:.3f}", (ACC_REF, ECE_ORACLE), textcoords="offset points",
            xytext=(14, 10), fontsize=10.5, color=INK,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2, alpha=0.9))
ax.scatter([ACC_REF], [ECE_UNK], s=80, color=INK, zorder=5, marker="D")
ax.annotate(f"constant [UNKNOWN]  {ECE_UNK:.3f}", (ACC_REF, ECE_UNK), textcoords="offset points",
            xytext=(14, -12), fontsize=10.5, color=INK,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2, alpha=0.9))
ax.set_xlabel("true answer accuracy"); ax.set_ylabel("ECE")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
          title="Constant strategy", alignment="left", handlelength=3.0)
ax.set_title("Below ~28% accuracy, refusing everything is the ECE-optimal policy",
             pad=12, color=INK)
tidy(ax, "y")
plt.savefig(f"{FIG}/results_fig3_metric_degeneracy.png", dpi=200); plt.close()

# ══ Figure 4 — informativeness ═══════════════════════════════════════════════
conds = []
for lab, seeds in (("SFT (42/123/456)", A), ("TSCT (42/123/456)", A),
                   ("SFT (501–505)", B), ("TSCT (501–505)", B)):
    c = "sft" if lab.startswith("SFT") else "tsct"
    R = [r for s in seeds for r in load(c, s, "temporal_delta")]
    conds.append((lab, auroc([HEDGE_TO_CONFIDENCE.get(r["predicted_hedge"], .5) for r in R],
                             [bool(r.get("correct")) for r in R])))
for lab, pfx in abl:
    f = glob.glob(os.path.join(ABL, f"{pfx}_seed42_temporal_delta_predictions.jsonl"))[0]
    R = [json.loads(l) for l in open(f)]
    conds.append((lab.replace("ablation: ", "abl ") + " (42)",
                  auroc([HEDGE_TO_CONFIDENCE.get(r["predicted_hedge"], .5) for r in R],
                        [bool(r.get("correct")) for r in R])))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.8),
                               gridspec_kw={"width_ratios": [1.55, 1]})
y = np.arange(len(conds))[::-1]
ax1.barh(y, [c[1] - 0.5 for c in conds], left=0.5, height=0.6, color=T_DARK, zorder=3)
for yi, (lab, v) in zip(y, conds):
    ax1.text(v + (0.004 if v >= 0.5 else -0.004), yi, f"{v:.3f}", va="center",
             ha="left" if v >= 0.5 else "right", fontsize=10.5, color=INK, zorder=4)
ax1.axvline(0.5, color=FRAME, linewidth=1.2, zorder=2)
ax1.text(0.5, -0.72, "  0.50 = chance", fontsize=10.5, color="#333333", va="center")
ax1.set_yticks(y); ax1.set_yticklabels([c[0] for c in conds], fontsize=11)
ax1.set_xlim(0.40, 0.60); ax1.set_ylim(-1.05, len(conds) - 0.45)
ax1.set_xlabel("AUROC: does the hedge predict whether the answer is correct?")
ax1.set_title("Hedge tokens barely beat chance at predicting correctness",
              pad=12, color=INK, fontsize=13.5)
ax1.text(0.0, 1.20, "(a)", transform=ax1.transAxes, ha="left", va="bottom",
         fontsize=13.5, fontweight="bold", color=INK)
tidy(ax1)

pairs = []
for lab, seeds in (("SFT", A + B), ("TSCT", A + B)):
    c = lab.lower()
    td = [r for s in seeds for r in load(c, s, "temporal_delta")]
    mm = [r for s in seeds for r in load(c, s, "mmlu")]
    pairs.append((lab, np.mean([HEDGE_TO_CONFIDENCE.get(r["predicted_hedge"], .5) for r in mm]),
                  np.mean([HEDGE_TO_CONFIDENCE.get(r["predicted_hedge"], .5) for r in td])))
ENT = {"SFT": MUTED, "TSCT": T_DARK}          # colour follows the entity, not the result
for i, (lab, cs, cv) in enumerate(pairs):
    col = ENT[lab]
    ax2.plot([0, 1], [cv, cs], color=col, linewidth=2.2, zorder=3)
    ax2.scatter([0, 1], [cv, cs], s=70, color=col, zorder=4,
                edgecolor=SURFACE, linewidth=1.6)
    ax2.annotate(f"{lab}  ({cs-cv:+.3f})", (1, cs), textcoords="offset points",
                 xytext=(10, 0), fontsize=10.5, color=INK, va="center")
ax2.legend(handles=[Line2D([], [], color=ENT[k], linewidth=2.4, label=k) for k in ("SFT", "TSCT")],
           loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
           title="Condition", alignment="left")
ax2.set_xlim(-0.15, 1.85); ax2.set_xticks([0, 1])
ax2.set_xticklabels(["volatile\n(TemporalDelta)", "stable\n(MMLU)"])
ax2.set_ylabel("mean stated confidence")
ax2.set_title("Stable facts should get MORE confidence", pad=12,
              color=INK, fontsize=13.5)
ax2.text(0.0, 1.20, "(b)", transform=ax2.transAxes, ha="left", va="bottom",
         fontsize=13.5, fontweight="bold", color=INK)
tidy(ax2, "y")
plt.savefig(f"{FIG}/results_fig4_informativeness.png", dpi=200); plt.close()

print("wrote:")
for f in sorted(os.listdir(FIG)):
    print("  ", f)
print("\nAUROC values:")
for lab, v in conds:
    print(f"  {lab:<28}{v:.4f}")
print("\nstable/volatile confidence:")
for lab, cs, cv in pairs:
    print(f"  {lab:<6} stable {cs:.3f}  volatile {cv:.3f}  gap {cs-cv:+.3f}")
