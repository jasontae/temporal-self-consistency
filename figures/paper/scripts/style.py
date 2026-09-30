"""Shared style for every paper figure (ACL/ARR two-column template).

Import this module before creating a figure:

    from style import COL, FULL, C, apply, save

Sizes follow the ACL template: one column is 3.03 in wide, the full text
width is 7.0 in. Body text in the paper is 11 pt Times; figure text is 8 pt
(ticks and annotations 7 pt, never below 6 pt). Fonts are embedded as
TrueType (pdf.fonttype 42) so the PDFs pass ARR's font check.

Palette: Okabe & Ito (2008), safe for the common colour-vision deficiencies.
Colours are assigned by role, and every role keeps its colour in every figure.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "docs" / "orientation" / "results_sep10"
OUT = Path(__file__).resolve().parents[1]

COL = 3.03   # single column, inches
FULL = 7.0   # full text width, inches

# Okabe-Ito, by role
C = {
    "oracle": "#0072B2",     # blue: volatility-label (TVT) oracle
    "constant": "#D55E00",   # vermillion: constant policies
    "score": "#009E73",      # bluish green: informative continuous scores
    "tsct": "#CC79A7",       # reddish purple: TSCT (the team's method)
    "accent": "#E69F00",     # orange: a second condition (e.g. "As of 2026,")
    "sky": "#56B4E9",        # sky blue: a third condition / light fill
    "ink": "#222222",        # text and axes
    "muted": "#6E6E6E",      # secondary text, reference lines (4.9:1 on white)
    "light": "#BDBDBD",      # background points (e.g. hedge-trained adapters)
    "fill": "#F2F2F2",       # faint grid lines
    # darker shades for TEXT in a role colour (>= 4.5:1 on white and on the role's tinted band)
    "constant_text": "#B04A00",
    "tsct_text": "#9A4A76",
    "oracle_text": "#005A8C",
}

SIZE = {"base": 8, "small": 7, "tiny": 6.5}   # pt; 6 pt is the hard floor

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": SIZE["base"],
    "axes.titlesize": SIZE["base"],
    "axes.labelsize": SIZE["base"],
    "xtick.labelsize": SIZE["small"],
    "ytick.labelsize": SIZE["small"],
    "legend.fontsize": SIZE["small"],
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderaxespad": 0.3,
    "axes.linewidth": 0.6,
    "axes.edgecolor": C["ink"],
    "axes.labelcolor": C["ink"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "axes.titlepad": 3,
    "xtick.color": C["ink"],
    "ytick.color": C["ink"],
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "text.color": C["ink"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
    # exact page size: no tight-bbox re-cropping, so a 7.0 in figure stays 7.0 in
    "savefig.bbox": None,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.hspace": 0.04,
    "figure.constrained_layout.wspace": 0.04,
    "figure.dpi": 100,
}


def apply():
    plt.rcParams.update(RC)


def halo(width=2.0):
    """White outline for labels that sit over lines or error bars."""
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=width, foreground="white")]


def signed(v, fmt="+.2f"):
    """Format with a true minus sign (U+2212), as typeset text uses."""
    return format(v, fmt).replace("-", "\u2212")


def panel_label(ax, s, x=-0.02, y=1.0):
    """Bold panel letter at the top left, outside the axes."""
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=SIZE["base"],
            ha="right", va="bottom")


def panel_title(ax, letter, text, x=0.0, fontsize=None, pad=3):
    """Panel title that says what the panel shows: a bold "(A)" then plain text.

    Drawn above the axes starting at x (axes fraction; negative to start over
    the y label). text may hold "\\n" for a second line; the letter sits on
    the first line. The drawn title is kept out of constrained layout (a wide
    title would squeeze the axes sideways); an empty axes title with the same
    number of lines reserves the height instead.
    """
    fs = fontsize or SIZE["base"]
    n = text.count("\n") + 1
    slot = ax.set_title("\n".join(["(Ag)"] * n), loc="left", x=x, fontsize=fs, pad=pad, color="none")
    lead = ax.annotate(f"({letter})" + "\n " * (n - 1), xy=(0.0, 0.0), xycoords=slot, fontweight="bold",
                       fontsize=fs, ha="left", va="bottom", annotation_clip=False)
    body = ax.annotate(text, xy=(1.0, 0.0), xycoords=lead, xytext=(3, 0), textcoords="offset points",
                       fontsize=fs, ha="left", va="bottom", annotation_clip=False)
    lead.set_in_layout(False)
    body.set_in_layout(False)
    return lead


def audit(fig, name):
    """Print QA problems: text under 6 pt, overlapping text boxes, text past the page edge.

    Checks every visible, non-empty Text artist (tick labels included) at the
    figure's real size. Returns the list of problems.
    """
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # tick labels exist for ticks outside the view limits but are never drawn; keep only drawn ones
    all_ticks, drawn = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticks |= {id(tk.label1), id(tk.label2)}
            for tk in axis._update_ticks():
                drawn |= {id(tk.label1), id(tk.label2)}
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and t.get_color() != "none"
             and t.get_window_extent(r).width > 0 and (id(t) not in all_ticks or id(t) in drawn)]
    probs = []
    for t in texts:
        if t.get_fontsize() < 6:
            probs.append(f"{name}: text under 6 pt ({t.get_fontsize():.1f}): {t.get_text()!r}")
    boxes = [(t, t.get_window_extent(r).expanded(1.0, 1.0)) for t in texts]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (t1, b1), (t2, b2) = boxes[i], boxes[j]
            if b1.overlaps(b2):
                ov = (min(b1.x1, b2.x1) - max(b1.x0, b2.x0)) * (min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
                if ov > 4:   # px^2 at 100 dpi; ignore touching corners
                    probs.append(f"{name}: overlap {t1.get_text()!r} / {t2.get_text()!r}")
    fb = fig.bbox
    for t in texts:
        b = t.get_window_extent(r)
        if b.x0 < fb.x0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y0 < fb.y0 - 0.5 or b.y1 > fb.y1 + 0.5:
            probs.append(f"{name}: text clipped by the page edge: {t.get_text()!r}")
    w, h = fig.get_size_inches()
    print(f"QA {name}: page {w:.2f} x {h:.2f} in, {len(texts)} text items, {len(probs)} problems")
    for p in probs:
        print("QA", p)
    return probs


def save(fig, name, svg=False):
    """Write <name>.pdf and a 300-dpi <name>.png (and <name>.svg if asked) into figures/paper/."""
    audit(fig, name)
    meta = {"Creator": "figures/paper/scripts (matplotlib)"}
    fig.savefig(OUT / f"{name}.pdf", metadata={**meta, "CreationDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=300)
    if svg:
        fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)
    return [OUT / f"{name}.{e}" for e in (("pdf", "png", "svg") if svg else ("pdf", "png"))]


def place_labels(ax, pts, names, colors=None, fontsize=None, avoid=(), soft=(), merge_pt=6.0, radii=(4, 7, 11),
                 keep_out=(), halo_width=2.0, fixed=None):
    """Label scatter points with short text beside them, chosen to avoid clutter.

    Greedy placement in display space: points that sit within merge_pt points of
    each other share one stacked label; each label then takes the candidate
    offset (8 directions x radii, in points) with the lowest cost, where cost
    counts overlap with already placed labels, with any marker, with the axes
    edge and with keep_out boxes (all heavy), crossings of the `avoid` curves
    (heavy) and of the `soft` curves such as CI bars (light), plus distance.
    `avoid`/`soft` are lists of (xs, ys) arrays in data coordinates.
    fixed maps a name to (dx, dy, ha, va) in points: those labels are placed
    first, exactly there, and the rest avoid them. Returns the Text artists.
    """
    import numpy as np
    fs = fontsize or SIZE["tiny"]
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ppt = fig.dpi / 72.0
    to_disp = ax.transData.transform
    P = to_disp(np.asarray(pts, float))
    colors = colors or [C["ink"]] * len(pts)
    # merge near-coincident points
    groups, used = [], set()
    for i in range(len(P)):
        if i in used:
            continue
        g = [i] + [j for j in range(i + 1, len(P)) if j not in used and np.hypot(*(P[i] - P[j])) < merge_pt * ppt]
        used |= set(g)
        groups.append(g)

    def sample(curves, n=400):
        out = []
        for xs, ys in curves:
            xs, ys = np.asarray(xs, float), np.asarray(ys, float)
            t = np.linspace(0, 1, n)
            seg = np.interp(t * (len(xs) - 1), np.arange(len(xs)), xs), np.interp(t * (len(ys) - 1), np.arange(len(ys)), ys)
            out.append(to_disp(np.column_stack(seg)))
        return np.vstack(out) if out else np.zeros((0, 2))

    hard, light = sample(avoid), sample(soft)
    ab = ax.get_window_extent(r)
    mr = 2.5 * ppt   # marker half-size plus a little air
    marks = [(p[0] - mr, p[1] - mr, p[0] + mr, p[1] + mr) for p in P]
    keep = [tuple(to_disp([[b[0], b[1]]])[0]) + tuple(to_disp([[b[2], b[3]]])[0]) for b in keep_out]
    placed, texts = [], []

    def ov(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    dirs = [(1, 0, "left", "center"), (-1, 0, "right", "center"), (0.7, 0.7, "left", "bottom"),
            (-0.7, 0.7, "right", "bottom"), (0.7, -0.7, "left", "top"), (-0.7, -0.7, "right", "top"),
            (0, 1, "center", "bottom"), (0, -1, "center", "top")]
    fixed = fixed or {}
    # fixed labels first, then the most crowded groups
    crowd = [sum(np.hypot(*(P[g[0]] - q)) < 40 * ppt for q in P) + 1000 * (names[g[0]] in fixed) for g in groups]
    for gi in np.argsort(crowd, kind="stable")[::-1]:
        g = groups[gi]
        c = P[g].mean(axis=0)
        label = "\n".join(names[i] for i in g)
        best = None
        fx = fixed.get(names[g[0]])
        cands = [(fx[0], fx[1], fx[2], fx[3], 1)] if fx else \
            [(dx * rad, dy * rad, ha, va, rad) for rad in radii for dx, dy, ha, va in dirs]
        for dx, dy, ha, va, rad in cands:
            t = ax.text(0, 0, label, fontsize=fs, ha=ha, va=va, color=colors[g[0]], transform=None,
                        linespacing=1.0)
            t.set_position((c[0] + dx * ppt, c[1] + dy * ppt))
            bb = t.get_window_extent(r)
            t.remove()
            box = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
            cost = rad * 2.0
            gap = 3 * ppt   # keep separate labels visibly apart, so two never read as one stacked label
            cost += 50 * sum(ov(box, (p[0] - gap, p[1] - gap, p[2] + gap, p[3] + gap)) for p in placed)
            cost += 50 * sum(ov(box, m) for k, m in enumerate(marks))
            cost += 50 * sum(ov(box, k) for k in keep)
            inside = ov(box, (ab.x0, ab.y0, ab.x1, ab.y1))
            cost += 50 * ((box[2] - box[0]) * (box[3] - box[1]) - inside)
            if len(hard):
                cost += 40 * np.sum((hard[:, 0] > box[0]) & (hard[:, 0] < box[2]) & (hard[:, 1] > box[1]) & (hard[:, 1] < box[3]))
            if len(light):
                cost += 2 * np.sum((light[:, 0] > box[0]) & (light[:, 0] < box[2]) & (light[:, 1] > box[1]) & (light[:, 1] < box[3]))
            # a label must sit nearer its own point than any other point (no ambiguity)
            def bdist(q):
                return np.hypot(max(box[0] - q[0], 0, q[0] - box[2]), max(box[1] - q[1], 0, q[1] - box[3]))
            own = min(bdist(P[i]) for i in g)
            others = [bdist(P[j]) for j in range(len(P)) if j not in g]
            cost += 300 * sum(o < own for o in others) + 40 * sum(own <= o < own + 2 * ppt for o in others)
            if best is None or cost < best[0]:
                best = (cost, (dx, dy), ha, va, box)
        _, (ox, oy), ha, va, box = best
        placed.append(box)
        xy = ax.transData.inverted().transform(c)
        texts.append(ax.annotate(label, xy, xytext=(ox, oy), textcoords="offset points", ha=ha, va=va, fontsize=fs,
                                 color=colors[g[0]], linespacing=1.0, path_effects=halo(halo_width), zorder=6))
    return texts
