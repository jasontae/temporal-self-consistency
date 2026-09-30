"""Item 7: the inverse truth-discrimination result, with the statistics requested
in review (Aug 27 A4, Sep 10 item 7).

Claim under test: across admitted models, the better a model separates true
from false founders (truth AUROC), the weaker its apparent regime separation
(immutable_true vs volatile_current AUROC).

Inputs: data/prep/predictions_7b/<prefix><name>.jsonl for the ten admitted
models (and the two excluded ones, plotted but never fitted). Scores are `pmi`.

Analyses:
  1. Per-model AUROCs (truth, regime, staleness) with entity-cluster bootstrap
     CIs. The same entity resample is applied to every model, since all models
     score the same 106 entities.
  2. Model-level fits (n = admitted models; 9 by default): OLS slope with (a) classical SE, (b) CR1 SE
     clustered by family (only 3 clusters, so shown for comparison with the
     Sep 2 text, not relied on); Pearson, Spearman; an exact permutation p over
     all 10! orderings; leave-one-model-out and leave-one-family-out slopes;
     entity-bootstrap CI for the slope.
  3. Split-half control. Truth and regime both use the immutable_true arm, so
     item-level noise in that arm is shared. Truth is computed on a random half
     of the entities and regime on the other half, 2,000 times.
  4. Crossed random-effects model (REML) on per-entity-per-model outcomes:
     y_em = b0 + b1 * truth_m + u_entity + v_model + w_family + e,
     y_em = 1[pmi(immutable_true) > pmi(volatile_current)] (paired win), and,
     as a check, the pmi difference standardized within model. b1 is tested
     with t on n_models - 2 df, because the predictor varies only between
     models.

Run from the repo root:
    python3 docs/orientation/results_sep10/inverse/inverse_truth.py [--prefix matched_]
"""
import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[4]
PRED = ROOT / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent
SEED, N_BOOT, N_SPLIT = 20260928, 2000, 2000

# name -> (display label, family, quantization as loaded)
ADMITTED = {
    "g3_4b": ("gemma-3-4B", "gemma", "4-bit"),
    "qwen3_4b_th": ("Qwen3-4B-Thinking", "qwen", "8-bit"),
    "tsct": ("trained model (Qwen2.5-7B + TSCT)", "qwen", "4-bit"),
    "qwen3_4b_it": ("Qwen3-4B-Instruct", "qwen", "8-bit"),
    "qwen25_7b": ("Qwen2.5-7B", "qwen", "4-bit"),
    "gptoss_20b": ("gpt-oss-20B", "gpt-oss", "6.5-bit"),
    "qwen36_35b": ("Qwen3.6-35B-A3B", "qwen", "nvfp4"),
    "qwen35_27b": ("Qwen3.5-27B", "qwen", "4-bit"),
    "qwen36_27b_stock": ("Qwen3.6-27B", "qwen", "4-bit"),
}
# Edward, 2026-09-28: all Gemma-4 rows dropped (log-probs anomalous under the
# harness, results_sep10/MODELS.md finding 1). E4B passed the truth bar and is
# re-admitted only by --with-e4b, for the appendix sensitivity table.
GEMMA4_E4B = {"g4_e4b": ("gemma-4-E4B", "gemma", "4-bit")}
EXCLUDED = {"g4_31b": ("gemma-4-31B", "gemma", "8-bit"),
            "gemma4_26b": ("gemma-4-26B-A4B", "gemma", "4-bit")}
# Item 6: the third 27B vintage point, added only by --with-qwen38.
QWEN38 = {"qwen38_27b": ("Qwen3.8-27B", "qwen", "4-bit")}
# label offsets (points) so the scatter labels do not collide
LABEL_OFFSET = {"g3_4b": (-58, 6), "g4_e4b": (5, -10), "qwen3_4b_th": (-40, 9), "tsct": (-50, 8),
                "qwen25_7b": (5, 5), "qwen3_4b_it": (-70, -12), "gptoss_20b": (5, 5),
                "qwen36_35b": (5, -11), "qwen35_27b": (5, 5), "qwen36_27b_stock": (5, -2)}
ARMS = ("immutable_true", "immutable_false", "volatile_current", "volatile_stale")


def load(path):
    by = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        by[r["entity_id"]][r["arm"]] = r["pmi"]
    return {e: a for e, a in by.items() if len(a) == 4}


def auroc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), all pairs (Mann-Whitney)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def contrasts(S, idx=None):
    """S: (n_entities, 4) pmi matrix in ARMS order. Returns truth, regime, staleness."""
    if idx is not None:
        S = S[idx]
    T, F, V, St = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    return auroc(T, F), auroc(T, V), auroc(V, St)


def ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    n, k = X.shape
    s2 = res @ res / (n - k)
    XtXi = np.linalg.inv(X.T @ X)
    return b, XtXi, res, s2


def cr1_se(x, y, groups):
    b, XtXi, res, _ = ols(x, y)
    X = np.column_stack([np.ones_like(x), x])
    meat = np.zeros((2, 2))
    G = sorted(set(groups))
    for g in G:
        m = np.array([gg == g for gg in groups])
        s = X[m].T @ res[m]
        meat += np.outer(s, s)
    n, k, nG = len(y), 2, len(G)
    V = XtXi @ meat @ XtXi * (nG / (nG - 1)) * ((n - 1) / (n - k))
    return b[1], math.sqrt(V[1, 1]), nG


def exact_perm_p(x, y, n_mc=500000, seed=SEED):
    """Two-sided permutation p for the Pearson slope: exact over all n! orderings
    for n <= 10, Monte Carlo (n_mc draws) above that."""
    obs = abs(np.corrcoef(x, y)[0, 1])
    xs = x - x.mean()
    ys = y - y.mean()
    if len(y) <= 10:
        perms = np.array(list(itertools.permutations(range(len(y)))), dtype=np.int8)
    else:
        rng = np.random.default_rng(seed)
        perms = np.argsort(rng.random((n_mc, len(y))), axis=1)
    num = np.abs((ys[perms] * xs).sum(1))
    denom = math.sqrt((xs ** 2).sum() * (ys ** 2).sum())
    return float((num / denom >= obs - 1e-12).mean())


def reml_crossed(y, x, groups):
    """REML fit of y = b0 + b1 x + sum of random intercepts + e.
    `groups` is a list of integer-coded arrays (one per random factor)."""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    Zs = []
    for g in groups:
        Z = np.zeros((n, g.max() + 1))
        Z[np.arange(n), g] = 1.0
        Zs.append(Z @ Z.T)

    def build(theta):
        v = np.exp(theta)
        V = v[-1] * np.eye(n)
        for s, ZZ in zip(v[:-1], Zs):
            V = V + s * ZZ
        return V

    def negll(theta):
        V = build(theta)
        try:
            L = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return 1e12
        Vi_X = np.linalg.solve(L.T, np.linalg.solve(L, X))
        Vi_y = np.linalg.solve(L.T, np.linalg.solve(L, y))
        XtViX = X.T @ Vi_X
        b = np.linalg.solve(XtViX, X.T @ Vi_y)
        r = y - X @ b
        Vi_r = np.linalg.solve(L.T, np.linalg.solve(L, r))
        sign, logdet_x = np.linalg.slogdet(XtViX)
        return 0.5 * (2 * np.log(np.diag(L)).sum() + logdet_x + r @ Vi_r)

    theta0 = np.full(len(groups) + 1, np.log(np.var(y) / (len(groups) + 1)))
    fit = optimize.minimize(negll, theta0, method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
    V = build(fit.x)
    Vi = np.linalg.inv(V)
    cov = np.linalg.inv(X.T @ Vi @ X)
    b = cov @ X.T @ Vi @ y
    return {"b0": float(b[0]), "b1": float(b[1]), "se_b1": float(math.sqrt(cov[1, 1])),
            "variances": [float(v) for v in np.exp(fit.x)], "converged": bool(fit.success)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="matched_")
    ap.add_argument("--with-e4b", action="store_true", help="sensitivity: re-admit gemma-4-E4B")
    ap.add_argument("--with-qwen38", action="store_true", help="add Qwen3.8-27B (item 6)")
    ap.add_argument("--common-with", default=None,
                    help="keep only models that also have a <this prefix> file (same-model comparison)")
    args = ap.parse_args()
    tag = (args.prefix.rstrip("_") + ("_with_e4b" if args.with_e4b else "")
           + ("_with_qwen38" if args.with_qwen38 else "")
           + (f"_common_{args.common_with.rstrip('_')}" if args.common_with else ""))
    if args.with_e4b:
        ADMITTED.update(GEMMA4_E4B)
    else:
        EXCLUDED.update(GEMMA4_E4B)
    if args.with_qwen38:
        ADMITTED.update(QWEN38)

    data = {}
    for name in list(ADMITTED) + list(EXCLUDED):
        p = PRED / f"{args.prefix}{name}.jsonl"
        if args.common_with and not (PRED / f"{args.common_with}{name}.jsonl").exists():
            continue
        if p.exists():
            data[name] = load(p)
    adm = [n for n in ADMITTED if n in data]
    ents = sorted(set.intersection(*(set(data[n]) for n in data)))
    S = {n: np.array([[data[n][e][a] for a in ARMS] for e in ents]) for n in data}
    nE = len(ents)

    # 1. per-model AUROCs and entity-bootstrap CIs (shared resample)
    rng = np.random.default_rng(SEED)
    boots = rng.integers(0, nE, (N_BOOT, nE))
    point = {n: contrasts(S[n]) for n in data}
    bt = {n: np.array([contrasts(S[n], b) for b in boots]) for n in data}
    paired = {n: float((S[n][:, 0] > S[n][:, 2]).mean()) for n in data}
    sign_p = {}
    for n in data:
        d = S[n][:, 0] - S[n][:, 2]
        d = d[d != 0]
        sign_p[n] = float(stats.binomtest(int((d > 0).sum()), len(d), 0.5).pvalue)

    # 2. model-level fits on the admitted ten
    x = np.array([point[n][0] for n in adm])
    y = np.array([point[n][1] for n in adm])
    fam = [ADMITTED[n][1] for n in adm]
    b, XtXi, res, s2 = ols(x, y)
    se_cls = math.sqrt(s2 * XtXi[1, 1])
    t_cls = b[1] / se_cls
    p_cls = 2 * stats.t.sf(abs(t_cls), len(y) - 2)
    b1_cr, se_cr, nG = cr1_se(x, y, fam)
    p_cr_t = 2 * stats.t.sf(abs(b1_cr / se_cr), nG - 1)
    r_p, p_pear = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    p_perm = exact_perm_p(x, y)
    loo = {ADMITTED[adm[i]][0]: float(ols(np.delete(x, i), np.delete(y, i))[0][1]) for i in range(len(adm))}
    lofo = {}
    for f in sorted(set(fam)):
        keep = np.array([ff != f for ff in fam])
        if keep.sum() >= 3:
            lofo[f"without {f}"] = {"slope": float(ols(x[keep], y[keep])[0][1]), "n": int(keep.sum()),
                                    "spearman": float(stats.spearmanr(x[keep], y[keep])[0])}
    slope_boot = np.array([ols(np.array([bt[n][i, 0] for n in adm]),
                               np.array([bt[n][i, 1] for n in adm]))[0][1] for i in range(N_BOOT)])

    # 3. split-half control
    split_slopes, split_rho = [], []
    for _ in range(N_SPLIT):
        perm = rng.permutation(nE)
        A, B = perm[: nE // 2], perm[nE // 2:]
        xs = np.array([contrasts(S[n], A)[0] for n in adm])
        ys = np.array([contrasts(S[n], B)[1] for n in adm])
        split_slopes.append(ols(xs, ys)[0][1])
        split_rho.append(stats.spearmanr(xs, ys)[0])
    split_slopes, split_rho = np.array(split_slopes), np.array(split_rho)

    # 4. crossed random-effects model
    ent_code = np.tile(np.arange(nE), len(adm))
    mod_code = np.repeat(np.arange(len(adm)), nE)
    fams = sorted(set(fam))
    fam_code = np.repeat([fams.index(f) for f in fam], nE)
    xm = np.repeat(x - x.mean(), nE)
    win = np.concatenate([(S[n][:, 0] > S[n][:, 2]).astype(float) for n in adm])
    dz = np.concatenate([(S[n][:, 0] - S[n][:, 2]) / S[n].std() for n in adm])
    lmm = {}
    for label, yy in (("paired win 1[T > V]", win), ("pmi difference, z within model", dz)):
        fit = reml_crossed(yy, xm, [ent_code, mod_code, fam_code])
        t = fit["b1"] / fit["se_b1"]
        fit["t"] = float(t)
        fit["df"] = len(adm) - 2
        fit["p_t"] = float(2 * stats.t.sf(abs(t), len(adm) - 2))
        fit["variance_names"] = ["entity", "model", "family", "residual"]
        lmm[label] = fit

    out = {
        "prefix": args.prefix, "n_entities": nE, "admitted": adm,
        "per_model": {n: {"label": (ADMITTED | EXCLUDED)[n][0], "family": (ADMITTED | EXCLUDED)[n][1],
                          "quant": (ADMITTED | EXCLUDED)[n][2], "admitted": n in ADMITTED,
                          "truth": point[n][0], "truth_ci": np.percentile(bt[n][:, 0], [2.5, 97.5]).tolist(),
                          "regime": point[n][1], "regime_ci": np.percentile(bt[n][:, 1], [2.5, 97.5]).tolist(),
                          "staleness": point[n][2], "staleness_ci": np.percentile(bt[n][:, 2], [2.5, 97.5]).tolist(),
                          "regime_paired_win": paired[n], "regime_sign_p": sign_p[n]} for n in data},
        "model_level": {
            "ols_slope": float(b[1]), "ols_intercept": float(b[0]),
            "se_classical": se_cls, "p_classical_t": float(p_cls), "df": len(adm) - 2,
            "se_cr1_family": se_cr, "n_clusters": nG, "p_cr1_t_df2": float(p_cr_t),
            "pearson_r": float(r_p), "pearson_p": float(p_pear),
            "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "exact_permutation_p": p_perm,
            "slope_entity_bootstrap_ci": np.percentile(slope_boot, [2.5, 97.5]).tolist(),
            "leave_one_model_out_slopes": loo, "leave_one_family_out": lofo,
        },
        "split_half": {"slope_median": float(np.median(split_slopes)),
                       "slope_2.5_97.5": np.percentile(split_slopes, [2.5, 97.5]).tolist(),
                       "frac_slope_negative": float((split_slopes < 0).mean()),
                       "spearman_median": float(np.median(split_rho))},
        "staleness_mean_admitted": float(np.mean([point[n][2] for n in adm])),
        "staleness_mean_ci": np.percentile(bt_mean := np.mean([bt[n][:, 2] for n in adm], axis=0), [2.5, 97.5]).tolist(),
        "lmm": lmm,
    }
    json.dump(out, open(OUT / f"inverse_{tag}.json", "w"), indent=2)

    # table
    with open(OUT / f"inverse_table_{tag}.md", "w") as f:
        f.write("| model | family | quant | truth AUROC [95% CI] | regime AUROC [95% CI] | regime paired win | sign p | staleness [95% CI] |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        order = sorted(data, key=lambda n: (n not in ADMITTED, point[n][0]))
        for n in order:
            pm = out["per_model"][n]
            nm = pm["label"] + ("" if pm["admitted"] else " (excluded)")
            f.write(f"| {nm} | {pm['family']} | {pm['quant']} | {pm['truth']:.3f} [{pm['truth_ci'][0]:.3f}, {pm['truth_ci'][1]:.3f}] | "
                    f"{pm['regime']:.3f} [{pm['regime_ci'][0]:.3f}, {pm['regime_ci'][1]:.3f}] | {pm['regime_paired_win']:.3f} | "
                    f"{pm['regime_sign_p']:.4f} | {pm['staleness']:.3f} [{pm['staleness_ci'][0]:.3f}, {pm['staleness_ci'][1]:.3f}] |\n")
        ml = out["model_level"]
        f.write(f"\n**Model level (n = {len(adm)} admitted).** OLS slope {ml['ols_slope']:.3f}; "
                f"classical SE {ml['se_classical']:.3f}, p = {ml['p_classical_t']:.4f} (t, {ml['df']} df); "
                f"family-clustered CR1 SE {ml['se_cr1_family']:.3f} ({ml['n_clusters']} clusters), p = {ml['p_cr1_t_df2']:.3f} (t, 2 df). "
                f"Pearson r = {ml['pearson_r']:.3f} (p = {ml['pearson_p']:.4f}); Spearman rho = {ml['spearman_rho']:.3f} (p = {ml['spearman_p']:.4f}); "
                f"permutation p = {ml['exact_permutation_p']:.4f} ({'exact' if len(adm) <= 10 else 'Monte Carlo'}). Entity-bootstrap 95% CI for the slope "
                f"[{ml['slope_entity_bootstrap_ci'][0]:.3f}, {ml['slope_entity_bootstrap_ci'][1]:.3f}].\n")
        f.write(f"\nLeave-one-model-out slopes: {min(ml['leave_one_model_out_slopes'].values()):.3f} to "
                f"{max(ml['leave_one_model_out_slopes'].values()):.3f}. Leave-one-family-out: "
                + "; ".join(f"{k}: slope {v['slope']:.3f}, rho {v['spearman']:.3f} (n={v['n']})" for k, v in ml["leave_one_family_out"].items()) + ".\n")
        sh = out["split_half"]
        f.write(f"\n**Split-half control** (truth on one half of entities, regime on the other, {N_SPLIT} splits): "
                f"median slope {sh['slope_median']:.3f}, 95% of splits in [{sh['slope_2.5_97.5'][0]:.3f}, {sh['slope_2.5_97.5'][1]:.3f}], "
                f"{100 * sh['frac_slope_negative']:.1f}% negative; median Spearman {sh['spearman_median']:.3f}.\n")
        f.write("\n**Crossed random-effects model** (REML; random intercepts entity, model, family; b1 tested on t with n_models - 2 df):\n\n")
        f.write("| outcome | b1 (per unit truth AUROC) | SE | t | p | variances entity / model / family / residual |\n|---|---|---|---|---|---|\n")
        for k, v in lmm.items():
            f.write(f"| {k} | {v['b1']:.3f} | {v['se_b1']:.3f} | {v['t']:.2f} | {v['p_t']:.4f} ({v['df']} df) | "
                    + " / ".join(f"{s:.4f}" for s in v["variances"]) + " |\n")
        f.write(f"\nStaleness: mean over admitted models {out['staleness_mean_admitted']:.3f}, entity-bootstrap 95% CI "
                f"[{out['staleness_mean_ci'][0]:.3f}, {out['staleness_mean_ci'][1]:.3f}].\n")

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    colors = {"gemma": "#c0504d", "qwen": "#1f4e79", "gpt-oss": "#7f7f7f"}
    for n in data:
        pm = out["per_model"][n]
        xe = [[pm["truth"] - pm["truth_ci"][0]], [pm["truth_ci"][1] - pm["truth"]]]
        ye = [[pm["regime"] - pm["regime_ci"][0]], [pm["regime_ci"][1] - pm["regime"]]]
        ax.errorbar(pm["truth"], pm["regime"], xerr=xe, yerr=ye, fmt="o" if pm["admitted"] else "x",
                    color=colors[pm["family"]], alpha=1 if pm["admitted"] else 0.4, ms=6, lw=0.8, capsize=0)
        ax.annotate(pm["label"].replace(" (Qwen2.5-7B + TSCT)", ""), (pm["truth"], pm["regime"]),
                    fontsize=7, xytext=LABEL_OFFSET.get(n, (4, 4)), textcoords="offset points",
                    alpha=1 if pm["admitted"] else 0.5)
    xx = np.linspace(0.70, 0.87, 50)
    ax.plot(xx, b[0] + b[1] * xx, color="k", lw=1, ls="--")
    ax.axhline(0.5, color="k", lw=0.6)
    ax.axvline(0.70, color="k", lw=0.6, ls=":")
    ax.text(0.702, 0.33, "admission bar 0.70", fontsize=7, rotation=90)
    ax.set_xlabel("truth AUROC (true vs false founder)")
    ax.set_ylabel("regime AUROC (immutable true vs volatile current)")
    ax.set_title(f"slope {b[1]:.2f}, Spearman {rho:.2f}, permutation p = {p_perm:.3f}, n = {len(adm)}\n"
                 "bars: 95% entity-bootstrap CI; x = excluded (not fitted)", fontsize=9)
    for f_, c in colors.items():
        ax.scatter([], [], color=c, label=f_)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax = axes[1]
    ax.hist(split_slopes, bins=50, color="#9dc3e6")
    ax.axvline(b[1], color="k", lw=1.2, label="full-data slope")
    ax.axvline(0, color="#c0504d", lw=1)
    ax.set_xlabel("slope, truth from one half of entities, regime from the other")
    ax.set_ylabel("splits")
    ax.set_title(f"split-half control: {100 * (split_slopes < 0).mean():.1f}% of {N_SPLIT} splits negative", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"inverse_{tag}.png", dpi=180)
    fig.savefig(OUT / f"inverse_{tag}.pdf")
    print(open(OUT / f"inverse_table_{tag}.md").read())


if __name__ == "__main__":
    main()
