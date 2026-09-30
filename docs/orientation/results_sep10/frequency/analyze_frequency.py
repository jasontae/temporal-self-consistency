"""Frequency experiment: does log(count_expired / count_current) predict the per-entity
preference for the expired office-holder? Implements PREREG.md exactly; any
deviation is listed in RESULTS.md.

Inputs:
  counts.jsonl                               (fetch_counts.py; infini-gram API)
  data/prep/predictions_7b/matched_<m>.jsonl (pmi per claim, original build)
  data/stress_tests/matched_regime_claims.jsonl
  A5 covariates: data/prep/wikidata_a6/set_persons.json (sitelinks),
                 data/prep/wikidata_matched_cache/ + gold_currency_audit.json (start dates)

Run from the repo root:
    python3 docs/orientation/results_sep10/frequency/analyze_frequency.py
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/orientation/results_sep10/inverse"))
sys.path.insert(0, str(ROOT / "docs/orientation/results_sep10/a5_prominence"))
from inverse_truth import reml_crossed  # noqa: E402
import a5_prominence  # noqa: E402

HERE = Path(__file__).resolve().parent
PRED = ROOT / "data" / "prep" / "predictions_7b"
SEED, N_PERM, CV_FOLDS, CV_REPEATS = 20260928, 10000, 10, 50
PRIMARY_INDEX, PRIMARY_FORM = "v4_dolma-v1_7_llama", "F1"
SECONDARY_INDEXES = ["v4_rpj_llama_s4", "v4_piletrain_llama"]
NINE = ["g3_4b", "qwen3_4b_th", "tsct", "qwen3_4b_it", "qwen25_7b", "gptoss_20b",
        "qwen36_35b", "qwen35_27b", "qwen36_27b_stock"]
MODEL_SETS = {"nine": NINE, "plus_qwen38": NINE + ["qwen38_27b"], "plus_e4b": NINE + ["g4_e4b"]}


def start_years(claims_rows):
    """A5's start_years, reading the tracked data/prep/wikidata_cache/ (the untracked
    wikidata_matched_cache/ A5 used is not in this worktree; deviation in RESULTS.md)."""
    orig = a5_prominence.glob.glob

    def redirected(pattern):
        return orig(pattern.replace("wikidata_matched_cache", "wikidata_cache"))
    a5_prominence.glob.glob = redirected
    try:
        return a5_prominence.start_years(claims_rows)
    finally:
        a5_prominence.glob.glob = orig


# ---------------------------------------------------------------- data
def load_counts(path):
    C = {}
    for line in open(path):
        r = json.loads(line)
        C[(r["index"], r["form"], r["entity_id"], r["value"])] = r
    # F3 was sent as two API-legal clauses (fetch_counts.py); its count is their sum
    for k in [k for k in C if k[1] == "F3a"]:
        a, b = C[k], C.get((k[0], "F3b", k[2], k[3]))
        ok = b is not None and a["count"] is not None and b["count"] is not None
        C[(k[0], "F3", k[2], k[3])] = {**a, "form": "F3", "count": (a["count"] + b["count"]) if ok else None,
                                       "approx": bool(a["approx"]) or bool(b and b["approx"])}
    return C


def load_scores(models):
    S = {}
    for m in models:
        by = defaultdict(dict)
        for line in open(PRED / f"matched_{m}.jsonl"):
            r = json.loads(line)
            by[r["entity_id"]][r["arm"]] = r["pmi"]
        S[m] = by
    return S


def outcome(S, models, ents):
    """z[m, e] = (pmi_stale - pmi_current) / SD_m over entities; ybar[e] = mean over models;
    win[m, e] = 1[pmi_stale > pmi_current]."""
    Z, W = [], []
    for m in models:
        d = np.array([S[m][e]["volatile_stale"] - S[m][e]["volatile_current"] for e in ents])
        Z.append(d / d.std(ddof=1))
        W.append((d > 0).astype(float))
    Z, W = np.array(Z), np.array(W)
    return Z, Z.mean(0), W


def xvec(C, index, form, ents, k=0.5):
    """x_e = ln((c_exp + k) / (c_cur + k)); None where a count is missing (API error)."""
    x, meta = [], []
    for e in ents:
        a, b = C.get((index, form, e, "expired")), C.get((index, form, e, "current"))
        if a is None or b is None or a["count"] is None or b["count"] is None:
            x.append(np.nan)
            meta.append(None)
            continue
        x.append(math.log((a["count"] + k) / (b["count"] + k)))
        meta.append((a["count"], b["count"], bool(a["approx"]) or bool(b["approx"])))
    return np.array(x), meta


# ---------------------------------------------------------------- statistics
def slope(x, y):
    xc = x - x.mean()
    return float(xc @ (y - y.mean()) / (xc @ xc))


def perm_slope(x, y, rng, n=N_PERM):
    """OLS slope of y on x with a two-sided permutation p (x permuted across entities)."""
    b = slope(x, y)
    xc = x - x.mean()
    yc = y - y.mean()
    P = np.argsort(rng.random((n, len(x))), axis=1)
    bp = (xc[P] @ yc) / (xc @ xc)
    p = (1 + np.sum(np.abs(bp) >= abs(b) - 1e-12)) / (n + 1)
    return b, float(p), bp


def perm_partial(x, y, C, rng, n=N_PERM):
    """Coefficient of x in y ~ 1 + x + C, with x permuted across entities and C held
    fixed (Frisch-Waugh). Two-sided permutation p."""
    X0 = np.column_stack([np.ones(len(y)), C]) if C is not None and C.size else np.ones((len(y), 1))
    M = np.eye(len(y)) - X0 @ np.linalg.pinv(X0)
    yr = M @ y

    def coef(xx):
        xr = xx @ M.T if xx.ndim == 2 else M @ xx
        return (xr @ yr) / np.einsum("...i,...i->...", xr, xr)

    b = float(coef(x))
    P = np.argsort(rng.random((n, len(x))), axis=1)
    bp = coef(x[P])
    p = (1 + np.sum(np.abs(bp) >= abs(b) - 1e-12)) / (n + 1)
    return b, float(p)


def lmm(Z, x):
    nM, nE = Z.shape
    y = Z.reshape(-1)
    xx = np.tile(x, nM)
    ent = np.tile(np.arange(nE), nM)
    mod = np.repeat(np.arange(nM), nE)
    f = reml_crossed(y, xx, [ent, mod])
    t = f["b1"] / f["se_b1"]
    df = nE - 2
    return {"b1": f["b1"], "se": f["se_b1"], "t": t, "df": df, "p": float(2 * stats.t.sf(abs(t), df)),
            "b0": f["b0"], "variances_entity_model_resid": f["variances"], "converged": f["converged"]}


def cv_mse(y, designs, rng):
    """Repeated k-fold CV by entity. designs: name -> (n, p) predictor matrix (no intercept)."""
    n = len(y)
    out = {k: [] for k in designs}
    for _ in range(CV_REPEATS):
        folds = np.array_split(rng.permutation(n), CV_FOLDS)
        err = {k: np.zeros(n) for k in designs}
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            for k, D in designs.items():
                Xtr = np.column_stack([np.ones(len(tr))] + ([D[tr]] if D is not None else []))
                Xte = np.column_stack([np.ones(len(te))] + ([D[te]] if D is not None else []))
                b, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
                err[k][te] = (y[te] - Xte @ b) ** 2
        for k in designs:
            out[k].append(err[k].mean())
    return {k: {"mean": float(np.mean(v)), "sd_over_repeats": float(np.std(v))} for k, v in out.items()}


def reml_multi(y, X, groups):
    """REML with fixed-effect matrix X (intercept included) and crossed random intercepts.
    Same estimator as inverse_truth.reml_crossed, generalized to several fixed effects."""
    from scipy import optimize
    n = len(y)
    ZZ = [np.equal.outer(g, g).astype(float) for g in groups]

    def build(theta):
        v = np.exp(np.clip(theta, -30, 30))
        V = v[-1] * np.eye(n)
        for s_, Z in zip(v[:-1], ZZ):
            V = V + s_ * Z
        return V

    def negll(theta):
        V = build(theta)
        try:
            L = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return 1e12
        ViX = np.linalg.solve(L.T, np.linalg.solve(L, X))
        Viy = np.linalg.solve(L.T, np.linalg.solve(L, y))
        A = X.T @ ViX
        try:
            b = np.linalg.solve(A, X.T @ Viy)
        except np.linalg.LinAlgError:
            return 1e12
        r = y - X @ b
        Vir = np.linalg.solve(L.T, np.linalg.solve(L, r))
        return 0.5 * (2 * np.log(np.diag(L)).sum() + np.linalg.slogdet(A)[1] + r @ Vir)

    th0 = np.full(len(groups) + 1, np.log(np.var(y) / (len(groups) + 1)))
    fit = optimize.minimize(negll, th0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
    Vi = np.linalg.inv(build(fit.x))
    cov = np.linalg.inv(X.T @ Vi @ X)
    b = cov @ X.T @ Vi @ y
    return b, np.sqrt(np.diag(cov)), bool(fit.success)


def expired_tenure(claims_rows, qid_of):
    """Years between the expired holder's P580 and P582 on the entity's volatile property
    (latest statement for that QID), from the tracked data/prep/wikidata_cache/."""
    import glob
    import datetime as dt
    ents = {}
    for f in sorted(glob.glob(str(ROOT / "data/prep/wikidata_cache/claims_*.json"))):
        x = json.load(open(f))
        ents.update(x.get("entities", x))

    def date(q):
        t = q.get("datavalue", {}).get("value", {}).get("time")
        return dt.date(int(t[1:5]), max(1, int(t[6:8])), 1) if t else None
    out = {}
    for r in claims_rows:
        if r["arm"] != "volatile_stale":
            continue
        qid = qid_of.get(r["person"])
        best = None
        for st in ents.get(r["entity_id"], {}).get("claims", {}).get(r["property"], []):
            if not qid or st["mainsnak"].get("datavalue", {}).get("value", {}).get("id") != qid:
                continue
            s0 = [date(q) for q in st.get("qualifiers", {}).get("P580", []) if date(q)]
            s1 = [date(q) for q in st.get("qualifiers", {}).get("P582", []) if date(q)]
            if s0 and s1 and (best is None or max(s0) > best[0]):
                best = (max(s0), max(s1))
        if best and best[1] >= best[0]:
            out[r["entity_id"]] = (best[1] - best[0]).days / 365.25
    return out


def auroc(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# ---------------------------------------------------------------- main
def main():
    global HERE, HERE_IN
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=str(HERE / "counts.jsonl"))
    ap.add_argument("--out", default=str(HERE), help="output directory (tests only)")
    args = ap.parse_args()
    counts_path = Path(args.counts)
    HERE_IN = counts_path.parent
    HERE = Path(args.out)
    rng = np.random.default_rng(SEED)
    C = load_counts(counts_path)
    claims = [json.loads(l) for l in open(ROOT / "data/stress_tests/matched_regime_claims.jsonl")]
    persons = defaultdict(dict)
    for r in claims:
        persons[r["entity_id"]][r["arm"]] = r["person"]
    label = {r["entity_id"]: r["entity"] for r in claims}
    ents_all = sorted(persons)
    assert len(ents_all) == 106

    S_all = load_scores(sorted(set(sum(MODEL_SETS.values(), []))))
    res = {"prereg": "PREREG.md", "seed": SEED, "n_perm": N_PERM, "deviations": []}

    # count quality
    errs = [k for k, r in C.items() if r["count"] is None]
    res["count_quality"] = {
        "rows": len(C), "errors": len(errs),
        "by_index_form": {f"{i}|{f}": {"rows": sum(1 for k in C if k[0] == i and k[1] == f),
                                       "zero": sum(1 for k, r in C.items() if k[0] == i and k[1] == f and r["count"] == 0),
                                       "approx": sum(1 for k, r in C.items() if k[0] == i and k[1] == f and r["approx"])}
                          for i, f in sorted({(k[0], k[1]) for k in C})},
    }

    Z, ybar, W = outcome(S_all, NINE, ents_all)

    # ---- primary
    x, meta = xvec(C, PRIMARY_INDEX, PRIMARY_FORM, ents_all)
    ok = ~np.isnan(x)
    if (~ok).any():
        res["deviations"].append(f"{int((~ok).sum())} entities dropped from the primary analysis for API errors")
    e_ok = [e for e, o in zip(ents_all, ok) if o]
    b, p, bnull = perm_slope(x[ok], ybar[ok], rng)
    decision = ("supported" if (b > 0 and p < 0.05) else "contradicted" if (b < 0 and p < 0.05) else "null")
    res["primary"] = {"index": PRIMARY_INDEX, "form": PRIMARY_FORM, "n_entities": int(ok.sum()),
                      "slope": b, "perm_p_two_sided": p, "decision": decision,
                      "pearson_r": float(stats.pearsonr(x[ok], ybar[ok])[0]),
                      "spearman_rho": float(stats.spearmanr(x[ok], ybar[ok])[0]),
                      "lmm": lmm(Z[:, ok], x[ok]),
                      "x_summary": {"both_zero": int(sum(1 for m in meta if m and m[0] == 0 and m[1] == 0)),
                                    "expired_gt_current": int(sum(1 for m in meta if m and m[0] > m[1])),
                                    "current_gt_expired": int(sum(1 for m in meta if m and m[1] > m[0])),
                                    "any_approx": int(sum(1 for m in meta if m and m[2])),
                                    "median_x": float(np.median(x[ok]))}}

    # ---- no-count baseline (98-entity subset with sitelinks and start years)
    sl = {}
    for pr in json.load(open(ROOT / "data/prep/wikidata_a6/set_persons.json"))["rows"]:
        if pr["label"]:
            sl[pr["label"]] = max(sl.get(pr["label"], 0), pr["sitelinks"])
    years = start_years(claims)
    res["covariate_check"] = {"n_with_years": len(years), "median_years": float(np.median(list(years.values())))}
    sub = [i for i, e in enumerate(ents_all) if ok[i] and e in years
           and persons[e]["volatile_current"] in sl and persons[e]["volatile_stale"] in sl]
    es = [ents_all[i] for i in sub]
    dsl = np.array([np.log1p(sl[persons[e]["volatile_current"]]) - np.log1p(sl[persons[e]["volatile_stale"]]) for e in es])
    yrs = np.array([years[e] for e in es])
    cov = np.column_stack([dsl, yrs])
    xs, ys = x[sub], ybar[sub]
    cv = cv_mse(ys, {"M0 intercept": None, "M1 no-count covariates": cov,
                     "M2 counts": xs[:, None], "M3 counts + covariates": np.column_stack([xs, cov])}, rng)
    b3, p3 = perm_partial(xs, ys, cov, rng)
    res["no_count_baseline"] = {
        "n_entities": len(sub), "cv_mse": cv,
        "M3_x_coef": b3, "M3_x_perm_p": p3,
        "counts_add": bool(cv["M3 counts + covariates"]["mean"] < cv["M1 no-count covariates"]["mean"] and p3 < 0.05),
        "corr_x_years": float(stats.pearsonr(xs, yrs)[0]), "spearman_x_years": float(stats.spearmanr(xs, yrs)[0]),
        "corr_x_dsl": float(stats.pearsonr(xs, dsl)[0]),
        "M1_slopes_perm": {"dsl": perm_partial(dsl, ys, yrs[:, None], rng), "years": perm_partial(yrs, ys, dsl[:, None], rng)},
    }

    # ---- PREREG amendment 1: controlled test C1
    qid_of = {}
    for pr in json.load(open(ROOT / "data/prep/wikidata_a6/set_persons.json"))["rows"]:
        if pr["label"]:
            qid_of.setdefault(pr["label"], pr["qid"])
    tenure = expired_tenure(claims, qid_of)
    ntok = {}
    tl = HERE_IN / "token_lengths.jsonl"
    if tl.exists():
        for line in open(tl):
            r = json.loads(line)
            ntok[r["person"]] = r["n_tokens"]
    role = {r["entity_id"]: r["role"] for r in claims if r["arm"] == "volatile_current"}
    ent_cnt = {e: (C.get((PRIMARY_INDEX, "F5", e, "entity")) or {}).get("count") for e in ents_all}

    def controls(e, use_char=False):
        cur, exp_ = persons[e]["volatile_current"], persons[e]["volatile_stale"]
        tok = (len(exp_) - len(cur)) if use_char else (
            None if ntok.get(exp_) is None or ntok.get(cur) is None else ntok[exp_] - ntok[cur])
        vals = {"d_tokens": tok,
                "role_ceo": float(role[e] == "CEO"), "role_other": float(role[e] not in ("CEO", "chairperson")),
                "log_entity_count": None if ent_cnt[e] is None else math.log1p(ent_cnt[e]),
                "years_current": years.get(e), "tenure_expired": tenure.get(e),
                "d_log_sitelinks": (np.log1p(sl[cur]) - np.log1p(sl[exp_])) if cur in sl and exp_ in sl else None}
        # PREREG defines prominence as current - expired (M1); keep that sign
        return vals
    CTRL = ["d_tokens", "role_ceo", "role_other", "log_entity_count", "years_current", "tenure_expired", "d_log_sitelinks"]
    GROUPS = {"token length": ["d_tokens"], "relation": ["role_ceo", "role_other"], "entity frequency": ["log_entity_count"],
              "time in office (current)": ["years_current"], "time in office (expired)": ["tenure_expired"],
              "prominence": ["d_log_sitelinks"]}

    def run_c(cols, use_char=False):
        rows = [(i, controls(e, use_char)) for i, e in enumerate(ents_all) if ok[i]]
        rows = [(i, v) for i, v in rows if all(v[c] is not None for c in cols)]
        idx = np.array([i for i, _ in rows])
        Cm = np.array([[v[c] for c in cols] for _, v in rows], dtype=float) if cols else None
        bc, pc = perm_partial(x[idx], ybar[idx], Cm, rng)
        # REML with the same fixed effects, entity + model random intercepts
        nM = Z.shape[0]
        Xf = np.column_stack([np.ones(len(idx) * nM), np.tile(x[idx], nM)] +
                             ([np.tile(Cm[:, k], nM) for k in range(Cm.shape[1])] if cols else []))
        bb, se, conv = reml_multi(Z[:, idx].reshape(-1), Xf,
                                  [np.tile(np.arange(len(idx)), nM), np.repeat(np.arange(nM), len(idx))])
        df = len(idx) - Xf.shape[1]
        t = bb[1] / se[1]
        return {"n": int(len(idx)), "x_coef": bc, "x_perm_p": pc,
                "reml_b1": float(bb[1]), "reml_se": float(se[1]), "reml_p": float(2 * stats.t.sf(abs(t), df)), "reml_df": int(df),
                "reml_converged": conv}

    c1 = run_c(CTRL)
    c1["dropped_for_missing_control"] = int(ok.sum()) - c1["n"]
    c1["missing_by_control"] = {c: int(sum(1 for i, e in enumerate(ents_all) if ok[i] and controls(e)[c] is None)) for c in CTRL}
    res["controlled_C1"] = c1
    if c1["dropped_for_missing_control"] > 25:
        res["controlled_C1b_no_tenure"] = run_c([c for c in CTRL if c != "tenure_expired"])
    res["S9_drop_one"] = {f"drop {g}": run_c([c for c in CTRL if c not in cols]) for g, cols in GROUPS.items()}
    res["S9_drop_one"]["char length in place of tokens"] = run_c(CTRL, use_char=True)
    primary_ok = res["primary"]["decision"] == "supported"
    res["reading_rule"] = ("familiarity wording" if primary_ok and c1["x_coef"] > 0 and c1["x_perm_p"] < 0.05
                           else "behaves-like wording" if primary_ok else "primary not supported")

    # ---- secondary (exploratory)
    sec = {}
    # S1: other forms on Dolma
    for form in ("F2", "F3"):
        xf, mf = xvec(C, PRIMARY_INDEX, form, ents_all)
        o = ~np.isnan(xf)
        cover = int(sum(1 for m in mf if m and (m[0] + m[1]) > 0))
        if form == "F3" and cover < 30:
            sec["S1_F3"] = {"coverage_nonzero_entities": cover, "analysed": False}
            continue
        bb, pp, _ = perm_slope(xf[o], ybar[o], rng)
        sec[f"S1_{form}"] = {"coverage_nonzero_entities": cover, "slope": bb, "perm_p": pp, "n": int(o.sum())}
    # S2: other indexes, F1, Holm across the three
    s2 = {PRIMARY_INDEX: (b, p, int(ok.sum()))}
    for idx in SECONDARY_INDEXES:
        xi, mi = xvec(C, idx, "F1", ents_all)
        o = ~np.isnan(xi)
        bb, pp, _ = perm_slope(xi[o], ybar[o], rng)
        s2[idx] = (bb, pp, int(o.sum()))
        sec[f"S2_{idx}_coverage_nonzero"] = int(sum(1 for m in mi if m and (m[0] + m[1]) > 0))
    adj = holm([v[1] for v in s2.values()])
    sec["S2_indexes_F1"] = {k: {"slope": v[0], "perm_p": v[1], "holm_p": a, "n": v[2]} for (k, v), a in zip(s2.items(), adj)}
    # S3: co-occurrence vs marginal name frequency
    xn, _ = xvec(C, PRIMARY_INDEX, "F4", ents_all)
    o = ok & ~np.isnan(xn)
    bx, px = perm_partial(x[o], ybar[o], xn[o][:, None], rng)
    bn, pn = perm_partial(xn[o], ybar[o], x[o][:, None], rng)
    bn_alone, pn_alone, _ = perm_slope(xn[o], ybar[o], rng)
    sec["S3_name_frequency"] = {"n": int(o.sum()), "x_coef_given_name": bx, "x_perm_p": px,
                                "name_coef_given_x": bn, "name_perm_p": pn,
                                "name_alone_slope": bn_alone, "name_alone_perm_p": pn_alone,
                                "corr_x_xname": float(stats.pearsonr(x[o], xn[o])[0])}
    # S4: binary outcome
    wbar = W.mean(0)
    bw, pw, _ = perm_slope(x[ok], wbar[ok], rng)
    sec["S4_paired_win"] = {"slope": bw, "perm_p": pw}
    # S5: per-model slopes
    per = {m: slope(x[ok], Z[i, ok]) for i, m in enumerate(NINE)}
    sec["S5_per_model"] = {"slopes": per, "n_positive": int(sum(v > 0 for v in per.values()))}
    # S6: robustness
    both0 = np.array([m is not None and m[0] == 0 and m[1] == 0 for m in meta])
    apx = np.array([m is not None and m[2] for m in meta])
    for name, mask in (("S6a_drop_both_zero", ok & ~both0), ("S6b_drop_approx", ok & ~apx)):
        bb, pp, _ = perm_slope(x[mask], ybar[mask], rng)
        sec[name] = {"n": int(mask.sum()), "slope": bb, "perm_p": pp}
    x1, _ = xvec(C, PRIMARY_INDEX, PRIMARY_FORM, ents_all, k=1.0)
    bb, pp, _ = perm_slope(x1[ok], ybar[ok], rng)
    sec["S6c_k1"] = {"slope": bb, "perm_p": pp}
    # S7: model sets
    for name in ("plus_qwen38", "plus_e4b"):
        Zs, ybs, _ = outcome(S_all, MODEL_SETS[name], ents_all)
        bb, pp, _ = perm_slope(x[ok], ybs[ok], rng)
        sec[f"S7_{name}"] = {"slope": bb, "perm_p": pp}
    # S8: descriptive AUROC of x for sign of ybar
    sec["S8_auroc_x_for_expired_preferred"] = auroc(x[ok], ybar[ok] > 0)
    sec["share_entities_expired_preferred"] = float(np.mean(ybar > 0))
    res["secondary"] = sec

    # per-entity table
    res["per_entity"] = [{"entity_id": e, "entity": label[e], "current": persons[e]["volatile_current"],
                          "expired": persons[e]["volatile_stale"],
                          "c_cur": (meta[i][1] if meta[i] else None), "c_exp": (meta[i][0] if meta[i] else None),
                          "approx": (meta[i][2] if meta[i] else None),
                          "x": (None if np.isnan(x[i]) else float(x[i])), "ybar": float(ybar[i])}
                         for i, e in enumerate(ents_all)]

    json.dump(res, open(HERE / "frequency_results.json", "w"), indent=1, ensure_ascii=False)
    write_md(res)
    plot(x[ok], ybar[ok], b, bnull, p)
    print(json.dumps({k: res[k] for k in ("primary", "no_count_baseline")}, indent=1, default=str))
    print(json.dumps(res["secondary"], indent=1, default=str))


def write_md(res):
    P = res["primary"]
    L = P["lmm"]
    B = res["no_count_baseline"]
    S = res["secondary"]
    lines = [
        "# Frequency experiment: results table (generated by analyze_frequency.py)", "",
        f"**Primary decision: {P['decision'].upper()}.** Index `{P['index']}`, form {P['form']} "
        f"(entity AND person within 100 tokens), n = {P['n_entities']} entities, nine admitted models.", "",
        "| statistic | value |", "|---|---|",
        f"| OLS slope of ȳ_e on x_e | {P['slope']:+.4f} |",
        f"| permutation p (two-sided, {res['n_perm']:,} perms) | {P['perm_p_two_sided']:.4f} |",
        f"| Pearson r / Spearman ρ | {P['pearson_r']:+.3f} / {P['spearman_rho']:+.3f} |",
        f"| crossed REML (entity, model): b1 (SE), p ({L['df']} df) | {L['b1']:+.4f} ({L['se']:.4f}), p = {L['p']:.4f} |",
        f"| entities with both counts zero | {P['x_summary']['both_zero']} |",
        f"| entities with c_exp > c_cur / c_cur > c_exp | {P['x_summary']['expired_gt_current']} / {P['x_summary']['current_gt_expired']} |",
        f"| entities with an approximate count | {P['x_summary']['any_approx']} |", "",
        f"## No-count baseline ({B['n_entities']} entities with sitelinks and start dates)", "",
        "| model | CV-MSE (mean over 50 × 10-fold) |", "|---|---|",
    ] + [f"| {k} | {v['mean']:.4f} (sd {v['sd_over_repeats']:.4f}) |" for k, v in B["cv_mse"].items()] + [
        "",
        f"x in M3 (holding covariates fixed): coefficient {B['M3_x_coef']:+.4f}, permutation p = {B['M3_x_perm_p']:.4f}. "
        f"Counts add over the no-count baseline (pre-registered rule): **{'yes' if B['counts_add'] else 'no'}**.",
        f"Correlation of x with years since the change: r = {B['corr_x_years']:+.3f} (Spearman {B['spearman_x_years']:+.3f}); "
        f"with Δlog sitelinks: r = {B['corr_x_dsl']:+.3f}.", "",
        "## Controlled test C1 (PREREG amendment 1)", "",
        f"n = {res['controlled_C1']['n']} entities with every control ({res['controlled_C1']['dropped_for_missing_control']} dropped; "
        f"missing by control: {res['controlled_C1']['missing_by_control']}).", "",
        "| fit | n | x coefficient | permutation p | REML b1 (SE), p |", "|---|---|---|---|---|",
    ] + [f"| {name} | {v['n']} | {v['x_coef']:+.4f} | {v['x_perm_p']:.4f} | {v['reml_b1']:+.4f} ({v['reml_se']:.4f}), p = {v['reml_p']:.4f} |"
         for name, v in ([("C1 all controls", res["controlled_C1"])] +
                         ([("C1b without expired tenure", res["controlled_C1b_no_tenure"])] if "controlled_C1b_no_tenure" in res else []) +
                         [(f"S9 {g}", v) for g, v in res["S9_drop_one"].items()])] + [
        "", f"Reading rule (amendment 1): **{res['reading_rule']}**.", "",
        "## Secondary (exploratory)", "",
        "| analysis | slope | p | n / note |", "|---|---|---|---|",
    ]
    for k in ("S1_F2", "S1_F3"):
        v = S.get(k, {})
        if v.get("analysed") is False:
            lines.append(f"| {k} (exact-phrase templates) | — | — | coverage {v['coverage_nonzero_entities']} < 30: not analysed |")
        elif v:
            lines.append(f"| {k} | {v['slope']:+.4f} | {v['perm_p']:.4f} | n {v['n']}, nonzero {v['coverage_nonzero_entities']} |")
    for idx, v in S["S2_indexes_F1"].items():
        lines.append(f"| S2 F1 on `{idx}` | {v['slope']:+.4f} | {v['perm_p']:.4f} (Holm {v['holm_p']:.4f}) | n {v['n']} |")
    s3 = S["S3_name_frequency"]
    lines += [
        f"| S3 x given name-frequency ratio | {s3['x_coef_given_name']:+.4f} | {s3['x_perm_p']:.4f} | r(x, x_name) = {s3['corr_x_xname']:+.3f} |",
        f"| S3 name-frequency ratio given x | {s3['name_coef_given_x']:+.4f} | {s3['name_perm_p']:.4f} | alone: {s3['name_alone_slope']:+.4f}, p {s3['name_alone_perm_p']:.4f} |",
        f"| S4 paired-win outcome | {S['S4_paired_win']['slope']:+.4f} | {S['S4_paired_win']['perm_p']:.4f} | |",
        f"| S5 per-model slopes | — | — | {S['S5_per_model']['n_positive']} of 9 positive |",
        f"| S6a drop both-zero | {S['S6a_drop_both_zero']['slope']:+.4f} | {S['S6a_drop_both_zero']['perm_p']:.4f} | n {S['S6a_drop_both_zero']['n']} |",
        f"| S6b drop approximate | {S['S6b_drop_approx']['slope']:+.4f} | {S['S6b_drop_approx']['perm_p']:.4f} | n {S['S6b_drop_approx']['n']} |",
        f"| S6c zero correction 1 | {S['S6c_k1']['slope']:+.4f} | {S['S6c_k1']['perm_p']:.4f} | |",
        f"| S7 + Qwen3.8-27B | {S['S7_plus_qwen38']['slope']:+.4f} | {S['S7_plus_qwen38']['perm_p']:.4f} | |",
        f"| S7 + gemma-4-E4B | {S['S7_plus_e4b']['slope']:+.4f} | {S['S7_plus_e4b']['perm_p']:.4f} | |",
        f"| S8 AUROC of x for ȳ_e > 0 | {S['S8_auroc_x_for_expired_preferred']:.3f} | | share ȳ_e > 0: {S['share_entities_expired_preferred']:.3f} |",
        "", "## Count quality", "", "| index / form | rows | zero | approx |", "|---|---|---|---|",
    ] + [f"| {k} | {v['rows']} | {v['zero']} | {v['approx']} |" for k, v in res["count_quality"]["by_index_form"].items()] + [
        "", f"API errors: {res['count_quality']['errors']}.",
    ]
    if res["deviations"]:
        lines += ["", "Deviations: " + "; ".join(res["deviations"])]
    (HERE / "frequency_results.md").write_text("\n".join(lines) + "\n")


def plot(x, y, b, bnull, p):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    a1.axhline(0, color="0.7", lw=0.8)
    a1.axvline(0, color="0.7", lw=0.8)
    a1.scatter(x, y, s=14, color="#3b6ea5", alpha=0.8, edgecolor="none")
    xx = np.linspace(x.min(), x.max(), 50)
    a1.plot(xx, y.mean() + b * (xx - x.mean()), color="#c0392b", lw=1.5)
    a1.set_xlabel("x = ln((c_expired + 0.5) / (c_current + 0.5)), Dolma v1.7, entity AND person ≤100 tokens")
    a1.set_ylabel("expired-value preference ȳ\n(PMI stale − current, z, mean of 9 models)")
    a1.xaxis.label.set_size(7.5)
    a2.hist(bnull, bins=50, color="0.75")
    a2.axvline(b, color="#c0392b", lw=1.5)
    a2.set_xlabel("slope under permutation")
    a2.set_title(f"observed {b:+.4f}, p = {p:.3f}", fontsize=9)
    a2.set_yticks([])
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"frequency.{ext}", dpi=200)


if __name__ == "__main__":
    main()
