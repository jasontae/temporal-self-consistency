"""A5: does the staleness preference survive name prominence, and does it depend
on how long ago the office changed?

The Sep 2 PDF reports "p = 0.39 for the prominence covariate" (sitelinks) and a
split at 2 years since the change (mean staleness AUROC 0.451 vs 0.438). No
code or data for either exists in the repo, so both are rebuilt here.

Staleness uses only the volatile_current and volatile_stale arms, which are
identical in the original and the disjoint-pool (A6) builds, so this result
does not depend on the false-founder rebuild.

Per entity:
  dsl    = log(1 + sitelinks(current holder)) - log(1 + sitelinks(stale holder))
           (Wikidata, data/prep/wikidata_a6/set_persons.json, fetched 2026-09-28)
  years  = years from the current holder's start (P580 on the entity's volatile
           property, cached claims) to the audit date 2026-08-11
Per entity x model:
  win    = 1[pmi(current) > pmi(stale)]     (mean over entities ~ staleness AUROC, paired)
  dz     = (pmi(current) - pmi(stale)) / sd of that model's pmi

Models: the nine admitted (Gemma-4 dropped); `--with-e4b` for the sensitivity.

Analyses:
  1. Crossed random-effects model (entity, model) of win and of dz on centred
     dsl. The intercept is the staleness preference at equal prominence
     (tested on t with n_entities - 1 df, since dsl varies by entity); the
     slope is the prominence covariate.
  2. Stratification by years since change (<= 2 vs > 2, the Sep 2 split, and
     tertiles), mean staleness AUROC across models with entity-bootstrap CIs.

Run from the repo root:
    python3 docs/orientation/results_sep10/a5_prominence/a5_prominence.py [--with-e4b]
"""
import argparse
import datetime as dt
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/orientation/results_sep10/inverse"))
from inverse_truth import reml_crossed  # noqa: E402

PRED = ROOT / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent
AUDIT_DATE = dt.date(2026, 8, 11)
SEED, N_BOOT = 20260928, 4000
MODELS = ["g3_4b", "qwen3_4b_th", "tsct", "qwen3_4b_it", "qwen25_7b", "gptoss_20b",
          "qwen36_35b", "qwen35_27b", "qwen36_27b_stock"]


def auroc(pos, neg):
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def start_years(claims_rows):
    ents = {}
    for f in sorted(glob.glob(str(ROOT / "data/prep/wikidata_matched_cache/claims_*.json"))):
        x = json.load(open(f))
        ents.update(x.get("entities", x))
    audit = {r["entity_id"]: r for r in json.load(open(ROOT / "data/prep/gold_currency_audit.json"))["rows"]}
    out = {}
    for r in claims_rows:
        if r["arm"] != "volatile_current":
            continue
        cq = audit.get(r["entity_id"], {}).get("wikidata_current_qid")
        best = None
        for s in ents.get(r["entity_id"], {}).get("claims", {}).get(r["property"], []):
            if s["mainsnak"].get("datavalue", {}).get("value", {}).get("id") != cq:
                continue
            for q in s.get("qualifiers", {}).get("P580", []):
                t = q.get("datavalue", {}).get("value", {}).get("time")
                if t:
                    y, m = int(t[1:5]), max(1, int(t[6:8]))
                    d = dt.date(y, m, 1)
                    best = d if best is None or d > best else best
        if best:
            out[r["entity_id"]] = (AUDIT_DATE - best).days / 365.25
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-e4b", action="store_true")
    args = ap.parse_args()
    models = MODELS + (["g4_e4b"] if args.with_e4b else [])
    tag = "with_e4b" if args.with_e4b else "nine"

    claims = [json.loads(l) for l in open(ROOT / "data/stress_tests/matched_regime_claims.jsonl")]
    sl = {}
    for p in json.load(open(ROOT / "data/prep/wikidata_a6/set_persons.json"))["rows"]:
        if p["label"]:
            sl[p["label"]] = max(sl.get(p["label"], 0), p["sitelinks"])
    persons = defaultdict(dict)
    for r in claims:
        persons[r["entity_id"]][r["arm"]] = r["person"]
    years = start_years(claims)

    scores = {}
    for m in models:
        by = defaultdict(dict)
        for line in open(PRED / f"matched_{m}.jsonl"):
            r = json.loads(line)
            by[r["entity_id"]][r["arm"]] = r["pmi"]
        scores[m] = by
    ents = sorted(e for e in persons
                  if persons[e]["volatile_current"] in sl and persons[e]["volatile_stale"] in sl
                  and e in years and all(len(scores[m].get(e, {})) == 4 for m in models))
    dsl = np.array([np.log1p(sl[persons[e]["volatile_current"]]) - np.log1p(sl[persons[e]["volatile_stale"]])
                    for e in ents])
    yrs = np.array([years[e] for e in ents])
    V = {m: np.array([scores[m][e]["volatile_current"] for e in ents]) for m in models}
    St = {m: np.array([scores[m][e]["volatile_stale"] for e in ents]) for m in models}
    nE = len(ents)

    # 1. random-effects model
    ent_code = np.tile(np.arange(nE), len(models))
    mod_code = np.repeat(np.arange(len(models)), nE)
    x = np.tile(dsl, len(models))  # uncentred: intercept = preference at equal prominence
    win = np.concatenate([(V[m] > St[m]).astype(float) for m in models])
    dz = np.concatenate([(V[m] - St[m]) / np.concatenate([V[m], St[m]]).std() for m in models])
    lmm = {}
    for name, y, null in (("paired win 1[current > stale]", win, 0.5), ("pmi difference, z within model", dz, 0.0)):
        f = reml_crossed(y, x, [ent_code, mod_code])
        # intercept SE from the same GLS fit
        n = len(y)
        X = np.column_stack([np.ones(n), x])
        v = f["variances"]
        Z = [np.equal.outer(g, g).astype(float) for g in (ent_code, mod_code)]
        Vm = v[0] * Z[0] + v[1] * Z[1] + v[2] * np.eye(n)
        cov = np.linalg.inv(X.T @ np.linalg.solve(Vm, X))
        se0 = float(np.sqrt(cov[0, 0]))
        t0 = (f["b0"] - null) / se0
        t1 = f["b1"] / f["se_b1"]
        lmm[name] = {"intercept": f["b0"], "null": null, "se_intercept": se0,
                     "p_intercept": float(2 * stats.t.sf(abs(t0), nE - 1)),
                     "slope_dsl": f["b1"], "se_slope": f["se_b1"],
                     "p_slope": float(2 * stats.t.sf(abs(t1), nE - 1)),
                     "variances_entity_model_resid": f["variances"]}

    # 2. stratification by years since change
    rng = np.random.default_rng(SEED)

    def strat(mask):
        idx = np.where(mask)[0]
        pt = np.mean([auroc(V[m][idx], St[m][idx]) for m in models])
        bs = []
        for _ in range(N_BOOT):
            b = rng.choice(idx, len(idx))
            bs.append(np.mean([auroc(V[m][b], St[m][b]) for m in models]))
        return {"n_entities": int(len(idx)), "mean_staleness_auroc": float(pt),
                "ci95": np.percentile(bs, [2.5, 97.5]).tolist()}

    t1, t2 = np.percentile(yrs, [100 / 3, 200 / 3])
    strata = {"<= 2 years": strat(yrs <= 2), "> 2 years": strat(yrs > 2),
              f"tertile 1 (<= {t1:.1f} y)": strat(yrs <= t1),
              f"tertile 2": strat((yrs > t1) & (yrs <= t2)),
              f"tertile 3 (> {t2:.1f} y)": strat(yrs > t2)}
    rho, p_rho = stats.spearmanr(yrs, np.mean([(V[m] > St[m]).astype(float) for m in models], axis=0))

    res = {"models": models, "n_entities": nE, "audit_date": str(AUDIT_DATE),
           "dsl_mean": float(dsl.mean()), "dsl_share_current_more_prominent": float((dsl > 0).mean()),
           "years_median": float(np.median(yrs)), "lmm": lmm, "strata": strata,
           "spearman_years_vs_mean_win": {"rho": float(rho), "p": float(p_rho)}}
    json.dump(res, open(OUT / f"a5_{tag}.json", "w"), indent=2)
    lines = [f"### A5 ({tag}: {len(models)} models, {nE} entities)\n",
             f"Current holder more prominent than stale holder on {100 * (dsl > 0).mean():.0f}% of entities "
             f"(mean Δlog(1+sitelinks) {dsl.mean():+.2f}). Median time since change {np.median(yrs):.1f} y (to {AUDIT_DATE}).\n",
             "| outcome | staleness preference at equal prominence (intercept) | p | prominence covariate (slope on Δlog sitelinks) | p |",
             "|---|---|---|---|---|"]
    for k, v in lmm.items():
        lines.append(f"| {k} | {v['intercept']:.3f} (null {v['null']}), SE {v['se_intercept']:.3f} | {v['p_intercept']:.2g} | "
                     f"{v['slope_dsl']:+.3f}, SE {v['se_slope']:.3f} | {v['p_slope']:.3f} |")
    lines += ["", "| stratum (years since change) | entities | mean staleness AUROC [95% CI] |", "|---|---|---|"]
    for k, v in strata.items():
        lines.append(f"| {k} | {v['n_entities']} | {v['mean_staleness_auroc']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |")
    lines.append(f"\nSpearman(years since change, mean paired win across models) = {rho:.3f} (p = {p_rho:.3f}).")
    (OUT / f"a5_table_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
