"""Directive 3 (worker B): are truth AUROC and regime AUROC algebraically coupled?

Fused-critique attack 10: "the temporality score is coupled to the truth score".
Both contrasts share the immutable_true arm (truth = T vs F, regime = T vs V), so a
model whose T scores happen to sit high gets both AUROCs pushed up together. The
split-half control in ../inverse/ removes the shared items; this simulation asks
what slope the design produces when temporal sensitivity is held FIXED across models
and only truth discrimination varies.

Generative null, parameters estimated from the nine admitted models' PMI (z within model):
  score(e, arm, m) = u_em + mu_arm,m + eps
  u_em = sigma_u (sqrt(rho) g_e + sqrt(1 - rho) h_em)   entity effect, shared across arms,
                                                         correlated across models (same entities)
  mu_T = 0, mu_F = -d_m, mu_V = -r, mu_S = -r
  d_m set so the expected unpaired truth AUROC equals each model's observed truth AUROC;
  r is the SAME for every model (fixed temporal sensitivity).
Scenarios: r = 0 (no temporal sensitivity) and r at the observed mean regime AUROC.
For each of 2,000 simulated 9-model panels: regime and truth AUROC per model (same
unpaired Mann-Whitney as inverse_truth.py), OLS slope of regime on truth.

    python3 docs/orientation/results_sep10/capability/null_simulation.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/orientation/results_sep10/inverse"))
from inverse_truth import ADMITTED, ARMS, auroc, load, ols  # noqa: E402

OUT = Path(__file__).resolve().parent
PRED = ROOT / "data" / "prep" / "predictions_7b"
SEED, N_SIM = 20260928, 2000


def main():
    rng = np.random.default_rng(SEED)
    models = list(ADMITTED)
    obs = json.load(open(ROOT / "docs/orientation/results_sep10/inverse/inverse_matched.json"))["per_model"]
    truth_obs = np.array([obs[m]["truth"] for m in models])
    regime_obs = np.array([obs[m]["regime"] for m in models])
    b_obs = float(ols(truth_obs, regime_obs)[0][1])

    # variance components from the real scores (z within model)
    mats = []
    for m in models:
        d = load(PRED / f"matched_{m}.jsonl")
        ents = sorted(d)
        S = np.array([[d[e][a] for a in ARMS] for e in ents])
        mats.append((S - S.mean()) / S.std())
    M = np.array(mats)                      # models x entities x arms
    resid = M - M.mean(1, keepdims=True)    # remove arm means per model
    ent_mean = resid.mean(2)                # models x entities
    within = (resid - ent_mean[..., None]).var(axis=2, ddof=1).mean()
    sig_u2 = max(ent_mean.var(axis=1, ddof=1).mean() - within / 4, 1e-6)
    sig_e2 = within
    C = np.corrcoef(ent_mean)
    rho = float(C[np.triu_indices(len(models), 1)].mean())
    tot = np.sqrt(2 * (sig_u2 + sig_e2))
    d_m = stats.norm.ppf(truth_obs) * tot
    nE = M.shape[1]

    def panel(r):
        g = rng.standard_normal(nE)
        tr, rg = [], []
        for k in range(len(models)):
            u = np.sqrt(sig_u2) * (np.sqrt(rho) * g + np.sqrt(1 - rho) * rng.standard_normal(nE))
            eps = np.sqrt(sig_e2) * rng.standard_normal((nE, 3))
            T, F, V = u + eps[:, 0], u - d_m[k] + eps[:, 1], u - r + eps[:, 2]
            tr.append(auroc(T, F))
            rg.append(auroc(T, V))
        tr, rg = np.array(tr), np.array(rg)
        return float(ols(tr, rg)[0][1]), tr, rg

    scen = {"r = 0 (no temporal sensitivity)": 0.0,
            "r fixed at the observed mean regime AUROC": float(stats.norm.ppf(regime_obs.mean()) * tot)}
    res = {"observed_slope": b_obs, "variance_components": {"sigma_u2": sig_u2, "sigma_e2": sig_e2, "rho_across_models": rho},
           "n_sim": N_SIM, "scenarios": {}}
    for name, r in scen.items():
        sl = []
        trs = []
        for _ in range(N_SIM):
            b, tr, _ = panel(r)
            sl.append(b)
            trs.append(tr)
        sl = np.array(sl)
        res["scenarios"][name] = {
            "r": r, "slope_mean": float(sl.mean()), "slope_sd": float(sl.std()),
            "slope_2.5_97.5": [float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))],
            "share_slopes_le_observed": float((sl <= b_obs).mean()),
            "sim_truth_auroc_mean": float(np.mean(trs)), "obs_truth_auroc_mean": float(truth_obs.mean())}
    json.dump(res, open(OUT / "null_simulation.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
