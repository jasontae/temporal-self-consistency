"""Recompute the Sep 2 section 4.3 paired bootstrap: model AUROC minus the
person-name regex AUROC on the 22 distinct withdrawn-set claims (Qwen2.5-7B,
the best model there, 0.9000). Sep 2 prints dAUROC = +0.067 [0.000, 0.182],
p = 0.24; no script for it exists in the repo.

Run from the repo root:
    python3 docs/orientation/results_sep10/provenance/paired_bootstrap_a3.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.evaluation.b11_dedup import distinct_claims  # noqa: E402
from src.evaluation.b11_surface_control import auroc, has_person_name  # noqa: E402

rows = [json.loads(l) for l in open(ROOT / "data/prep/predictions_7b/mixed_qwen25_7b.jsonl")]
u = distinct_claims(rows)
lab = [not r["is_volatile"] for r in u]
m = [r["mean_logprob"] for r in u]
g = [float(not has_person_name(r["text"])) for r in u]


def diff(idx):
    L = [lab[i] for i in idx]
    if all(L) or not any(L):
        return None
    return auroc([(m[i], L[j]) for j, i in enumerate(idx)]) - auroc([(g[i], L[j]) for j, i in enumerate(idx)])


obs = diff(range(len(u)))
rng = np.random.default_rng(20260928)
d = [x for x in (diff(rng.integers(0, len(u), len(u))) for _ in range(10000)) if x is not None]
d = np.array(d)
out = {"n_distinct": len(u), "n_volatile": int(sum(not x for x in lab)), "delta_auroc": obs,
       "ci95_percentile": np.percentile(d, [2.5, 97.5]).tolist(),
       "p_two_sided_sign": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
       "frac_boot_le_0": float((d <= 0).mean()), "n_valid_boot": int(len(d))}
print(json.dumps(out, indent=2))
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=2)

# Stratified variant: resample volatile and stable claims separately, keeping
# 10 and 12, which is the other common scheme for AUROC bootstraps.
pos = [i for i, x in enumerate(lab) if x]
neg = [i for i, x in enumerate(lab) if not x]
ds = np.array([diff(list(rng.choice(pos, len(pos))) + list(rng.choice(neg, len(neg))))
               for _ in range(10000)])
out["stratified"] = {"ci95_percentile": np.percentile(ds, [2.5, 97.5]).tolist(),
                     "p_two_sided_sign": float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean())))}
print(json.dumps(out["stratified"], indent=2))
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=2)
