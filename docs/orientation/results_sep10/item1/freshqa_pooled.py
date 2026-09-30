"""FreshQA staleness pooled over models, with a question-level bootstrap (the same
resample across models) -- the analogue of a5_prominence/staleness_estimands.py.
Also restricted to the models that pass the FreshQA truth check (>= 0.70).

Run from the repo root:
    python3 docs/orientation/results_sep10/item1/freshqa_pooled.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent
res_all = json.load(open(OUT / "freshqa_results.json"))["contrasts"]


def auroc(pos, neg):
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


data = {}
for f in sorted(P.glob("fq_*.jsonl")):
    by = defaultdict(dict)
    for r in map(json.loads, open(f)):
        by[r["entity_id"]][r["arm"]] = r["pmi"]
    data[f.stem] = {q: (a["fq_current"], a["fq_stale"]) for q, a in by.items() if "fq_current" in a and "fq_stale" in a}
qs = sorted(set.intersection(*(set(v) for v in data.values())))
rng = np.random.default_rng(20260928)
out = {}
for label, models in (("all", list(data)), ("truth >= 0.70", [m for m in data if res_all[m]["admitted_truth_ge_0.70"]])):
    A = {m: np.array([data[m][q] for q in qs]) for m in models}
    pt = np.mean([auroc(A[m][:, 0], A[m][:, 1]) for m in models])
    win = np.mean([(A[m][:, 0] > A[m][:, 1]).mean() for m in models])
    bs, bw = [], []
    for _ in range(4000):
        ix = rng.integers(0, len(qs), len(qs))
        bs.append(np.mean([auroc(A[m][ix, 0], A[m][ix, 1]) for m in models]))
        bw.append(np.mean([(A[m][ix, 0] > A[m][ix, 1]).mean() for m in models]))
    below = sum(auroc(A[m][:, 0], A[m][:, 1]) < 0.5 for m in models)
    out[label] = {"models": models, "n_questions": len(qs), "mean_auroc": float(pt),
                  "ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
                  "p_boot": float(min(1, 2 * min((np.array(bs) >= 0.5).mean(), (np.array(bs) <= 0.5).mean()))),
                  "mean_paired_win": float(win), "paired_win_ci95": np.percentile(bw, [2.5, 97.5]).tolist(),
                  "models_below_0.5": int(below), "of": len(models)}
json.dump(out, open(OUT / "freshqa_pooled.json", "w"), indent=2)
print(json.dumps(out, indent=1))
