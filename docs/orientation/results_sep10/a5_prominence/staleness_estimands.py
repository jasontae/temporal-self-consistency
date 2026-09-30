"""The staleness headline under three estimands, all with the same entity
bootstrap (entities resampled jointly across models; 106 entities):
  unpaired AUROC(current vs stale), mean over models
  paired win rate 1[pmi(current) > pmi(stale)], mean over models
  mean PMI difference (current - stale), z-scored within model, mean over models
and the across-model sign test the Sep 2 PDF reports (models as independent units).

Run from the repo root:
    python3 docs/orientation/results_sep10/a5_prominence/staleness_estimands.py [--with-e4b]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent))
from a5_prominence import MODELS, auroc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--with-e4b", action="store_true")
args = ap.parse_args()
models = MODELS + (["g4_e4b"] if args.with_e4b else [])
V, St = {}, {}
for m in models:
    by = defaultdict(dict)
    for line in open(ROOT / f"data/prep/predictions_7b/matched_{m}.jsonl"):
        r = json.loads(line)
        by[r["entity_id"]][r["arm"]] = r["pmi"]
    ents = sorted(by)
    V[m] = np.array([by[e]["volatile_current"] for e in ents])
    St[m] = np.array([by[e]["volatile_stale"] for e in ents])
n = len(ents)
sd = {m: np.concatenate([V[m], St[m]]).std() for m in models}


def est(ix):
    return (np.mean([auroc(V[m][ix], St[m][ix]) for m in models]),
            np.mean([(V[m][ix] > St[m][ix]).mean() for m in models]),
            np.mean([((V[m][ix] - St[m][ix]) / sd[m]).mean() for m in models]))


rng = np.random.default_rng(20260928)
pt = est(np.arange(n))
bs = np.array([est(rng.integers(0, n, n)) for _ in range(4000)])
out = {}
for i, (name, null) in enumerate((("unpaired AUROC", 0.5), ("paired win rate", 0.5), ("z PMI difference", 0.0))):
    b = bs[:, i]
    out[name] = {"point": float(pt[i]), "ci95": np.percentile(b, [2.5, 97.5]).tolist(),
                 "p_boot_two_sided": float(min(1, 2 * min((b >= null).mean(), (b <= null).mean())))}
below = sum(auroc(V[m], St[m]) < 0.5 for m in models)
out["across-model sign test (Sep 2 style)"] = {"below": int(below), "of": len(models),
                                                "p": float(stats.binomtest(below, len(models)).pvalue)}
tag = "with_e4b" if args.with_e4b else "nine"
json.dump(out, open(Path(__file__).parent / f"staleness_estimands_{tag}.json", "w"), indent=2)
print(json.dumps(out, indent=1))
