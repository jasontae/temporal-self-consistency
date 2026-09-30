"""Table 1 AUROC column for the ARR paper: correctness AUROC of each Table 1 policy's
stated confidence (ties count half). Same records as ../ece_check/ece_levels.py.

    python3 docs/orientation/results_sep10/paper_tables/table1_auroc.py
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
PRED = ROOT / "data" / "prep" / "predictions_7b"
H = {"[CONFIDENT]": 0.95, "[COND_CONFIDENT]": 0.75, "[TEMPORAL_HEDGE]": 0.45, "[UNKNOWN]": 0.10}


def auroc(c, y):
    pos, neg = c[y == 1], c[y == 0]
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(f):
    R = [json.loads(l) for l in open(PRED / f)]
    return R, np.array([float(r["correct"]) for r in R])


out = {}
R, y = load("tsct_test.jsonl")
out["oracle (gold hedge)"] = auroc(np.array([H[r["gold_hedge"]] for r in R]), y)
out["trained model (TSCT seed 0, emitted hedge)"] = auroc(np.array([H.get(r["predicted_hedge"], 0.10) for r in R]), y)
out["constant [TEMPORAL_HEDGE]"] = 0.5
out["constant [UNKNOWN]"] = 0.5
R, y = load("sft_test.jsonl")
out["cross-entropy baseline (SFT seed 0, emitted hedge)"] = auroc(np.array([H.get(r["predicted_hedge"], 0.10) for r in R]), y)
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
print(json.dumps(out, indent=1))
