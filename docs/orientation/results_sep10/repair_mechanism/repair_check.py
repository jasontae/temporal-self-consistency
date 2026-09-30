"""Directive 3 (worker B): why does repairing the answer key make every ECE worse?

Claim: when every bin is overconfident (checked in ../ece_check/), ECE of any policy
equals mean asserted confidence minus accuracy, ECE = m - a. It does not depend on
which items are correct. So the repair, which lowers accuracy a, adds the accuracy
drop to every policy's ECE, and the oracle/constant ratio (m_oracle - a)/(0.10 - a)
falls toward its a -> 0 limit m_oracle / 0.10.

Checks:
  1. On the released key, ECE(pipeline) == m - a for oracle, TSCT, SFT, and constants.
  2. On the repaired key (Sep 2 Table 2; the set itself needs FETCH_ALL step 2), the
     printed numbers satisfy the same identity: constant = 0.10 - a, oracle = m - a,
     with the implied class mix of the 1,953 items.

    python3 docs/orientation/results_sep10/repair_mechanism/repair_check.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/orientation/results_sep10/vocabulary"))
from vocab_sweep import H, PRED, ece_eqfreq  # noqa: E402

OUT = Path(__file__).resolve().parent
rows = []
for f, arm in (("tsct_test", "TSCT seed 0"), ("sft_test", "SFT (cross-entropy) seed 0")):
    R = [json.loads(l) for l in open(PRED / f"{f}.jsonl")]
    y = np.array([float(r["correct"]) for r in R])
    pols = {f"{arm}, emitted hedge": np.array([H.get(r["predicted_hedge"], 0.10) for r in R])}
    if f == "tsct_test":
        pols["volatility-label oracle (gold hedge)"] = np.array([H[r["gold_hedge"]] for r in R])
        pols["constant [TEMPORAL_HEDGE]"] = np.full(len(y), 0.45)
        pols["constant [UNKNOWN]"] = np.full(len(y), 0.10)
    for k, c in pols.items():
        rows.append({"policy": k, "accuracy": float(y.mean()), "mean_conf": float(c.mean()),
                     "m_minus_a": float(c.mean() - y.mean()), "ece_pipeline": float(ece_eqfreq(c, y))})

gold = [json.loads(l)["gold_hedge"] for l in open(PRED / "tsct_test.jsonl")]
m_oracle = float(np.mean([H[g] for g in gold]))
# repaired key, printed Sep 2 Table 2 values
a_trained, a_ce = 0.0026, 0.0046
ece_const_rep, ece_oracle_rep, ece_trained_rep, ece_ce_rep = 0.0974, 0.4788, 0.4789, 0.4768
rep = {
    "constant: 0.10 - a (trained-arm accuracy)": (0.10 - a_trained, ece_const_rep),
    "implied mean oracle confidence on the 1,953 items: ECE_oracle + a": (ece_oracle_rep + a_trained, None),
    "implied share of [COND_CONFIDENT] items: (m - 0.45) / 0.30": ((ece_oracle_rep + a_trained - 0.45) / 0.30, None),
    "ratio oracle/constant, printed": (ece_oracle_rep / ece_const_rep, None),
    "ratio limit as a -> 0 on the released mix: m_oracle / 0.10": (m_oracle / 0.10, None),
    "ratio on released key: (m - a)/(0.10 - a)": ((m_oracle - 0.0273) / (0.10 - 0.0273), None),
}
out = {"released_key_identity": rows, "m_oracle_released": m_oracle,
       "repaired_key": {k: {"value": v[0], "printed": v[1]} for k, v in rep.items()}}
json.dump(out, open(OUT / "repair_check.json", "w"), indent=1)
print("released key: policy | accuracy | mean conf | m - a | ECE (pipeline)")
for r in rows:
    print(f"  {r['policy']:40s} {r['accuracy']:.4f} {r['mean_conf']:.4f} {r['m_minus_a']:.4f} {r['ece_pipeline']:.4f}")
print(f"m_oracle (released) = {m_oracle:.4f}")
for k, v in rep.items():
    print(f"  {k}: {v[0]:.4f}" + (f"  (printed {v[1]:.4f})" if v[1] is not None else ""))
