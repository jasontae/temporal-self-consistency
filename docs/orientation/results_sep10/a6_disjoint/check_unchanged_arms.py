"""The A6 rebuild changes only the immutable_false arm. The other three arms
must reproduce the Aug 13 scores exactly; any drift means the harness changed.

Run from the repo root:
    python3 docs/orientation/results_sep10/a6_disjoint/check_unchanged_arms.py
"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data/prep/predictions_7b"
out = {}
for f in sorted(glob.glob(str(P / "disjoint_*.jsonl"))):
    name = Path(f).stem.replace("disjoint_", "")
    old = P / f"matched_{name}.jsonl"
    if not old.exists():
        continue
    a = {(r["entity_id"], r["arm"]): r for r in map(json.loads, open(old))}
    b = {(r["entity_id"], r["arm"]): r for r in map(json.loads, open(f))}
    d = [abs(a[k]["mean_logprob"] - b[k]["mean_logprob"]) for k in b if k[1] != "immutable_false"]
    dn = [abs(a[k]["name_logprob"] - b[k]["name_logprob"]) for k in b if k[1] != "immutable_false"]
    out[name] = {"n": len(d), "max_abs_diff_claim": max(d), "max_abs_diff_name": max(dn)}
json.dump(out, open(Path(__file__).parent / "unchanged_arms.json", "w"), indent=2)
for k, v in out.items():
    print(f"{k:20s} n={v['n']}  max|Δ| claim {v['max_abs_diff_claim']:.2e}  name {v['max_abs_diff_name']:.2e}")
