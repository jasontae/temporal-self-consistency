"""Truth AUROC (PMI) per model, derangement build vs disjoint-pool build (A6),
with entity-paired bootstrap CIs on the change, and the 0.70 admission bar.

Run from the repo root:
    python3 docs/orientation/results_sep10/a6_disjoint/truth_before_after.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/orientation/results_sep10/inverse"))
from inverse_truth import load, auroc  # noqa: E402

P = ROOT / "data/prep/predictions_7b"
NAMES = {"g3_4b": "gemma-3-4B", "qwen3_4b_th": "Qwen3-4B-Thinking", "qwen3_4b_it": "Qwen3-4B-Instruct",
         "gptoss_20b": "gpt-oss-20B", "qwen36_35b": "Qwen3.6-35B-A3B", "qwen35_27b": "Qwen3.5-27B",
         "qwen36_27b_stock": "Qwen3.6-27B", "qwen38_27b": "Qwen3.8-27B (item 6)",
         "g4_e4b": "gemma-4-E4B (dropped)", "gemma4_26b": "gemma-4-26B-A4B (dropped)", "g4_31b": "gemma-4-31B (dropped)",
         "qwen25_7b": "Qwen2.5-7B", "tsct": "trained model"}
rng = np.random.default_rng(20260928)
rows, out = [], {}
for k, lab in NAMES.items():
    a, b = P / f"matched_{k}.jsonl", P / f"disjoint_{k}.jsonl"
    if not a.exists() or not b.exists():
        rows.append(f"| {lab} | {'—' if not a.exists() else ''} | not rescored (base checkpoint missing; FETCH_ALL step 1) | | |")
        continue
    A, B = load(a), load(b)
    ents = sorted(set(A) & set(B))
    T = np.array([A[e]["immutable_true"] for e in ents])
    F0 = np.array([A[e]["immutable_false"] for e in ents])
    F1 = np.array([B[e]["immutable_false"] for e in ents])
    t0, t1 = auroc(T, F0), auroc(T, F1)
    d = []
    for _ in range(4000):
        ix = rng.integers(0, len(ents), len(ents))
        d.append(auroc(T[ix], F1[ix]) - auroc(T[ix], F0[ix]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    out[k] = {"derangement": t0, "disjoint": t1, "delta": t1 - t0, "ci95": [lo, hi], "passes_0.70": bool(t1 >= 0.70)}
    rows.append(f"| {lab} | {t0:.3f} | {t1:.3f} | {t1 - t0:+.3f} [{lo:+.3f}, {hi:+.3f}] | {'yes' if t1 >= 0.70 else 'no'} |")
hdr = ["| model | truth AUROC, derangement (Aug 13) | truth AUROC, disjoint pool | change [95% CI] | ≥ 0.70 |", "|---|---|---|---|---|"]
(Path(__file__).parent / "truth_before_after.md").write_text("\n".join(hdr + rows) + "\n")
json.dump(out, open(Path(__file__).parent / "truth_before_after.json", "w"), indent=2)
print("\n".join(hdr + rows))
