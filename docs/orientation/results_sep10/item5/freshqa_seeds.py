"""Item 5, second benchmark: the prespecified endpoints (ENDPOINTS.md) on FreshQA,
for the five paired seeds per arm. Correctness = containment of any accepted
answer under the current (2026-04-21) and the aged (2024-02-26) key, since the
generator's own `correct` compares against one exact string.

Run from the repo root:
    python3 docs/orientation/results_sep10/item5/freshqa_seeds.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/orientation/results_sep10/item5"))
sys.path.insert(0, str(ROOT / "src/evaluation"))
from seed_variance import metrics  # noqa: E402
from eval_pipeline import normalize_answer as N  # noqa: E402

P = ROOT / "data/prep/predictions_7b"
built = {r["id"]: r for r in map(json.loads, open(ROOT / "data/prep/freshqa/freshqa_built.jsonl"))}
RUNS = {"tsct": {0: "tsct_seed0_paired", 1: "tsct_seed1", 2: "tsct_seed2_r2", 3: "tsct_seed3_r2", 4: "tsct_seed4_r2"},
        "sft": {0: "sft_only_seed0", 1: "sft_only_seed1", 2: "sft_only_seed2_r2", 3: "sft_only_seed3_r2", 4: "sft_only_seed4_r2"}}
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--llama3", action="store_true", help="the LLaMA-3 seeds (data/prep/predictions_llama3)")
args = ap.parse_args()
if args.llama3:
    P = ROOT / "data/prep/predictions_llama3"
    RUNS = {arm: {s: f"{'tsct' if arm == 'tsct' else 'sft_only'}_seed{s}" for s in range(5)} for arm in ("tsct", "sft")}
    SUFFIX, OUTNAME = "_freshqa", "freshqa_seeds_llama3.json"
else:
    SUFFIX, OUTNAME = "", "freshqa_seeds.json"
out = {}
for key in ("current", "aged"):
    m = {}
    for arm, seeds in RUNS.items():
        for s, name in seeds.items():
            f = (P / f"{name}{SUFFIX}.jsonl") if args.llama3 else (P / f"freshqa_{name}.jsonl")
            recs = [json.loads(l) for l in open(f)]
            for r in recs:
                ans = built[r["freshqa_id"]][f"{key}_answers"]
                p = N(r["predicted_answer"])
                r["correct"] = any(N(a) and N(a) in p for a in ans)
            m[(arm, s)] = metrics(recs)
    res = {}
    for ep in ("auroc_conf_correct", "resolution", "aurc", "selective_risk_at_50", "ece_by_level", "reliability", "accuracy"):
        d = np.array([m[("tsct", s)][ep] - m[("sft", s)][ep] for s in range(5)])
        ci = stats.t.interval(0.95, 4, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(5))
        res[ep] = {"diffs": d.tolist(), "mean": float(d.mean()), "sd": float(d.std(ddof=1)), "ci95": [float(ci[0]), float(ci[1])],
                   "sign_consistent": bool((d > 0).all() or (d < 0).all()),
                   "sft_seed_sd": float(np.std([m[("sft", s)][ep] for s in range(5)], ddof=1))}
    p = res["auroc_conf_correct"]
    res["effect_by_prespecified_rule"] = bool(p["sign_consistent"] and (p["ci95"][0] > 0 or p["ci95"][1] < 0)
                                              and abs(p["mean"]) > 2 * p["sft_seed_sd"])
    res["per_run"] = {f"{a}_s{s}": v for (a, s), v in m.items()}
    out[key] = res
json.dump(out, open(Path(__file__).parent / OUTNAME, "w"), indent=2)
for key, res in out.items():
    print(f"## FreshQA, {key} key; effect by rule: {res['effect_by_prespecified_rule']}")
    for ep in ("auroc_conf_correct", "resolution", "aurc", "ece_by_level", "accuracy"):
        v = res[ep]
        print(f"  {ep:20s} mean {v['mean']:+.4f} CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] sign-consistent {v['sign_consistent']} sft-sd {v['sft_seed_sd']:.4f}")
