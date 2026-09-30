"""Item 5: seed variance of the TSCT audit.

Route A (preferred): the team's LLaMA-3 predictions from HF DavidS64/llama3-hedge-sft
(FETCH_ALL.sh step 4 -> data/prep/llama3_hedge_sft/). Route B: local Qwen2.5-7B
seeds (run_local_seeds.sh -> data/prep/predictions_7b/{tsct,sft_only}_seed*_test.jsonl).

Checks, in order:
  1. Duplicates. Several exp3_v7 seed files have identical byte sizes on HF.
     Every prediction file is content-hashed, and so is its (question, answer,
     hedge) projection; seeds whose outputs are identical are not independent
     seeds and are reported, not averaged.
  2. Per run: n, accuracy, ECE (pipeline binned and per level), Brier,
     resolution, hedge distribution, AUROC of stated confidence for correctness.
  3. Per paired seed: TSCT - SFT difference in ECE and in resolution; mean,
     SD and a t-interval across distinct seeds.

Field names are taken from the local prediction format (predicted_hedge,
correct, gold_hedge); a file lacking them is reported and skipped.

Run from the repo root:
    python3 docs/orientation/results_sep10/item5/seed_variance.py --route A [--dir data/prep/llama3_hedge_sft/final_predictions]
    python3 docs/orientation/results_sep10/item5/seed_variance.py --route B
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from eval_pipeline import HEDGE_TO_CONFIDENCE as H, compute_ece  # noqa: E402

OUT = Path(__file__).resolve().parent


def metrics(recs):
    recs = [r for r in recs if r.get("predicted_hedge") in H and "correct" in r]
    if not recs:
        return None
    c = np.array([H[r["predicted_hedge"]] for r in recs])
    y = np.array([float(bool(r["correct"])) for r in recs])
    ob = y.mean()
    lv = np.unique(c)
    out = {"n": len(recs), "accuracy": float(ob),
           "ece_binned_10": compute_ece(recs)["ece"],
           "ece_by_level": float(sum((c == k).mean() * abs(y[c == k].mean() - k) for k in lv)),
           "brier": float(((c - y) ** 2).mean()),
           "resolution": float(sum((c == k).mean() * (y[c == k].mean() - ob) ** 2 for k in lv)),
           "hedge_dist": dict(Counter(r["predicted_hedge"] for r in recs))}
    out["auroc_conf_correct"] = (float(stats.mannwhitneyu(c[y == 1], c[y == 0]).statistic / (y.sum() * (1 - y).sum()))
                                 if 0 < y.sum() < len(y) and len(lv) > 1 else 0.5)
    out["reliability"] = float(sum((c == k).mean() * (k - y[c == k].mean()) ** 2 for k in lv))
    # selective risk-coverage: rank by stated confidence, ties by answer mean log-prob if present
    tie = np.array([r.get("mean_logprob") if r.get("mean_logprob") is not None else 0.0 for r in recs])
    order = np.lexsort((-tie, -c))
    risk = np.cumsum(1 - y[order]) / np.arange(1, len(y) + 1)
    out["aurc"] = float(risk.mean())
    out["selective_risk_at_50"] = float(risk[max(0, len(y) // 2 - 1)])
    return out


def parse(name):
    m = re.match(r"(exp\d)_(v\d+|fixed)_seed(\d+)_(.+?)_predictions$", name)
    if m:
        return {"arm": "sft" if m.group(1) == "exp2" else "tsct", "version": m.group(2),
                "seed": int(m.group(3)), "benchmark": m.group(4)}
    m = re.match(r"(tsct|sft_only)_seed(\d+)_test$", name)
    if m:
        return {"arm": "tsct" if m.group(1) == "tsct" else "sft", "version": "local",
                "seed": int(m.group(2)), "benchmark": "temporal_delta"}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=("A", "B"), required=True)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--pred-dir", default=None,
                    help="route B only: prediction dir (default data/prep/predictions_7b, the Qwen seeds); "
                         "data/prep/predictions_llama3 for the LLaMA-3 seeds")
    ap.add_argument("--tag", default=None, help="suffix for the output json")
    args = ap.parse_args()
    if args.route == "A":
        d = Path(args.dir or ROOT / "data/prep/llama3_hedge_sft/final_predictions")
        files = sorted(d.glob("*_predictions.jsonl"))
    else:
        d = Path(args.pred_dir) if args.pred_dir else ROOT / "data/prep/predictions_7b"
        files = sorted(d.glob("*_seed*_test.jsonl"))
        if not args.pred_dir:  # the Aug Qwen seeds 0-1 use older file names
            files += [d / "tsct_test.jsonl", d / "tsctS1_test.jsonl", d / "sft_test.jsonl", d / "sftS1_test.jsonl"]
    if not files:
        raise SystemExit(f"no prediction files under {d}: run FETCH_ALL.sh step 4 (route A) or run_local_seeds.sh (route B)")

    runs, hashes, proj = {}, defaultdict(list), defaultdict(list)
    for f in files:
        if not f.exists():
            continue
        raw = f.read_bytes()
        recs = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
        meta = parse(f.stem) or {"arm": "?", "version": "?", "seed": None, "benchmark": f.stem}
        if args.route == "B" and f.stem in ("tsct_test", "tsctS1_test", "sft_test", "sftS1_test"):
            meta = {"arm": "tsct" if f.stem.startswith("tsct") else "sft", "version": "local",
                    "seed": 1 if "S1" in f.stem else 0, "benchmark": "temporal_delta"}
        hashes[hashlib.sha256(raw).hexdigest()].append(f.name)
        key = hashlib.sha256(json.dumps([(r.get("question"), r.get("predicted_answer"), r.get("predicted_hedge"))
                                         for r in recs]).encode()).hexdigest()
        proj[key].append(f.name)
        runs[f.name] = meta | {"metrics": metrics(recs)}
    dups = [v for v in proj.values() if len(v) > 1]
    res = {"route": args.route, "n_files": len(runs), "identical_bytes": [v for v in hashes.values() if len(v) > 1],
           "identical_outputs": dups, "runs": runs, "paired": {}}

    dup_names = {n for g in dups for n in g[1:]}  # keep the first of each identical group
    grp = defaultdict(dict)
    for name, r in runs.items():
        if r["metrics"] and name not in dup_names:
            grp[(r["benchmark"], r["arm"], r["version"])][r["seed"]] = r["metrics"]
    for bench in sorted({k[0] for k in grp}):
        tsct = {k: v for k, v in grp.items() if k[0] == bench and k[1] == "tsct"}
        sft = {k: v for k, v in grp.items() if k[0] == bench and k[1] == "sft"}
        for tk, tv in tsct.items():
            for sk, sv in sft.items():
                seeds = sorted(set(tv) & set(sv))
                if len(seeds) < 2:
                    continue
                for metric in ("auroc_conf_correct", "resolution", "aurc", "selective_risk_at_50",
                               "ece_by_level", "reliability", "accuracy"):
                    d = np.array([tv[s][metric] - sv[s][metric] for s in seeds])
                    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
                    res["paired"][f"{bench}: {tk[2]} tsct - {sk[2]} sft: {metric}"] = {
                        "seeds": seeds, "diffs": d.tolist(), "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                        "ci95": [float(ci[0]), float(ci[1])],
                        "sign_consistent": bool((d > 0).all() or (d < 0).all())}
    # noise floor: seed SD of the plain-fine-tuning arm alone, per benchmark and endpoint
    res["sft_seed_sd"] = {}
    for (bench, arm, ver), seeds in grp.items():
        if arm == "sft" and len(seeds) >= 2:
            for metric in ("auroc_conf_correct", "resolution", "aurc", "ece_by_level"):
                res["sft_seed_sd"][f"{bench}: {ver}: {metric}"] = float(np.std([v[metric] for v in seeds.values()], ddof=1))
    # prespecified decision rule (ENDPOINTS.md), on the primary endpoint
    for k, v in res["paired"].items():
        if k.endswith("auroc_conf_correct"):
            bench = k.split(":")[0]
            sd = [x for kk, x in res["sft_seed_sd"].items() if kk.startswith(bench + ":") and kk.endswith("auroc_conf_correct")]
            v["effect_by_prespecified_rule"] = bool(len(v["seeds"]) >= 5 and v["sign_consistent"]
                                                    and (v["ci95"][0] > 0 or v["ci95"][1] < 0)
                                                    and sd and abs(v["mean"]) > 2 * min(sd))
    tag = "route" + args.route + (f"_{args.tag}" if args.tag else "")
    json.dump(res, open(OUT / f"seed_variance_{tag}.json", "w"), indent=2)
    print(f"files {len(runs)}; identical outputs: {dups}")
    for k, v in res["paired"].items():
        print(f"{k}: n={len(v['seeds'])} mean {v['mean']:+.4f} sd {v['sd']:.4f} "
              f"CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] sign-consistent {v['sign_consistent']}")


if __name__ == "__main__":
    main()
