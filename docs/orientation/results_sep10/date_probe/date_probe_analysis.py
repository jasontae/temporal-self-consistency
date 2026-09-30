"""Directive 3, Worker A item 6: does the stale-value preference survive date conditioning?

For each admitted model (4-bit): staleness AUROC (current vs stale holder, PMI,
the name frame unconditioned) with no context (disjoint4b_) and after each
context prefix (date_<ctx>_). Per model and pooled over models: the change from
no context, and the key contrast 2020 - 2026, with an entity bootstrap (the same
resample across models). If models condition on the stated date, "As of 2026,"
should raise staleness AUROC (favour the current holder) and "As of 2020," should
lower it.

Run from the repo root:
    python3 docs/orientation/results_sep10/date_probe/date_probe_analysis.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent
MODELS = ["g3_4b", "qwen3_4b_th", "qwen3_4b_it", "qwen25_7b", "tsct", "gptoss_20b", "qwen36_35b", "qwen35_27b", "qwen36_27b_stock"]
CTX = {"none": "disjoint4b_", "asof2026": "date_asof2026_", "in2026": "date_in2026_", "asof2020": "date_asof2020_"}


def load(p):
    by = defaultdict(dict)
    for r in map(json.loads, open(p)):
        by[r["entity_id"]][r["arm"]] = r["pmi"]
    return by


def auroc(pos, neg):
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    data = {}
    for m in MODELS:
        d = {}
        for k, pre in CTX.items():
            f = P / f"{pre}{m}.jsonl"
            if f.exists():
                d[k] = load(f)
        if len(d) == len(CTX):
            data[m] = d
    if not data:
        raise SystemExit("no complete date-probe files yet")
    ents = sorted(set.intersection(*(set(v[k]) for v in data.values() for k in CTX)))
    V = {m: {k: np.array([data[m][k][e]["volatile_current"] for e in ents]) for k in CTX} for m in data}
    S = {m: {k: np.array([data[m][k][e]["volatile_stale"] for e in ents]) for k in CTX} for m in data}
    rng = np.random.default_rng(20260928)
    boots = rng.integers(0, len(ents), (4000, len(ents)))
    st = lambda m, k, ix=slice(None): auroc(V[m][k][ix], S[m][k][ix])
    res = {"models": list(data), "n_entities": len(ents), "per_model": {}, "pooled": {}}
    for m in data:
        res["per_model"][m] = {k: float(st(m, k)) for k in CTX}
        res["per_model"][m]["win_rate"] = {k: float((V[m][k] > S[m][k]).mean()) for k in CTX}
    for name, (a, b) in {"asof2026 - none": ("asof2026", "none"), "in2026 - none": ("in2026", "none"),
                         "asof2020 - none": ("asof2020", "none"), "asof2020 - asof2026": ("asof2020", "asof2026")}.items():
        pt = np.mean([st(m, a) - st(m, b) for m in data])
        bs = np.array([np.mean([st(m, a, ix) - st(m, b, ix) for m in data]) for ix in boots])
        per = [st(m, a) - st(m, b) for m in data]
        res["pooled"][name] = {"mean_diff": float(pt), "ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
                               "p": float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))),
                               "models_positive": int(sum(x > 0 for x in per)), "of": len(per)}
    for k in CTX:
        bs = np.array([np.mean([st(m, k, ix) for m in data]) for ix in boots])
        res["pooled"][f"mean staleness, {k}"] = {"mean": float(np.mean([st(m, k) for m in data])),
                                                 "ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
                                                 "excludes_0.5": bool(np.percentile(bs, 97.5) < 0.5 or np.percentile(bs, 2.5) > 0.5)}
    json.dump(res, open(OUT / "date_probe.json", "w"), indent=2)
    lines = ["| model | none | As of 2026 | In 2026 | As of 2020 |", "|---|---|---|---|---|"]
    for m, v in res["per_model"].items():
        lines.append(f"| {m} | {v['none']:.3f} | {v['asof2026']:.3f} | {v['in2026']:.3f} | {v['asof2020']:.3f} |")
    lines += ["", "| pooled over models | value [95% entity-bootstrap CI] | p | models with positive change |", "|---|---|---|---|"]
    for k, v in res["pooled"].items():
        if "mean_diff" in v:
            lines.append(f"| {k} | {v['mean_diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['p']:.3f} | {v['models_positive']}/{v['of']} |")
        else:
            lines.append(f"| {k} | {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | excludes 0.5: {v['excludes_0.5']} | |")
    (OUT / "date_probe_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
