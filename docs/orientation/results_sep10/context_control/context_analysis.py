"""PLAN_context_control.md: staleness paired win 1[PMI(current) > PMI(expired)] under each context, pooled over
the nine admitted models, entity-cluster bootstrap (4,000 draws), and the verdict under the plan's rule.

    python3 docs/orientation/results_sep10/context_control/context_analysis.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
P = REPO / "data/prep/predictions_7b"
NINE = ["g3_4b", "qwen3_4b_th", "qwen3_4b_it", "qwen25_7b", "tsct", "gptoss_20b",
        "qwen36_35b", "qwen35_27b", "qwen36_27b_stock"]
CONDS = ["neutral", "current_evidence", "expired_evidence", "no_context"]
B = 4000


def main():
    win = defaultdict(dict)  # (tag, cond) -> {entity: 0/1}
    have = []
    for tag in NINE:
        f = P / f"context_{tag}.jsonl"
        if not f.exists():
            continue
        have.append(tag)
        d = defaultdict(dict)
        for r in map(json.loads, open(f)):
            d[(r["context_condition"], r["entity_id"])][r["arm"]] = r["pmi"]
        for (cond, e), v in d.items():
            if v.get("volatile_current") is not None and v.get("volatile_stale") is not None:
                win[(tag, cond)][e] = 1.0 if v["volatile_current"] > v["volatile_stale"] else 0.0
        base = P / f"disjoint4b_{tag}.jsonl"
        if base.exists():
            d0 = defaultdict(dict)
            for r in map(json.loads, open(base)):
                if r["arm"] in ("volatile_current", "volatile_stale"):
                    d0[r["entity_id"]][r["arm"]] = r["pmi"]
            for e, v in d0.items():
                if None not in (v.get("volatile_current"), v.get("volatile_stale")):
                    win[(tag, "no_context")][e] = 1.0 if v["volatile_current"] > v["volatile_stale"] else 0.0
    ents = sorted({e for k in win for e in win[k]})

    def pooled(cond, sample):
        xs = [win[(t, cond)][e] for e in sample for t in have if e in win[(t, cond)]]
        return sum(xs) / len(xs) if xs else None

    rng = random.Random(20260930)
    boot = defaultdict(list)
    for _ in range(B):
        smp = [rng.choice(ents) for _ in ents]
        vals = {c: pooled(c, smp) for c in CONDS}
        for c in CONDS:
            if vals[c] is not None:
                boot[c].append(vals[c])
        for c in ("current_evidence", "expired_evidence"):
            if vals[c] is not None and vals["neutral"] is not None:
                boot[c + "_minus_neutral"].append(vals[c] - vals["neutral"])

    def ci(xs):
        xs = sorted(xs)
        return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]

    res = {"models": have, "n_entities": len(ents), "win": {}, "diff": {}, "per_model": {}}
    for c in CONDS:
        v = pooled(c, ents)
        if v is not None:
            res["win"][c] = {"value": v, "ci": ci(boot[c])}
    for c in ("current_evidence", "expired_evidence"):
        k = c + "_minus_neutral"
        if boot[k]:
            res["diff"][k] = {"value": res["win"][c]["value"] - res["win"]["neutral"]["value"], "ci": ci(boot[k])}
    for t in have:
        res["per_model"][t] = {c: (sum(win[(t, c)].values()) / len(win[(t, c)]) if win[(t, c)] else None) for c in CONDS}
    cur = res["diff"].get("current_evidence_minus_neutral")
    exp = res["diff"].get("expired_evidence_minus_neutral")
    if len(have) < 9:
        verdict = f"INCOMPLETE ({len(have)}/9 models)"
    elif cur["value"] >= 0.20 and cur["ci"][0] > 0 and exp["value"] < 0:
        verdict = "INSTRUMENT PASSES (current evidence raises the win rate by >= 0.20, CI excludes 0; expired evidence lowers it)"
    elif abs(cur["value"]) < 0.05:
        verdict = "INSTRUMENT FAILS (current evidence moves the win rate by < 0.05)"
    else:
        verdict = "PARTIAL SENSITIVITY (between the pass and fail thresholds)"
    res["verdict"] = verdict
    json.dump(res, open(HERE / "context_results.json", "w"), indent=1)
    lines = [f"# In-context evidence control: {verdict}", "", f"Models: {', '.join(have)}; entities {len(ents)}.", "",
             "| context | pooled paired win 1[current > expired] [95% entity CI] |", "|---|---|"]
    for c, v in res["win"].items():
        lines.append(f"| {c} | {v['value']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] |")
    for k, v in res["diff"].items():
        lines.append(f"| {k} | {v['value']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] |")
    lines += ["", "Per model:", "", "| model | " + " | ".join(CONDS) + " |", "|---" * (len(CONDS) + 1) + "|"]
    for t, v in res["per_model"].items():
        lines.append(f"| {t} | " + " | ".join("—" if v[c] is None else f"{v[c]:.3f}" for c in CONDS) + " |")
    (HERE / "context_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
