"""Table 2: rebuild the repaired benchmark and rescore it.

Repair, as described in specs/CLAIMS-LEDGER.md F-1 and Sep 2 section 4.2:
  C5  replace each gold answer with the live Wikidata current value
      (data/prep/gold_currency_audit.json, `wikidata_current_label`; only pairs
      that have one)
  C7  drop every test question that also appears in the train split
Expected: 1,953 items. Existing predictions are rescored; nothing is regenerated.

Correctness uses the pipeline's own exact_match (src/evaluation/eval_pipeline.py).
ECE is reported binned (the pipeline's compute_ece) and per level; Brier and the
Murphy decomposition are exact by level.

Inputs:
  data/prep/predictions_7b/{tsct,sft}_test.jsonl
  data/prep/gold_currency_audit.json
  data/prep/temporal_delta/temporal_delta_train.jsonl   (FETCH_ALL.sh step 2)

Run from the repo root:
    python3 docs/orientation/results_sep10/table2/rebuild_repaired.py
    python3 docs/orientation/results_sep10/table2/rebuild_repaired.py --no-train-filter   # C5 only (dry run)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from eval_pipeline import HEDGE_TO_CONFIDENCE as H, compute_ece, exact_match  # noqa: E402

PRED = ROOT / "data" / "prep" / "predictions_7b"
TRAIN = ROOT / "data" / "prep" / "temporal_delta" / "temporal_delta_train.jsonl"
AUDIT = ROOT / "data" / "prep" / "gold_currency_audit.json"
OUT = Path(__file__).resolve().parent
PAPER_T2 = {"cross-entropy baseline": (0.0046, 0.4768, 0.231, 0.001),
            "trained model": (0.0026, 0.4789, 0.233, 0.001),
            "volatility-label oracle": (None, 0.4788, 0.232, 0.001),
            "constant [UNKNOWN]": (None, 0.0974, 0.012, 0.000)}


def by_level(c, y):
    ob = y.mean()
    lv = np.unique(c)
    ece = sum((c == k).mean() * abs(y[c == k].mean() - k) for k in lv)
    rel = sum((c == k).mean() * (k - y[c == k].mean()) ** 2 for k in lv)
    res = sum((c == k).mean() * (y[c == k].mean() - ob) ** 2 for k in lv)
    auc = 0.5
    if 0 < y.sum() < len(y) and len(lv) > 1:
        from scipy import stats
        auc = float(stats.mannwhitneyu(c[y == 1], c[y == 0]).statistic / (y.sum() * (len(y) - y.sum())))
    return {"ece_by_level": float(ece), "brier": float(((c - y) ** 2).mean()),
            "reliability": float(rel), "resolution": float(res), "uncertainty": float(ob * (1 - ob)),
            "auroc_conf_correct": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train-filter", action="store_true")
    args = ap.parse_args()

    current = {}
    for r in json.load(open(AUDIT))["rows"]:
        if r.get("wikidata_current_label"):
            current[(r["question"], r["dataset_gold"])] = r["wikidata_current_label"]
    train_q = set()
    if not args.no_train_filter:
        if not TRAIN.exists():
            raise SystemExit(f"{TRAIN} missing: run docs/orientation/FETCH_ALL.sh step 2, or pass --no-train-filter")
        train_q = {json.loads(l)["question"] for l in open(TRAIN)}

    out = {"train_filter": not args.no_train_filter, "arms": {}}
    rows_md = []
    for fname, label in (("sft_test", "cross-entropy baseline"), ("tsct_test", "trained model")):
        recs = [json.loads(l) for l in open(PRED / f"{fname}.jsonl")]
        kept, seen = [], set()
        for r in recs:
            cur = current.get((r["question"], r["gold_answer"]))
            if cur is None or r["question"] in train_q:
                continue
            kept.append(dict(r, gold_answer=cur, correct=exact_match(r["predicted_answer"], cur)))
            seen.add(r["question"])
        y = np.array([float(r["correct"]) for r in kept])
        pols = {label: [H[r["predicted_hedge"]] for r in kept],
                "volatility-label oracle": [H[r["gold_hedge"]] for r in kept],
                "constant [TEMPORAL_HEDGE]": [0.45] * len(kept),
                "constant [UNKNOWN]": [0.10] * len(kept)}
        arm = {"n_items": len(kept), "n_unique_questions": len(seen), "accuracy": float(y.mean()), "policies": {}}
        for name, c in pols.items():
            c = np.array(c)
            tagged = [dict(r, predicted_hedge={0.95: "[CONFIDENT]", 0.75: "[COND_CONFIDENT]", 0.45: "[TEMPORAL_HEDGE]",
                                                0.10: "[UNKNOWN]"}[float(k)]) for r, k in zip(kept, c)]
            m = by_level(c, y) | {"ece_binned_10": compute_ece(tagged)["ece"]}
            arm["policies"][name] = m
            if name != label and fname == "sft_test" and name != "constant [TEMPORAL_HEDGE]":
                continue  # oracle / constants reported once, from the tsct arm's correctness
            p = PAPER_T2.get(name)
            rows_md.append(f"| {name} ({fname}) | {len(kept)} | {y.mean():.4f} | {m['ece_binned_10']:.4f} | {m['ece_by_level']:.4f} | "
                           f"{m['brier']:.4f} | {m['reliability']:.4f} | {m['resolution']:.6f} | {m['auroc_conf_correct']:.3f} | {p if p else ''} |")
        # oracle / constant [UNKNOWN] ratio with a record bootstrap
        o, u = np.array(pols["volatility-label oracle"]), np.full(len(kept), 0.10)
        rng = np.random.default_rng(20260928)
        rs = []
        for _ in range(10000):
            ix = rng.integers(0, len(y), len(y))
            rs.append(by_level(o[ix], y[ix])["ece_by_level"] / by_level(u[ix], y[ix])["ece_by_level"])
        arm["oracle_over_unknown"] = arm["policies"]["volatility-label oracle"]["ece_by_level"] / arm["policies"]["constant [UNKNOWN]"]["ece_by_level"]
        arm["oracle_over_unknown_ci95"] = np.percentile(rs, [2.5, 97.5]).tolist()
        out["arms"][fname] = arm
    tag = "c5_only" if args.no_train_filter else "full"
    json.dump(out, open(OUT / f"table2_{tag}.json", "w"), indent=2)
    hdr = ["| policy (correctness from) | items | accuracy | ECE binned | ECE by level | Brier | reliability | resolution | AUROC | Sep 2 (acc, ECE, Brier, res) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    (OUT / f"table2_{tag}.md").write_text("\n".join(hdr + rows_md) + "\n")
    print("\n".join(hdr + rows_md))
    print({k: (v["n_items"], v["n_unique_questions"], round(v["oracle_over_unknown"], 3),
               [round(x, 2) for x in v["oracle_over_unknown_ci95"]]) for k, v in out["arms"].items()})


if __name__ == "__main__":
    main()
