"""Item 5 on LLaMA-3-8B: the prespecified TSCT endpoints (../item5/ENDPOINTS.md), same
metrics and intervals as the Qwen five-seed result (../item5/seed_variance.py --route B).

For every complete seed pair (TSCT and plain fine-tuning, same seed) in
data/prep/predictions_llama3/:
  TemporalDelta test (3,622): per arm-seed accuracy, ECE (per level and binned), Brier,
      reliability, resolution, AUROC(stated confidence -> correct), AURC, selective risk
      at 50%, hedge-label accuracy (predicted == gold hedge), hedge distribution; the
      volatility-label oracle and the constant policies on the same correctness, and the
      oracle / constant [UNKNOWN] ECE ratio.
  Paired TSCT - plain FT differences over seeds with 95% t-intervals, sign consistency,
      the plain-FT seed SD (noise floor) and the prespecified effect rule.
  FreshQA (413; containment scoring, current and aged key): the same paired endpoints.
Sanity checks: no byte-identical predictions or adapters across arm-seeds, 3,622 test
predictions per arm-seed, tokenizer round-trip on the base actually used.

Qwen2.5-7B's five-seed numbers (../item5/seed_variance_routeB.json, freshqa_seeds.json)
are carried alongside for comparison.

Usage (from repo root):
    python3 docs/orientation/results_sep10/item5_llama3/llama3_analysis.py [--tag dryrun]
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "docs/orientation/results_sep10/item5"))
sys.path.insert(0, str(ROOT / "src/evaluation"))
from seed_variance import metrics  # noqa: E402  (the Qwen analysis's own endpoint function)
from eval_pipeline import HEDGE_TO_CONFIDENCE as H, compute_ece, normalize_answer as N  # noqa: E402

P = ROOT / "data/prep/predictions_llama3"
T = ROOT / "data/prep/tcl_mlx_llama3"
BASE = Path.home() / ".oMLX/models/Meta-Llama-3-8B-4bit-tokfix"
OUT = Path(__file__).resolve().parent
ARMS = {"tsct": "tsct", "sft": "sft_only"}
EPS = ("auroc_conf_correct", "resolution", "ece_by_level", "reliability", "brier", "aurc",
       "selective_risk_at_50", "accuracy", "hedge_label_accuracy")
PRIMARY = "auroc_conf_correct"
FRESH = {r["id"]: r for r in map(json.loads, open(ROOT / "data/prep/freshqa/freshqa_built.jsonl"))}


def by_level(c, y):
    lv = np.unique(c)
    ob = y.mean()
    return {"ece_by_level": float(sum((c == k).mean() * abs(y[c == k].mean() - k) for k in lv)),
            "resolution": float(sum((c == k).mean() * (y[c == k].mean() - ob) ** 2 for k in lv)),
            "brier": float(((c - y) ** 2).mean())}


def controls(recs):
    y = np.array([float(bool(r["correct"])) for r in recs])
    out = {}
    for name, c in (("oracle", np.array([H[r["gold_hedge"]] for r in recs])),
                    ("constant [TEMPORAL_HEDGE]", np.full(len(y), 0.45)),
                    ("constant [UNKNOWN]", np.full(len(y), 0.10))):
        out[name] = by_level(c, y)
    out["oracle_over_unknown"] = out["oracle"]["ece_by_level"] / out["constant [UNKNOWN]"]["ece_by_level"]
    return out


def arm_metrics(recs):
    m = metrics(recs)
    m["hedge_label_accuracy"] = float(np.mean([r["predicted_hedge"] == r["gold_hedge"] for r in recs]))
    m["brier"] = float(np.mean([(H[r["predicted_hedge"]] - float(bool(r["correct"]))) ** 2 for r in recs]))
    return m


def paired(res, seeds, eps):
    out = {}
    for ep in eps:
        d = np.array([res[("tsct", s)][ep] - res[("sft", s)][ep] for s in seeds])
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
        out[ep] = {"per_seed": dict(zip(map(str, seeds), d.tolist())), "mean": float(d.mean()),
                   "sd": float(d.std(ddof=1)), "ci95": [float(ci[0]), float(ci[1])],
                   "sign_consistent": bool((d > 0).all() or (d < 0).all()),
                   "sft_seed_sd": float(np.std([res[("sft", s)][ep] for s in seeds], ddof=1)),
                   "tsct_mean": float(np.mean([res[("tsct", s)][ep] for s in seeds])),
                   "sft_mean": float(np.mean([res[("sft", s)][ep] for s in seeds]))}
    p = out[PRIMARY]
    out["effect_by_prespecified_rule"] = bool(len(seeds) >= 5 and p["sign_consistent"]
                                              and (p["ci95"][0] > 0 or p["ci95"][1] < 0)
                                              and abs(p["mean"]) > 2 * p["sft_seed_sd"])
    return out


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    a = ap.parse_args()
    seeds = [s for s in range(5) if all((P / f"{ARMS[k]}_seed{s}_test.jsonl").exists()
                                        and (P / f"{ARMS[k]}_seed{s}_freshqa.jsonl").exists() for k in ARMS)]
    if len(seeds) < 2:
        raise SystemExit(f"need >= 2 complete seed pairs, have {seeds}")

    # ---- sanity checks
    checks = {"seeds": seeds}
    pred_h, adp_h, counts = {}, {}, {}
    for k, arm in ARMS.items():
        for s in seeds:
            f = P / f"{arm}_seed{s}_test.jsonl"
            pred_h[f.name] = sha(f)
            counts[f.name] = sum(1 for _ in open(f))
            ad = T / f"{arm}_seed{s}" / "adapter_fixed" / "adapters.safetensors"
            adp_h[f"{arm}_seed{s}"] = sha(ad) if ad.exists() else None
    checks["prediction_files_identical"] = [sorted(n for n, h in pred_h.items() if h == v)
                                            for v, c in Counter(pred_h.values()).items() if c > 1]
    checks["adapters_identical"] = [sorted(n for n, h in adp_h.items() if h == v)
                                    for v, c in Counter(adp_h.values()).items() if c > 1 and v]
    checks["test_rows_per_file"] = counts
    checks["all_3622"] = all(v == 3622 for v in counts.values())
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(str(BASE))
        probe = "Mark Parker and Susan Wojcicki"
        checks["tokenizer_round_trip"] = tk.decode(tk(probe, add_special_tokens=False)["input_ids"]) == probe
        checks["tokenizer_class"] = type(tk).__name__
    except Exception as exc:
        checks["tokenizer_round_trip"] = f"error: {exc!r}"
    metas = {}
    for k, arm in ARMS.items():
        for s in seeds:
            mp = T / f"{arm}_seed{s}" / "run_meta.json"
            if mp.exists():
                m = json.load(open(mp))
                metas[f"{arm}_seed{s}"] = {"model": m.get("model"), "n_raw_examples": m.get("n_raw_examples"),
                                            "n_steps": m["conditions"]["fixed"].get("n_steps"),
                                            "lambda_over": m.get("lambda_over"), "hedge_token_mode": m.get("hedge_token_mode")}
    checks["run_meta"] = metas
    checks["compute_matched"] = len({(v["n_raw_examples"], v["n_steps"]) for v in metas.values()}) == 1

    # ---- TemporalDelta
    td, td_controls = {}, {}
    for k, arm in ARMS.items():
        for s in seeds:
            recs = [json.loads(l) for l in open(P / f"{arm}_seed{s}_test.jsonl")]
            td[(k, s)] = arm_metrics(recs)
            td_controls[f"{k}_s{s}"] = controls(recs)
    td_paired = paired(td, seeds, EPS)

    # ---- FreshQA, containment against both keys
    fq = {}
    for key in ("current", "aged"):
        res = {}
        for k, arm in ARMS.items():
            for s in seeds:
                recs = [json.loads(l) for l in open(P / f"{arm}_seed{s}_freshqa.jsonl")]
                for r in recs:
                    ans = FRESH[r["freshqa_id"]][f"{key}_answers"]
                    p = N(r["predicted_answer"])
                    r["correct"] = any(N(x) and N(x) in p for x in ans)
                    r["gold_hedge"] = FRESH[r["freshqa_id"]]["gold_hedge"]
                res[(k, s)] = arm_metrics(recs)
        fq[key] = {"paired": paired(res, seeds, EPS), "per_run": {f"{k}_s{s}": v for (k, s), v in res.items()}}

    qwen_td = json.load(open(ROOT / "docs/orientation/results_sep10/item5/seed_variance_routeB.json"))["paired"]
    qwen_fq = json.load(open(ROOT / "docs/orientation/results_sep10/item5/freshqa_seeds.json"))
    out = {"tag": a.tag, "n_seed_pairs": len(seeds), "sanity": checks,
           "temporal_delta": {"paired": td_paired, "per_run": {f"{k}_s{s}": v for (k, s), v in td.items()},
                              "controls": td_controls},
           "freshqa": fq,
           "qwen_reference": {"temporal_delta_primary": next(v for kk, v in qwen_td.items() if kk.endswith(PRIMARY)),
                              "freshqa_current_primary": qwen_fq["current"][PRIMARY]}}
    json.dump(out, open(OUT / f"llama3_results_{a.tag}.json", "w"), indent=2)

    def row(name, d):
        return (f"| {name} | {d['tsct_mean']:.4f} | {d['sft_mean']:.4f} | {d['mean']:+.4f} | "
                f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | {'yes' if d['sign_consistent'] else 'no'} | {d['sft_seed_sd']:.4f} |")
    lines = [f"### LLaMA-3-8B, {len(seeds)} paired seeds {seeds} ({a.tag})", "",
             "TemporalDelta test (3,622 per arm-seed):", "",
             "| endpoint | TSCT mean | plain FT mean | Δ (TSCT − FT) | 95% t-interval | sign-consistent | FT seed SD |",
             "|---|---|---|---|---|---|---|"]
    for ep in EPS:
        lines.append(row(ep, td_paired[ep]))
    lines.append(f"\nEffect by the prespecified rule: **{td_paired['effect_by_prespecified_rule']}**")
    oc = [v["oracle"]["ece_by_level"] for v in td_controls.values()]
    un = [v["constant [UNKNOWN]"]["ece_by_level"] for v in td_controls.values()]
    ra = [v["oracle_over_unknown"] for v in td_controls.values()]
    lines.append(f"\nControls on the same correctness (range over arm-seeds): oracle ECE {min(oc):.4f}–{max(oc):.4f}; "
                 f"constant [UNKNOWN] {min(un):.4f}–{max(un):.4f}; oracle / constant {min(ra):.2f}–{max(ra):.2f}×.")
    for key in ("current", "aged"):
        p = fq[key]["paired"]
        lines.append(f"\nFreshQA ({key} key): ΔAUROC {p[PRIMARY]['mean']:+.4f} [{p[PRIMARY]['ci95'][0]:+.4f}, {p[PRIMARY]['ci95'][1]:+.4f}], "
                     f"ΔECE {p['ece_by_level']['mean']:+.4f} [{p['ece_by_level']['ci95'][0]:+.4f}, {p['ece_by_level']['ci95'][1]:+.4f}], "
                     f"Δresolution {p['resolution']['mean']:+.4f}; effect by rule: {p['effect_by_prespecified_rule']}")
    lines.append(f"\nSanity: identical prediction files {checks['prediction_files_identical']}; identical adapters "
                 f"{checks['adapters_identical']}; all 3,622 rows: {checks['all_3622']}; tokenizer round-trip: "
                 f"{checks['tokenizer_round_trip']}; compute-matched: {checks['compute_matched']}")
    (OUT / f"llama3_table_{a.tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
