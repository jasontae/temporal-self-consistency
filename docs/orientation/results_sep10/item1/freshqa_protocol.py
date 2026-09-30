"""Item 1: the measurement-validity protocol on a second aged benchmark (FreshQA).

FreshQA (Vu et al. 2024, Apache-2.0) labels each question never-, slow- or
fast-changing and republishes the whole set with dated answers. The oldest
listed snapshot (2024-02-26) is the aged key; the newest (2026-04-21) is the
current key. See SEP10_PLAN.md, item 1, for why this benchmark.

Subcommands (run from the repo root, in order):

  build     data/prep/freshqa/freshqa_<date>.csv  ->  data/prep/freshqa/freshqa_built.jsonl
            Drops false-premise questions. Maps fact_type to the hedge
            vocabulary (never -> [CONFIDENT], slow -> [COND_CONFIDENT],
            fast -> [TEMPORAL_HEDGE]). Keeps every accepted answer per snapshot.

  claims    freshqa_built.jsonl -> data/stress_tests/freshqa_claims.jsonl, in the
            matched-set scorer's format, one frame for every arm:
              "Question: <q> Answer: <a>."   neutral frame "Answer: <a>." for PMI
            arms, paired within question:
              fq_current / fq_stale   changed questions: current vs aged answer (staleness)
              fq_true / fq_false      never-changing questions: true answer vs the true
                                      answer of another never-changing question of the
                                      same coarse answer type (number / year / text),
                                      each used once as a false answer (manipulation check)
            Score with:
              python3 -m src.training.score_matched_regime --base <model> \
                --claims data/stress_tests/freshqa_claims.jsonl --out data/prep/predictions_7b/fq_<tag>.jsonl

  analyze   (a) calibration controls, both keys, for every predictions_7b/freshqa_*.jsonl
                (generate_predictions --source freshqa; needs the hedge-trained adapters):
                oracle and constant ECE (binned and by level), Brier / Murphy,
                ratio with bootstrap CI, and the constant-vs-oracle crossover
            (b) reference repair: aged key vs current key
            (c) no-model form screen on the question text (fast vs never-changing)
                beside each model's claim log-probability, with a paired bootstrap
            (d) truth (manipulation check) and staleness AUROC per model from fq_*.jsonl

Correctness: the pipeline's exact_match against ANY accepted answer of the key
(strict); containment of any accepted answer is reported as a sensitivity. No
LLM judge.
"""
import argparse
import csv
import glob
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from eval_pipeline import HEDGE_TO_CONFIDENCE as H, compute_ece, exact_match, normalize_answer  # noqa: E402

FQ = ROOT / "data" / "prep" / "freshqa"
BUILT = FQ / "freshqa_built.jsonl"
CLAIMS = ROOT / "data" / "stress_tests" / "freshqa_claims.jsonl"
PRED = ROOT / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent
SEED = 20260928
FACT = {"never-changing": ("[CONFIDENT]", "immutable"), "slow-changing": ("[COND_CONFIDENT]", "slow"),
        "fast-changing": ("[TEMPORAL_HEDGE]", "fast")}


# ---------------------------------------------------------------- build
def read_snapshot(path):
    """FreshQA sheets carry preamble rows above the header; find the header."""
    rows = list(csv.reader(open(path, encoding="utf-8")))
    for i, r in enumerate(rows):
        low = [c.strip().lower() for c in r]
        if "question" in low and "fact_type" in low:
            hdr = low
            break
    else:
        raise SystemExit(f"{path}: no header row with 'question' and 'fact_type'")
    need = {"id", "question", "fact_type", "false_premise"}
    if not need <= set(hdr):
        raise SystemExit(f"{path}: header lacks {need - set(hdr)}; got {hdr}")
    ans_cols = [j for j, h in enumerate(hdr) if re.fullmatch(r"answer_\d+", h)]
    out = {}
    for r in rows[i + 1:]:
        if len(r) < len(hdr) or not r[hdr.index("id")].strip():
            continue
        d = dict(zip(hdr, r))
        out[d["id"].strip()] = {
            "question": d["question"].strip(), "fact_type": d["fact_type"].strip().lower(),
            "false_premise": d["false_premise"].strip().lower() in ("true", "yes", "1"),
            "split": d.get("split", "").strip().lower(),
            "answers": [r[j].strip() for j in ans_cols if j < len(r) and r[j].strip()],
        }
    return out


def cmd_build(_):
    snaps = sorted(FQ.glob("freshqa_20*.csv"))
    if len(snaps) < 2:
        raise SystemExit("need at least two snapshots in data/prep/freshqa (FETCH_ALL.sh step 5)")
    # FreshQA reassigns `id` between snapshots (only 119 of 600 ids keep their
    # question from 2024-02-26 to 2026-04-21), so snapshots are joined on the
    # normalized question text; `id` below is the current snapshot's.
    qkey = lambda q: re.sub(r"\s+", " ", q.strip().lower())
    data = {}
    for p in snaps:
        snap = read_snapshot(p)
        data[p.stem.replace("freshqa_", "")] = {qkey(v["question"]): dict(v, id=k) for k, v in snap.items()}
    dates = sorted(data)
    aged, cur = data[dates[0]], data[dates[-1]]
    built, dropped = [], defaultdict(int)
    for qk, c in cur.items():
        qid = c["id"]
        if c["false_premise"]:
            dropped["false_premise"] += 1
            continue
        if c["fact_type"] not in FACT:
            dropped[f"fact_type={c['fact_type']}"] += 1
            continue
        if qk not in aged or not aged[qk]["answers"] or not c["answers"]:
            dropped["question not in aged snapshot, or no answer"] += 1
            continue
        hedge, vol = FACT[c["fact_type"]]
        na = {normalize_answer(a) for a in aged[qk]["answers"]}
        nc = {normalize_answer(a) for a in c["answers"]}
        built.append({"id": qid, "split": c["split"], "question": c["question"], "fact_type": c["fact_type"],
                      "gold_hedge": hedge, "volatility": vol,
                      "aged_answers": aged[qk]["answers"], "current_answers": c["answers"],
                      "changed": na.isdisjoint(nc),
                      "aged_fact_type": aged[qk]["fact_type"],
                      "answers_by_date": {d: data[d][qk]["answers"] for d in dates if qk in data[d]}})
    with open(BUILT, "w") as f:
        for r in built:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"snapshots": dates, "aged": dates[0], "current": dates[-1], "n_built": len(built),
           "dropped": dict(dropped),
           "by_fact_type": {k: sum(r["fact_type"] == k for r in built) for k in FACT},
           "changed_by_fact_type": {k: sum(r["changed"] for r in built if r["fact_type"] == k) for k in FACT}}
    json.dump(rep, open(OUT / "build_report.json", "w"), indent=2)
    print(json.dumps(rep, indent=2))


# ---------------------------------------------------------------- claims
def atype(a):
    s = a.strip()
    if re.fullmatch(r"(1[5-9]|20)\d\d", s):
        return "year"
    if re.fullmatch(r"[\d.,%$ ]+", s):
        return "number"
    return "text"


def cmd_claims(_):
    rows = [json.loads(l) for l in open(BUILT)]
    rng = random.Random(SEED)
    out = []

    def claim(r, arm, ans, truth):
        return {"entity_id": r["id"], "entity": r["question"], "arm": arm, "fact_type": r["fact_type"],
                "truth": truth, "person": ans, "role": "answer", "property": r["fact_type"],
                "text": f"Question: {r['question']} Answer: {ans}.", "name_only_text": f"Answer: {ans}."}

    for r in rows:
        if r["changed"] and r["fact_type"] != "never-changing":
            out.append(claim(r, "fq_current", r["current_answers"][0], "true"))
            out.append(claim(r, "fq_stale", r["aged_answers"][0], "false"))
    never = [r for r in rows if r["fact_type"] == "never-changing"]
    by_t = defaultdict(list)
    for r in never:
        by_t[atype(r["current_answers"][0])].append(r)
    for t, grp in by_t.items():
        if len(grp) < 2:
            continue
        order = grp[:]
        rng.shuffle(order)
        for i, r in enumerate(order):  # cyclic shift within type: each answer used once as false
            fa = order[(i + 1) % len(order)]["current_answers"][0]
            if normalize_answer(fa) in {normalize_answer(a) for a in r["current_answers"]}:
                continue
            out.append(claim(r, "fq_true", r["current_answers"][0], "true"))
            out.append(claim(r, "fq_false", fa, "false"))
    with open(CLAIMS, "w") as f:
        for c in out:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    cnt = defaultdict(int)
    for c in out:
        cnt[c["arm"]] += 1
    print(f"wrote {len(out)} claims to {CLAIMS}: {dict(cnt)}")


# ---------------------------------------------------------------- analyze
def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def by_level(c, y):
    ob = y.mean()
    lv = np.unique(c)
    return {"ece_by_level": float(sum((c == k).mean() * abs(y[c == k].mean() - k) for k in lv)),
            "brier": float(((c - y) ** 2).mean()),
            "resolution": float(sum((c == k).mean() * (y[c == k].mean() - ob) ** 2 for k in lv)),
            "reliability": float(sum((c == k).mean() * (k - y[c == k].mean()) ** 2 for k in lv))}


def correct(pred, answers, mode):
    if mode == "exact":
        return any(exact_match(pred, a) for a in answers)
    p = normalize_answer(pred)
    return any(normalize_answer(a) and normalize_answer(a) in p for a in answers)


def calib(recs, built, key, mode, rng):
    rows = [(r, built[r["freshqa_id"]]) for r in recs if r.get("freshqa_id") in built and r.get("predicted_hedge") in H]
    y = np.array([float(correct(r["predicted_answer"], b[f"{key}_answers"], mode)) for r, b in rows])
    pol = {"model": np.array([H[r["predicted_hedge"]] for r, _ in rows]),
           "oracle": np.array([H[b["gold_hedge"]] for _, b in rows]),
           "constant [TEMPORAL_HEDGE]": np.full(len(rows), 0.45), "constant [UNKNOWN]": np.full(len(rows), 0.10)}
    names = {0.95: "[CONFIDENT]", 0.75: "[COND_CONFIDENT]", 0.45: "[TEMPORAL_HEDGE]", 0.10: "[UNKNOWN]"}
    res = {"n": len(rows), "accuracy": float(y.mean()),
           "accuracy_by_fact_type": {ft: float(y[[b["fact_type"] == ft for _, b in rows]].mean())
                                     for ft in FACT if any(b["fact_type"] == ft for _, b in rows)}}
    for k, c in pol.items():
        res[k] = by_level(c, y) | {"ece_binned_10": compute_ece(
            [{"predicted_hedge": names[float(v)], "correct": bool(t)} for v, t in zip(c, y)])["ece"]}
    o, u = pol["oracle"], pol["constant [UNKNOWN]"]
    ratios = []
    for _ in range(4000):
        ix = rng.integers(0, len(y), len(y))
        eo, eu = by_level(o[ix], y[ix])["ece_by_level"], by_level(u[ix], y[ix])["ece_by_level"]
        if eu > 0:
            ratios.append(eo / eu)
    res["oracle_over_unknown"] = res["oracle"]["ece_by_level"] / max(res["constant [UNKNOWN]"]["ece_by_level"], 1e-12)
    res["ratio_ci95"] = np.percentile(ratios, [2.5, 97.5]).tolist() if ratios else None
    res["ratio_boot_resamples_used"] = len(ratios)
    # crossover: scale the fast class's accuracy with the others fixed (exact, as in ../crossover)
    w = {ft: np.mean([b["fact_type"] == ft for _, b in rows]) for ft in FACT}
    a = res["accuracy_by_fact_type"]
    lev = {"never-changing": 0.95, "slow-changing": 0.75, "fast-changing": 0.45}
    grid = np.linspace(0, 1, 10001)
    orc = sum(w[ft] * (np.abs(grid - lev[ft]) if ft == "fast-changing" else abs(a.get(ft, 0) - lev[ft])) for ft in FACT)
    abar = sum(w[ft] * (grid if ft == "fast-changing" else a.get(ft, 0)) for ft in FACT)
    cu = np.abs(abar - 0.10)
    wins = grid[cu < orc]
    res["constant_unknown_beats_oracle_for_fast_accuracy_below"] = float(wins.max()) if len(wins) and wins.min() == 0 else None
    return res


SCREEN = {
    "n_chars": lambda q: len(q), "n_words": lambda q: len(q.split()),
    "has_digit": lambda q: float(bool(re.search(r"\d", q))),
    "temporal_cue": lambda q: float(bool(re.search(r"\b(current|currently|latest|most recent|now|this year|today|recent)\b", q, re.I))),
    "past_tense_cue": lambda q: float(bool(re.search(r"\b(was|were|did|founded|born|died|wrote|invented)\b", q, re.I))),
    "starts_who": lambda q: float(q.lower().startswith("who")),
}


def cmd_analyze(_):
    built = {r["id"]: r for r in map(json.loads, open(BUILT))}
    rng = np.random.default_rng(SEED)
    out = {"calibration": {}, "screen": {}, "contrasts": {}}
    for f in sorted(PRED.glob("freshqa_*.jsonl")):
        recs = [json.loads(l) for l in open(f)]
        out["calibration"][f.stem] = {f"{key}/{mode}": calib(recs, built, key, mode, rng)
                                      for key in ("aged", "current") for mode in ("exact", "contains")}
    # (c) screen: fast (positive) vs never-changing, on question text
    fast = [r for r in built.values() if r["fact_type"] == "fast-changing"]
    never = [r for r in built.values() if r["fact_type"] == "never-changing"]
    for name, fn in SCREEN.items():
        a = auroc([fn(r["question"]) for r in fast], [fn(r["question"]) for r in never])
        out["screen"][name] = max(a, 1 - a)  # direction-free: the strongest either way
    best = max(out["screen"], key=out["screen"].get)
    # (d) per-model contrasts from fq_*.jsonl, and the model's screen entry
    for f in sorted(PRED.glob("fq_*.jsonl")):
        by = defaultdict(dict)
        for r in map(json.loads, open(f)):
            by[r["entity_id"]][r["arm"]] = r["pmi"]
        st = [(a["fq_current"], a["fq_stale"]) for a in by.values() if "fq_current" in a and "fq_stale" in a]
        tr = [(a["fq_true"], a["fq_false"]) for a in by.values() if "fq_true" in a and "fq_false" in a]
        res = {}
        for name, pairs in (("staleness", st), ("truth", tr)):
            if len(pairs) < 5:
                continue
            p = np.array(pairs)
            boots = [auroc(p[ix, 0], p[ix, 1]) for ix in (rng.integers(0, len(p), len(p)) for _ in range(4000))]
            d = p[:, 0] - p[:, 1]
            d = d[d != 0]
            res[name] = {"n": len(p), "auroc": float(auroc(p[:, 0], p[:, 1])),
                         "ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
                         "paired_win": float((p[:, 0] > p[:, 1]).mean()),
                         "sign_p": float(stats.binomtest(int((d > 0).sum()), len(d)).pvalue)}
        # model signal for the screen: claim log-prob of the current true answer, fast vs never
        cur = {e: a.get("fq_current", a.get("fq_true")) for e, a in by.items()}
        fs = [cur[r["id"]] for r in fast if cur.get(r["id"]) is not None]
        ns = [cur[r["id"]] for r in never if cur.get(r["id"]) is not None]
        if len(fs) > 5 and len(ns) > 5:
            am = auroc(fs, ns)
            res["screen_model_auroc"] = float(max(am, 1 - am))
        res["admitted_truth_ge_0.70"] = bool(res.get("truth", {}).get("auroc", 0) >= 0.70)
        out["contrasts"][f.stem] = res
    out["screen_best_no_model_feature"] = {"feature": best, "auroc": out["screen"][best]}
    json.dump(out, open(OUT / "freshqa_results.json", "w"), indent=2)
    print(json.dumps({"screen": out["screen"], "contrasts": out["contrasts"],
                      "calibration_keys": list(out["calibration"])}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "claims", "analyze"):
        sub.add_parser(c)
    args = ap.parse_args()
    {"build": cmd_build, "claims": cmd_claims, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
