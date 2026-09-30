"""Analysis for PLAN_knowledge_probe.md: staleness conditioned on what each model can name.

Labels each generated answer current / stale / other with a deterministic name
matcher, joins the labels to the fixed 4-bit PMI scores (disjoint4b_<tag>.jsonl),
and reports the paired win rate 1[PMI(current) > PMI(stale)] per knowledge group
with a 4,000-draw entity-cluster bootstrap.

    python3 docs/orientation/results_sep10/knowledge_probe/knowledge_analysis.py
"""
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
P = REPO / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent
NINE = ["g3_4b", "qwen3_4b_th", "qwen3_4b_it", "qwen25_7b", "tsct", "gptoss_20b",
        "qwen36_35b", "qwen35_27b", "qwen36_27b_stock"]
SUFFIX = {"jr", "sr", "ii", "iii", "iv", "dr", "mr", "ms", "mrs", "sir", "brother", "fr", "prof"}
B = 4000


def toks(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def core(name):
    t = [x for x in toks(name) if x not in SUFFIX and len(x) > 1]
    return t


def mentions(answer, name):
    """Surname (last core token) present; if the name has a given name, it or its initial must
    also appear, so 'Johnson' alone does not match two different Johnsons."""
    a = toks(answer)
    c = core(name)
    if not c or not a:
        return False
    if c[-1] not in a:
        return False
    if len(c) == 1:
        return True
    given = c[:-1]
    return any(g in a for g in given) or any(x == g[0] for g in given for x in a if len(x) == 1) \
        or len(core(answer)) <= 1  # bare surname answer


def label(answer, cur, stale):
    mc, ms = mentions(answer, cur), mentions(answer, stale)
    if mc and ms:
        # shared tokens: require the full core name
        fc = all(t in toks(answer) for t in core(cur))
        fs = all(t in toks(answer) for t in core(stale))
        if fc and not fs:
            return "current"
        if fs and not fc:
            return "stale"
        return "ambiguous"
    return "current" if mc else "stale" if ms else "other"


def load_pmi(tag):
    rows = [json.loads(l) for l in open(P / f"disjoint4b_{tag}.jsonl")]
    d = defaultdict(dict)
    for r in rows:
        if r["arm"] in ("volatile_current", "volatile_stale"):
            d[r["entity_id"]][r["arm"]] = r["pmi"]
    return d


def main(field="qa_answer"):
    data = []  # (tag, entity_id, group, win, zdiff)
    per_model = {}
    audit = []
    for tag in NINE:
        f = P / f"knowledge_{tag}.jsonl"
        if not f.exists():
            print(f"missing {f.name}", file=sys.stderr)
            continue
        kn = {r["entity_id"]: r for r in map(json.loads, open(f))}
        pmi = load_pmi(tag)
        diffs = {e: v["volatile_current"] - v["volatile_stale"] for e, v in pmi.items()
                 if v.get("volatile_current") is not None and v.get("volatile_stale") is not None}
        mu = sum(diffs.values()) / len(diffs)
        sd = (sum((x - mu) ** 2 for x in diffs.values()) / (len(diffs) - 1)) ** 0.5
        cnt = Counter()
        for e, r in kn.items():
            if e not in diffs:
                continue
            g = label(r[field], r["current"], r["stale"])
            cnt[g] += 1
            data.append((tag, e, g, 1.0 if diffs[e] > 0 else 0.0, diffs[e] / sd))
            audit.append({"tag": tag, "entity": r["entity"], "current": r["current"], "stale": r["stale"],
                          "answer": r[field], "label": g})
        per_model[tag] = dict(cnt)

    ents = sorted({e for _, e, *_ in data})
    by_ent = defaultdict(list)
    for row in data:
        by_ent[row[1]].append(row)

    def stat(rows, g):
        sel = [r for r in rows if g == "all" or r[2] == g]
        if not sel:
            return None, None, 0
        return (sum(r[3] for r in sel) / len(sel), sum(r[4] for r in sel) / len(sel), len(sel))

    rng = random.Random(20260930)
    groups = ["all", "current", "stale", "other", "ambiguous"]
    boots = {g: ([], []) for g in groups}
    for _ in range(B):
        rows = [r for e in (rng.choice(ents) for _ in ents) for r in by_ent[e]]
        for g in groups:
            w, z, n = stat(rows, g)
            if n:
                boots[g][0].append(w)
                boots[g][1].append(z)

    def ci(xs):
        xs = sorted(xs)
        return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]] if xs else [None, None]

    res = {"field": field, "per_model_counts": per_model, "groups": {}}
    for g in groups:
        w, z, n = stat(data, g)
        if not n:
            continue
        wb, zb = boots[g]
        res["groups"][g] = {"n_pairs": n, "n_entities": len({r[1] for r in data if g == "all" or r[2] == g}),
                            "win_rate": w, "win_ci": ci(wb), "p_win_ge_half": sum(x >= 0.5 for x in wb) / len(wb),
                            "z_diff": z, "z_ci": ci(zb)}
    # per-model win rate among 'current'
    pm = {}
    for tag in per_model:
        sel = [r for r in data if r[0] == tag and r[2] == "current"]
        pm[tag] = {"n": len(sel), "win_rate": (sum(r[3] for r in sel) / len(sel)) if sel else None}
    res["per_model_current_win"] = pm
    return res, audit


if __name__ == "__main__":
    allres = {}
    for field in ("qa_answer", "frame_completion"):
        res, audit = main(field)
        allres[field] = res
        with open(OUT / f"audit_{field}.jsonl", "w") as f:
            for a in audit:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        print(f"\n== {field}")
        print("counts per model:", json.dumps(res["per_model_counts"]))
        for g, v in res["groups"].items():
            print(f"{g:10s} pairs={v['n_pairs']:4d} ents={v['n_entities']:3d} win={v['win_rate']:.3f} "
                  f"[{v['win_ci'][0]:.3f}, {v['win_ci'][1]:.3f}] z={v['z_diff']:+.3f} "
                  f"[{v['z_ci'][0]:+.3f}, {v['z_ci'][1]:+.3f}]")
        print("per-model win among 'current':", json.dumps(res["per_model_current_win"]))
    json.dump(allres, open(OUT / "knowledge_results.json", "w"), indent=1)
