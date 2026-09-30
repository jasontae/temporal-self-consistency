"""A6: rebuild the false-founder arm from a disjoint pool.

The original build (src/evaluation/build_matched_regime_set.py) makes each false
founder a derangement of the true founders, so every founder name appears twice
(once true, once false) and every office-holder once. The Sep 2 Appendix C
describes a resample from a disjoint pool, but the scored data never had one.

This keeps the three other arms of every entity exactly as built and replaces
`immutable_false` with a human founder of some *other* organization that:
  - is not any person already in the set (true founder, current or stale holder);
  - is used once only;
  - is matched to the entity's true founder on Wikidata sitelink count (nearest
    in log(1 + sitelinks), greedy without replacement, entities processed in a
    seeded random order), so the false arm keeps the true arm's prominence
    distribution, which the derangement guaranteed by construction.

Pool: data/prep/wikidata_a6/human_founders.json (Wikidata SPARQL, P112 values
that are P31 = Q5, sitelinks >= 2). True founders' sitelinks:
data/prep/wikidata_a6/set_persons.json.

Output: data/stress_tests/matched_regime_claims_disjoint.jsonl (the original
file is left untouched), and a build report next to this script.

Run from the repo root:
    python3 docs/orientation/results_sep10/a6_disjoint/build_disjoint.py
"""
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
CLAIMS = ROOT / "data" / "stress_tests" / "matched_regime_claims.jsonl"
OUT = ROOT / "data" / "stress_tests" / "matched_regime_claims_disjoint.jsonl"
POOL = ROOT / "data" / "prep" / "wikidata_a6" / "human_founders.json"
SETP = ROOT / "data" / "prep" / "wikidata_a6" / "set_persons.json"
HERE = Path(__file__).resolve().parent
SEED = 20260928
QID_LIKE = re.compile(r"^Q\d+$")


def main():
    rows = [json.loads(l) for l in open(CLAIMS)]
    in_set = {r["person"] for r in rows}
    sl_by_label = {}
    for p in json.load(open(SETP))["rows"]:
        if p["label"]:
            sl_by_label[p["label"]] = max(sl_by_label.get(p["label"], 0), p["sitelinks"])

    pool_rows = json.load(open(POOL))["rows"]
    for p in pool_rows:  # founders missing from set_persons (label lookups) are in the pool file
        if p["label"] and p["label"] not in sl_by_label:
            sl_by_label[p["label"]] = p["sitelinks"]
    pool = {}
    for p in pool_rows:
        lab = p["label"]
        if not lab or QID_LIKE.match(lab) or lab in in_set:
            continue
        if len(lab.split()) < 2:  # true founders are full names; mononyms are rare there
            continue
        pool[lab] = max(pool.get(lab, 0), p["sitelinks"])
    pool_items = sorted(pool.items())

    by_ent = {}
    for r in rows:
        by_ent.setdefault(r["entity_id"], {})[r["arm"]] = r
    eids = sorted(by_ent)
    rng = random.Random(SEED)
    order = eids[:]
    rng.shuffle(order)

    used, choice, unmatched = set(), {}, []
    for e in order:
        tf = by_ent[e]["immutable_true"]["person"]
        target = sl_by_label.get(tf)
        if target is None:
            unmatched.append(tf)
            target = 20  # median-ish fallback; reported below
        lt = math.log1p(target)
        best = min((abs(math.log1p(s) - lt), rng.random(), lab, s)
                   for lab, s in pool_items if lab not in used)
        used.add(best[2])
        choice[e] = (best[2], best[3], target)

    out_rows = []
    for e in eids:
        for arm in ("volatile_current", "volatile_stale", "immutable_true", "immutable_false"):
            r = dict(by_ent[e][arm])
            if arm == "immutable_false":
                name = choice[e][0]
                r.update(person=name, text=f"The founder of {r['entity']} is {name}.",
                         name_only_text=f"{name} is a person.")
            out_rows.append(r)
    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = Counter(r["person"] for r in out_rows)
    t_sl = [choice[e][2] for e in eids]
    f_sl = [choice[e][1] for e in eids]
    ks = stats.ks_2samp(t_sl, f_sl)
    rep = {
        "pool_size_after_filters": len(pool_items), "entities": len(eids), "claims": len(out_rows),
        "persons_appearing_more_than_once": sorted(p for p, c in counts.items() if c > 1),
        "false_founders_that_are_true_founders_elsewhere": sum(
            choice[e][0] in {by_ent[x]["immutable_true"]["person"] for x in eids} for e in eids),
        "true_founders_without_sitelinks": unmatched,
        "sitelinks_true_median": sorted(t_sl)[len(t_sl) // 2],
        "sitelinks_false_median": sorted(f_sl)[len(f_sl) // 2],
        "ks_true_vs_false_sitelinks": {"stat": ks.statistic, "p": ks.pvalue},
        "mean_abs_log_sitelink_gap": sum(abs(math.log1p(a) - math.log1p(b)) for a, b in zip(t_sl, f_sl)) / len(eids),
    }
    json.dump(rep, open(HERE / "build_report.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
