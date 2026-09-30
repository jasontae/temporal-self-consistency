"""TSCT question-endpoint paragraph (ARR panel change 5): how often does the emitted
hedge equal the gold volatility (TVT) hedge, and how often is [UNKNOWN] emitted?
Route B arms on the TemporalDelta test set (3,622 items).

    python3 docs/orientation/results_sep10/paper_tables/hedge_vs_label.py
"""
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PRED = ROOT / "data" / "prep" / "predictions_7b"
out = {}
for f in ["tsct_test.jsonl", "tsctS1_test.jsonl", "sft_test.jsonl", "sftS1_test.jsonl"]:
    R = [json.loads(l) for l in open(PRED / f)]
    agree = sum(r["predicted_hedge"] == r["gold_hedge"] for r in R)
    c = collections.Counter(r["predicted_hedge"] for r in R)
    out[f] = {"n": len(R), "agree_with_gold_hedge": agree, "agree_share": agree / len(R),
              "emitted": dict(c), "gold": dict(collections.Counter(r["gold_hedge"] for r in R)),
              "unknown_emitted": c.get("[UNKNOWN]", 0)}
# Is the gold volatility label a function of the question's relation template?
import re


def template(q):
    q = re.sub(r"\s+", " ", q.strip())
    if q.startswith("Who owns"):
        return "Who owns"
    m = re.match(r"^(.*?\b(of|in|for|by)\b)\s", q)
    return m.group(1) if m else " ".join(q.split()[:4])


R = [json.loads(l) for l in open(PRED / "tsct_test.jsonl")]
g = collections.defaultdict(collections.Counter)
for r in R:
    g[template(r["question"])][r["gold_hedge"]] += 1
out["template_determines_gold_hedge"] = {"templates": len(g), "items": len(R),
                                         "items_matching_template_majority": sum(max(c.values()) for c in g.values()),
                                         "by_template": {t: dict(c) for t, c in g.items()}}
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
print("templates", len(g), "majority-consistent items", out["template_determines_gold_hedge"]["items_matching_template_majority"], "of", len(R))
for k, v in out.items():
    if "n" not in v:
        continue
    print(k, v["n"], v["agree_with_gold_hedge"], round(v["agree_share"], 4), v["emitted"], "gold", v["gold"])
