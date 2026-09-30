"""Claims for the in-context evidence control (PLAN_context_control.md, Amendment 1).

Each entity's current and expired office-holder claims, each under three contexts that share one template:
current evidence, expired evidence, and a neutral sentence naming neither holder.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "data" / "stress_tests" / "matched_regime_claims_disjoint.jsonl"
OUT = REPO / "data" / "stress_tests" / "matched_context_claims.jsonl"

rows = [json.loads(l) for l in open(SRC)]
by_ent = {}
for r in rows:
    if r["arm"] in ("volatile_current", "volatile_stale"):
        by_ent.setdefault(r["entity_id"], {})[r["arm"]] = r
n = 0
with open(OUT, "w") as f:
    for eid, arms in by_ent.items():
        cur, st = arms["volatile_current"], arms["volatile_stale"]
        ctxs = {
            "current_evidence": f"News: {cur['person']} has been appointed {cur['role']} of {cur['entity']}.",
            "expired_evidence": f"News: {st['person']} has been appointed {cur['role']} of {cur['entity']}.",
            "neutral": f"News: {cur['entity']} has published its annual report.",
        }
        for cname, ctx in ctxs.items():
            for r in (cur, st):
                f.write(json.dumps({**r, "context_condition": cname, "context": ctx}, ensure_ascii=False) + "\n")
                n += 1
print(f"wrote {n} rows ({len(by_ent)} entities x 3 contexts x 2 arms) to {OUT}")
