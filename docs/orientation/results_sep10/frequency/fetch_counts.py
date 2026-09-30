"""Fetch infini-gram co-occurrence counts for the 106 matched-set entities (PREREG.md).

Queries the public API only (no corpus download), sequentially, with >= 1 s between
requests and up to 5 retries with exponential backoff. Resumable: rows already in
counts.jsonl are skipped.

    python3 docs/orientation/results_sep10/frequency/fetch_counts.py
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[4]
CLAIMS = ROOT / "data" / "stress_tests" / "matched_regime_claims.jsonl"
OUT = Path(__file__).resolve().parent / "counts.jsonl"
API = "https://api.infini-gram.io/"
MIN_GAP_S = 2.0  # raised from 1.0 after intermittent HTTP 403 (rate limiting) on 2026-09-28
MAX_CLAUSE_FREQ = 500000

PRIMARY = "v4_dolma-v1_7_llama"
SECONDARY = ["v4_rpj_llama_s4", "v4_piletrain_llama"]


def forms(E, P, R):
    """form id -> (query, extra params). F5 (entity marginal) is added separately."""
    return {
        "F1": (f"{E} AND {P}", {"max_diff_tokens": 100, "max_clause_freq": MAX_CLAUSE_FREQ}),
        "F2": (f"{E} AND {P}", {"max_diff_tokens": 20, "max_clause_freq": MAX_CLAUSE_FREQ}),
        # F3 (PREREG) is one OR clause of five templates; the API allows at most four terms
        # per clause, so it is sent as F3a (four) + F3b (one) and summed in the analysis.
        # The templates are distinct n-grams, so the sum equals the single OR count.
        "F3a": (f"{R} of {E} is {P} OR {R} of {E} was {P} OR {P}, {R} of {E} OR {P}, the {R} of {E}", {}),
        "F3b": (f"{E} {R} {P}", {}),
        "F4": (P, {}),
    }


def plan():
    ents = {}
    for line in open(CLAIMS):
        r = json.loads(line)
        if r["arm"] in ("volatile_current", "volatile_stale"):
            ents.setdefault(r["entity_id"], {"entity": r["entity"], "role": r["role"]})[r["arm"]] = r["person"]
    assert len(ents) == 106, len(ents)
    jobs = []
    for eid, e in ents.items():
        for s in (e["entity"], e["volatile_current"], e["volatile_stale"], e["role"]):
            assert " AND " not in s and " OR " not in s, s
        for index in [PRIMARY] + SECONDARY:
            for value, arm in (("current", "volatile_current"), ("expired", "volatile_stale")):
                for fid, (q, params) in forms(e["entity"], e[arm], e["role"]).items():
                    if index != PRIMARY and fid.startswith("F3"):
                        continue
                    jobs.append(dict(index=index, form=fid, entity_id=eid, value=value,
                                     entity=e["entity"], person=e[arm], role=e["role"], query=q, params=params))
            if index == PRIMARY:
                jobs.append(dict(index=index, form="F5", entity_id=eid, value="entity",
                                 entity=e["entity"], person=None, role=e["role"], query=e["entity"], params={}))
    return jobs


def key(j):
    return (j["index"], j["form"], j["entity_id"], j["value"])


def query(index, q, params):
    payload = {"index": index, "query_type": "count", "query": q, **params}
    delay = 30.0
    for attempt in range(6):
        try:
            r = requests.post(API, json=payload, timeout=120)
            d = r.json()
            if "error" not in d and d.get("count") is not None:
                return d, attempt
            if "error" in d:
                err = d["error"]
                if attempt >= 1:
                    return d, attempt  # a query error, not a transient one: retry once only
            else:  # e.g. HTTP 403 {"message": "Forbidden"}: throttled; back off
                err = f"HTTP {r.status_code} {d}"
        except Exception as ex:  # network or JSON failure
            err = repr(ex)
        print(f"  retry {attempt + 1}: {err}", file=sys.stderr)
        time.sleep(delay)
        delay *= 2
    return {"error": err}, 6


def main():
    jobs = plan()
    done = set()
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            if r.get("count") is not None:  # failed rows are re-fetched; the analysis keeps the last row per key
                done.add(key(r))
    todo = [j for j in jobs if key(j) not in done]
    print(f"{len(jobs)} jobs, {len(done)} done, {len(todo)} to go", flush=True)
    last = 0.0
    with open(OUT, "a") as f:
        for i, j in enumerate(todo):
            wait = MIN_GAP_S - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            d, retries = query(j["index"], j["query"], j["params"])
            row = {**j, "count": d.get("count"), "approx": d.get("approx"), "latency_ms": d.get("latency"),
                   "error": d.get("error"), "retries": retries,
                   "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)} {j['index']} {j['form']} {j['entity_id']} -> {row['count']}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
