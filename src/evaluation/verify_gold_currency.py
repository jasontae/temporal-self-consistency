"""D8: check the test split's gold answers against live Wikidata.

CLAIMS-LEDGER C5 established a structural fact from the frozen snapshot alone:
every test row's gold value carries a `t_end` of 2023 or 2024, and 91.3% are
orphans -- the last value the dataset records, marked as ended, with no
successor. Two spot checks (YouTube CEO, News Corp chair) showed the base model
being scored wrong for giving the *actual current* answer.

What C5 could not establish is the **rate**. That needs ground truth outside the
frozen snapshot, which is what this script fetches.

## Method

For each unique (entity, property) in the test split, ask Wikidata for the value
that holds *now*: among non-deprecated claims, the one with no `P582` (end date)
qualifier, preferring `preferred` rank and otherwise the latest `P580` (start
date). Compare that to the dataset's gold label.

Three outcomes per pair:
  - `gold_is_current`   dataset gold == live current value. Label is fine.
  - `gold_is_stale`     live current value differs. **The dataset is asking a
                        present-tense question with an outdated answer, so a
                        model answering correctly is scored wrong.**
  - `indeterminate`     Wikidata has no unambiguous current value (no claim
                        without an end date, entity missing, property absent).
                        Reported separately and never counted as either --
                        per the ledger, insufficient evidence is its own outcome.

## Why the query date is pinned in the output

"Current" is only meaningful relative to a date. The output records
`queried_at_utc` so this audit can be re-run and compared. Raw API responses are
cached to disk so the aggregate can be recounted from them later, per the
ledger's raw-artifact commitment.

Usage:
    python3 src/evaluation/verify_gold_currency.py --out data/prep/gold_currency_audit.json
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

API = "https://www.wikidata.org/w/api.php"
UA = "TSCT-benchmark-audit/0.1 (academic research; github.com/edward-lcl/temporal-self-consistency)"
BATCH = 25          # claim payloads are large; smaller batches are gentler
SLEEP = 1.5         # 0.2s earned an immediate HTTP 429


def api_get(params, retries=6):
    """GET with backoff that honours Retry-After.

    Wikidata rate-limits 50-id claim batches quickly. Responses are cached by
    the callers, so a run interrupted here resumes rather than restarting.
    """
    params = dict(params, format="json")
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = int(exc.headers.get("Retry-After") or 0) or min(60, 5 * 2 ** attempt)
                print(f"    [429] backing off {wait}s", flush=True)
                time.sleep(wait)
                continue
            if attempt == retries - 1:
                raise
            time.sleep(min(30, 2 ** attempt))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"giving up after {retries} attempts: {url[:120]}")


def fetch_claims(entity_ids, cache_dir):
    """Batch-fetch claims, caching each batch so the run is resumable."""
    out = {}
    ids = sorted(entity_ids)
    for i in range(0, len(ids), BATCH):
        chunk = ids[i : i + BATCH]
        cache = cache_dir / f"claims_{i:05d}.json"
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            data = api_get({"action": "wbgetentities", "ids": "|".join(chunk),
                            "props": "claims"})
            cache.write_text(json.dumps(data))
            time.sleep(SLEEP)
        out.update(data.get("entities", {}))
        if (i // BATCH) % 10 == 0:
            print(f"  claims {i}/{len(ids)}", flush=True)
    return out


def fetch_labels(qids, cache_dir):
    out = {}
    ids = sorted(q for q in qids if q)
    for i in range(0, len(ids), BATCH):
        chunk = ids[i : i + BATCH]
        cache = cache_dir / f"labels_{i:05d}.json"
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            data = api_get({"action": "wbgetentities", "ids": "|".join(chunk),
                            "props": "labels", "languages": "en"})
            cache.write_text(json.dumps(data))
            time.sleep(SLEEP)
        for qid, ent in data.get("entities", {}).items():
            lab = ent.get("labels", {}).get("en", {}).get("value")
            if lab:
                out[qid] = lab
        if (i // BATCH) % 10 == 0:
            print(f"  labels {i}/{len(ids)}", flush=True)
    return out


def current_value(entity, prop):
    """QID of the value holding *now*, or None if not determinable.

    Rule: drop deprecated ranks; keep claims with no P582 end qualifier; prefer
    `preferred` rank; otherwise take the latest P580 start. Returning None is a
    real outcome (`indeterminate`), not a failure to be papered over.
    """
    claims = (entity or {}).get("claims", {}).get(prop, [])
    live = []
    for c in claims:
        if c.get("rank") == "deprecated":
            continue
        snak = c.get("mainsnak", {})
        if snak.get("snaktype") != "value":
            continue
        qid = snak.get("datavalue", {}).get("value", {}).get("id")
        if not qid:
            continue
        quals = c.get("qualifiers", {})
        if "P582" in quals:  # has an end date -> not current
            continue
        start = None
        for q in quals.get("P580", []):
            t = q.get("datavalue", {}).get("value", {}).get("time")
            if t:
                start = t
        live.append((c.get("rank") == "preferred", start or "", qid))
    if not live:
        return None
    live.sort(reverse=True)  # preferred first, then latest start
    return live[0][2]


def norm(s):
    return " ".join(str(s).lower().split())


def main():
    from datasets import load_dataset

    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="data/prep/gold_currency_audit.json")
    p.add_argument("--cache-dir", default="data/prep/wikidata_cache")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("jasontae/temporal-delta", split=args.split)
    pairs = {}
    for r in ds:
        pairs.setdefault((r["entity_id"], r["property"]), r)
    print(f"[d8] {len(ds)} rows, {len(pairs)} unique (entity,property) pairs", flush=True)

    entities = {e for e, _ in pairs}
    print(f"[d8] fetching claims for {len(entities)} entities", flush=True)
    claims = fetch_claims(entities, cache_dir)

    wanted, resolved = set(), {}
    for (ent, prop) in pairs:
        qid = current_value(claims.get(ent), prop)
        resolved[(ent, prop)] = qid
        if qid:
            wanted.add(qid)
    print(f"[d8] resolving {len(wanted)} value labels", flush=True)
    labels = fetch_labels(wanted, cache_dir)

    rows, counts = [], Counter()
    for (ent, prop), r in pairs.items():
        qid = resolved[(ent, prop)]
        live = labels.get(qid) if qid else None
        if not live:
            verdict = "indeterminate"
        elif norm(live) == norm(r["value_label"]):
            verdict = "gold_is_current"
        else:
            verdict = "gold_is_stale"
        counts[verdict] += 1
        rows.append({
            "entity_id": ent, "property": prop, "question": r["question"],
            "dataset_gold": r["value_label"], "t_start": r["t_start"], "t_end": r["t_end"],
            "wikidata_current_qid": qid, "wikidata_current_label": live,
            "verdict": verdict,
        })

    n = len(rows)
    decided = counts["gold_is_current"] + counts["gold_is_stale"]
    print(f"\n[d8] === RESULT ({n} unique pairs) ===")
    for k in ("gold_is_current", "gold_is_stale", "indeterminate"):
        print(f"   {k:18s} {counts[k]:5d}  {100*counts[k]/n:5.1f}%")
    if decided:
        print(f"\n   Of pairs where Wikidata gives an unambiguous current value "
              f"(n={decided}):")
        print(f"   STALE GOLD RATE = {counts['gold_is_stale']}/{decided} = "
              f"{100*counts['gold_is_stale']/decided:.1f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "dataset": "jasontae/temporal-delta",
        "n_pairs": n,
        "counts": dict(counts),
        "stale_rate_among_decided": (counts["gold_is_stale"] / decided) if decided else None,
        "rows": rows,
    }, indent=2))
    print(f"\n[d8] wrote {out}")


if __name__ == "__main__":
    main()
