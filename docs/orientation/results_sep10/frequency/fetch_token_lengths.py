"""PREREG amendment 1: Llama-2 token length of each person name, as tokenized by the
infini-gram API (the tokenization the counts are measured in). One count query per
distinct person on Dolma; >= 1 s between requests; resumable.

    python3 docs/orientation/results_sep10/frequency/fetch_token_lengths.py
"""
import json
import time
from pathlib import Path

from fetch_counts import CLAIMS, MIN_GAP_S, PRIMARY, query

OUT = Path(__file__).resolve().parent / "token_lengths.jsonl"


def main():
    persons = sorted({json.loads(l)["person"] for l in open(CLAIMS)
                      if json.loads(l)["arm"] in ("volatile_current", "volatile_stale")})
    done = {json.loads(l)["person"] for l in open(OUT)} if OUT.exists() else set()
    todo = [p for p in persons if p not in done]
    print(f"{len(persons)} persons, {len(todo)} to go", flush=True)
    last = 0.0
    with open(OUT, "a") as f:
        for p in todo:
            wait = MIN_GAP_S - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            d, _ = query(PRIMARY, p, {})
            f.write(json.dumps({"person": p, "index": PRIMARY, "n_tokens": len(d.get("token_ids") or []) or None,
                                "tokens": d.get("tokens"), "error": d.get("error")}, ensure_ascii=False) + "\n")
            f.flush()
    print("done", flush=True)


if __name__ == "__main__":
    main()
