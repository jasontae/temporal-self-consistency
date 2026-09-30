"""Do teacher-forced (full-sequence) logits agree with prefix-only forward passes?

For position t, log p(x_t | x_<t) from one forward pass over the whole
sentence should equal the same quantity from a forward pass over x_<=t-1 alone
(causal masking). A large gap means the multi-token forward pass is wrong,
which would corrupt every teacher-forced score the matched set uses while
leaving greedy generation (last position only) intact.

Usage: python3 diag_positions.py --base <path> --name <tag>
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.training.generate_predictions import _hf_tok, _logits, load_base_only  # noqa: E402

TEXT = "The river rises in the hills to the north and flows south through farmland before reaching the sea."


def lp_at(logits, pos, tok):
    row = logits[0, pos].astype(mx.float32)
    return (row[tok] - mx.logsumexp(row)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    model, tok, _, _ = load_base_only(a.base)
    hf = _hf_tok(tok)
    bos = getattr(tok, "bos_token_id", None) or getattr(hf, "bos_token_id", None)
    ids = hf(TEXT, add_special_tokens=True)["input_ids"]
    if bos is not None and ids[:1] != [bos]:
        ids = [bos] + ids
    full = _logits(model(mx.array([ids])))
    rows = []
    for t in range(1, len(ids)):
        pre = _logits(model(mx.array([ids[:t]])))
        rows.append({"tok": hf.decode([ids[t]]), "full": lp_at(full, t - 1, ids[t]),
                     "prefix": lp_at(pre, t - 1, ids[t])})
    gap = max(abs(r["full"] - r["prefix"]) for r in rows)
    out = {"name": a.name, "n_tokens": len(ids), "max_abs_gap": gap,
           "mean_nll_full": -sum(r["full"] for r in rows) / len(rows),
           "mean_nll_prefix": -sum(r["prefix"] for r in rows) / len(rows), "rows": rows}
    json.dump(out, open(Path(__file__).parent / "runs" / f"positions_{a.name}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))


if __name__ == "__main__":
    main()
