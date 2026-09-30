"""Fit the temperature that minimises NLL on the audit paragraph.

A correctly run model sits near T = 1. A logit-scale error (an extra factor on
the logits) leaves greedy decoding unchanged but sharpens every distribution,
so the best T moves far from 1 and NLL at T = 1 is inflated.

Usage: python3 diag_temperature.py --base <path> --name <tag>
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.training.generate_predictions import _hf_tok, _logits, load_base_only  # noqa: E402
from audit_models import PARAGRAPH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    model, tok, _, _ = load_base_only(a.base)
    hf = _hf_tok(tok)
    bos = getattr(tok, "bos_token_id", None) or getattr(hf, "bos_token_id", None)
    ids = hf(PARAGRAPH, add_special_tokens=True)["input_ids"]
    if bos is not None and ids[:1] != [bos]:
        ids = [bos] + ids
    lg = _logits(model(mx.array([ids])))[0, :-1].astype(mx.float32)
    tgt = mx.array(ids[1:])
    out = {}
    for T in [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12]:
        l = lg / T
        lp = mx.take_along_axis(l - mx.logsumexp(l, axis=-1, keepdims=True), tgt[:, None], axis=-1)
        out[T] = -float(mx.mean(lp).item())
    std = float(mx.mean(mx.std(lg, axis=-1)).item())
    best = min(out, key=out.get)
    rec = {"name": a.name, "nll_by_T": out, "best_T": best, "logit_std_mean": std,
           "logit_max_mean": float(mx.mean(mx.max(lg, axis=-1)).item())}
    json.dump(rec, open(Path(__file__).parent / "runs" / f"temperature_{a.name}.json", "w"), indent=2)
    print(json.dumps({"name": a.name, "T1": round(out[1], 3), "best_T": best, "best": round(out[best], 3),
                      "logit_std": round(std, 2), "logit_max": round(rec["logit_max_mean"], 2)}))


if __name__ == "__main__":
    main()
