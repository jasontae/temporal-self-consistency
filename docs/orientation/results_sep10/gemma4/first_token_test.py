"""Gemma-4 timebox, test 1: is the PMI degradation a first-token-after-BOS artifact?

Rescore the original matched set two ways with the harness's own tokenization and
BOS rule:
  standard     mean log-prob over all scored tokens (what score_matched_regime does)
  skip_first   the same, dropping the first scored token (the one right after BOS)
for both the claim and the neutral name frame, and report truth / regime /
staleness AUROC on raw and PMI scores. Controls: gemma-3-4B (BOS, healthy) and
Qwen3.5-27B (no BOS).

If Gemma-4's PMI truth AUROC recovers under skip_first while the controls do not
move, the first-token penalty (NOTES.md effect 1) explains the PMI degradation.

Usage (from repo root): python3 docs/orientation/results_sep10/gemma4/first_token_test.py --base <path> --name <tag>
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.training.generate_predictions import _hf_tok, _logits, load_base_only  # noqa: E402

CLAIMS = ROOT / "data" / "stress_tests" / "matched_regime_claims.jsonl"
OUT = Path(__file__).resolve().parent / "runs"


def token_lps(model, hf, text, bos):
    ids = hf(text, add_special_tokens=True)["input_ids"]
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids
    lg = _logits(model(mx.array([ids])))[:, :-1, :].astype(mx.float32)
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    return mx.take_along_axis(lp, mx.array([ids[1:]])[..., None], axis=-1)[0, :, 0].tolist()


def auroc(pos, neg):
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    model, tok, _, _ = load_base_only(a.base)
    hf = _hf_tok(tok)
    bos = getattr(tok, "bos_token_id", None) or getattr(hf, "bos_token_id", None)
    if "gpt-oss" in a.base:
        bos = None  # as in the Aug 13 runs (MODELS.md finding 2)
    cache = {}

    def lps(t):
        if t not in cache:
            cache[t] = token_lps(model, hf, t, bos)
        return cache[t]

    by = defaultdict(dict)
    for c in map(json.loads, open(CLAIMS)):
        by[c["entity_id"]][c["arm"]] = c
    ents = [e for e in by if len(by[e]) == 4]
    res = {"name": a.name, "bos": bos}
    for mode, k in (("standard", 0), ("skip_first", 1)):
        def score(c, field):
            v = lps(c[field])[k:]
            return sum(v) / len(v)
        S = {arm: {"raw": np.array([score(by[e][arm], "text") for e in ents]),
                   "pmi": np.array([score(by[e][arm], "text") - score(by[e][arm], "name_only_text") for e in ents])}
             for arm in ("immutable_true", "immutable_false", "volatile_current", "volatile_stale")}
        res[mode] = {f"{c}_{f}": float(auroc(S[h][f], S[l][f]))
                     for c, h, l in (("truth", "immutable_true", "immutable_false"),
                                     ("regime", "immutable_true", "volatile_current"),
                                     ("staleness", "volatile_current", "volatile_stale"))
                     for f in ("raw", "pmi")}
        res[mode]["first_token_nll_name_frame_sd"] = float(np.std([-lps(by[e][arm]["name_only_text"])[0]
                                                                   for e in ents for arm in by[e]]))
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / f"first_token_{a.name}.json", "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
