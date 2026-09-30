"""B11: can the model tell which regime a claim is in, within one passage?

CLAIMS-LEDGER B11 established that abstention is regime-dependent -- worth
everything on volatile facts (61% precision at 1% coverage against a 3.9% base
rate) and worth nothing on stable ones (accuracy already 87.8%). The two regimes
demand opposite policies, so a router's first job is deciding which regime it is
in. That is a per-item judgement, because real documents mix both.

`stress_mixed_paragraphs.jsonl` is the only artifact that puts both regimes
inside one input: 60 passages, 233 claims, 111 fast / 93 slow / 29 immutable.
It was built for exactly this and has never been run.

## What is measured

Mean token log-probability of each claim, scored standalone. The hypothesis is a
surprisal asymmetry:

  - A **stable** claim ("Carbon has an atomic number of 6") states something the
    model firmly believes, so it should be *unsurprising* -- high logprob.
  - A **volatile** claim states a value that may have changed since the model's
    cutoff. Where the claim carries the *current* value and the model believes a
    stale one, it should be *surprising* -- low logprob.

If that separation exists **within a passage**, the model already carries the
regime signal a router would need, with no training and no hedge vocabulary.

## Why within-passage is the control that matters

Comparing stable and volatile claims across different passages confounds
volatility with topic, register and length. Every passage here contains both, so
the within-passage AUROC holds the document fixed and asks only whether the model
separates the two kinds of claim inside it. The pooled figure is reported too,
but the within-passage number is the one that means something -- this is the same
distinction that collapsed the apparent 0.67 discrimination to 0.4997 in B5.

Scored standalone rather than conditioned on the passage prefix, so that a
claim's position in the paragraph cannot influence its score. That trades
deployment realism for a clean measurement; conditioning is the obvious follow-up.

Usage:
    python -m src.training.score_mixed_paragraphs \
        --base /Users/edward/.oMLX/models/gemma-4-26b-a4b-it-MLX-4bit \
        --out data/prep/predictions_7b/mixed_gemma4_26b.jsonl
"""
import argparse
import json
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from .generate_predictions import _hf_tok, _logits, load_base_only, rebuild_adapter

REPO_ROOT = Path(__file__).resolve().parents[2]
MIXED = REPO_ROOT / "data" / "stress_tests" / "stress_mixed_paragraphs.jsonl"


def claim_logprob(model, hf_tok, text):
    """Mean per-token log-probability of `text` under the model, teacher-forced."""
    ids = hf_tok(text, add_special_tokens=True)["input_ids"]
    if len(ids) < 2:
        return None, 0
    arr = mx.array([ids])
    logits = _logits(model(arr))[:, :-1, :].astype(mx.float32)
    targets = mx.array([ids[1:]])
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tok_lp = mx.take_along_axis(logprobs, targets[..., None], axis=-1)[0, :, 0]
    mx.eval(tok_lp)
    vals = tok_lp.tolist()
    return sum(vals) / len(vals), len(vals)


def auroc(pairs):
    """P(score of a positive > score of a negative). None if a class is absent."""
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    w = t = 0
    for a in pos:
        for b in neg:
            if a > b:
                w += 1
            elif a == b:
                t += 1
    return (w + 0.5 * t) / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base")
    p.add_argument("--adapter")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if bool(args.base) == bool(args.adapter):
        p.error("pass exactly one of --base or --adapter")

    if args.base:
        model, tokenizer, _, cfg = load_base_only(args.base)
        label = args.base.rstrip("/").split("/")[-1]
    else:
        model, tokenizer, _, cfg = rebuild_adapter(args.adapter)
        label = args.adapter.rstrip("/").split("/")[-2]
    hf_tok = _hf_tok(tokenizer)

    passages = [json.loads(l) for l in open(MIXED)]
    rows = []
    t0 = time.time()
    for pi, psg in enumerate(passages):
        for claim in psg["claims"]:
            lp, ntok = claim_logprob(model, hf_tok, claim["text"])
            if lp is None:
                continue
            vol = claim.get("volatility", "?")
            rows.append({
                "passage_idx": pi,
                "text": claim["text"],
                "volatility": vol,
                "expected_hedge": claim.get("expected_hedge"),
                "is_volatile": vol == "fast",
                "mean_logprob": lp,
                "n_tokens": ntok,
                "model": label,
            })
        if (pi + 1) % 15 == 0:
            print(f"[mixed] {pi+1}/{len(passages)} passages ({time.time()-t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # pooled: confounds volatility with topic/register across passages
    pooled = auroc([(r["mean_logprob"], not r["is_volatile"]) for r in rows])

    # within-passage: holds the document fixed, which is the claim that matters
    by_p = defaultdict(list)
    for r in rows:
        by_p[r["passage_idx"]].append(r)
    scored = []
    for pi, g in by_p.items():
        a = auroc([(r["mean_logprob"], not r["is_volatile"]) for r in g])
        if a is not None:
            scored.append((a, len(g)))
    tot = sum(n for _, n in scored)
    within = sum(a * n for a, n in scored) / tot if tot else float("nan")

    print(f"\n[mixed] {label}: {len(rows)} claims across {len(by_p)} passages")
    for vol in ("fast", "slow", "immutable"):
        v = [r["mean_logprob"] for r in rows if r["volatility"] == vol]
        if v:
            print(f"   {vol:10s} n={len(v):3d}  mean logprob {st.mean(v):+.4f}")
    print(f"\n   AUROC(logprob -> claim is STABLE), pooled          = {pooled:.4f}")
    print(f"   AUROC(logprob -> claim is STABLE), WITHIN passage  = {within:.4f}"
          f"   over {len(scored)} passages with both classes, n={tot}")
    print(f"\n[mixed] wrote {out}")


if __name__ == "__main__":
    main()
