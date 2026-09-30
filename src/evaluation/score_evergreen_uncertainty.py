"""Adjudicate Pletenev et al. (EMNLP 2025) §4.2 under a stricter control.

Their Table 3 reports Pearson correlation between evergreen-ness and two
uncertainty measures over generated answers, finding r = 0.17-0.35 and "a weak
trend suggesting that larger models correlate more strongly with evergreen-ness,
possibly indicating a greater internal reliance on temporal cues."

Our matched within-entity result points the other way: the regime signal is
present in small and old models (Qwen2.5-7B, AUROC 0.6336, p=0.0046) and gone in
large modern ones (Qwen3.5-27B 0.5206 at p=1.0000; Qwen3.6-27B 0.5097;
Qwen3.6-35B 0.5053). The two findings are measured differently, so neither
refutes the other as stated. This script puts them on one axis.

Two things are computed per question, on the model's own greedy generation,
following their definitions exactly:

    PPL      exp(-1/T sum_t log p(x_t | x_<t))
    entropy  -1/T sum_t sum_w p_t(w) log p_t(w)

and then reported three ways:

1. **Pooled Pearson r** -- replicates their Table 3 so the numbers are
   comparable to the published ones.
2. **Pooled AUROC** -- same signal on the scale the rest of our ledger uses.
3. **AUROC within form stratum** -- the stricter control. Our screen showed a
   past-tense marker alone separates their classes at AUROC 0.6640 with no
   model consulted, so an uncertainty signal that merely tracks question form
   is not evidence of internal temporal representation. Stratifying by that
   marker holds form fixed and asks what the signal adds beyond it. This is the
   same move that collapsed our own B5 (0.6700 -> 0.4997) and B11.

Usage:
    python3 -m src.evaluation.score_evergreen_uncertainty \
        --base /path/to/model --data <dir with test.csv> \
        --out data/prep/predictions_7b/eg_qwen25_7b.jsonl
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from ..training.generate_predictions import (
    _hf_tok, _logits, _stop_ids, load_base_only,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 20260812
N_PER_CLASS = 200
MAX_TOKENS = 32


def answer_prompt_ids(hf_tok, question):
    """Chat prompt that starts the model on its answer, not on a reasoning trace.

    Reasoning checkpoints otherwise spend the whole 32-token budget on a
    preamble ("Thinking Process:", gpt-oss "analysis..."), so the uncertainty
    measured is the preamble's (results_sep10/MODELS.md). Qwen3.x templates
    take `enable_thinking=False`; gpt-oss's harmony format is steered into its
    `final` channel directly.
    """
    msgs = [{"role": "user", "content": question}]
    text = hf_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
    if "<|start|>assistant" in text and text.rstrip().endswith("<|start|>assistant"):
        text = text.rstrip() + "<|channel|>final<|message|>"
    return hf_tok(text, add_special_tokens=False)["input_ids"]


def generate_with_uncertainty(model, hf_tok, question, stops, max_tokens=MAX_TOKENS,
                              answer_only=False):
    """Greedy decode; return (text, ppl, mean_entropy, n_tokens).

    Entropy is the full-vocabulary Shannon entropy of the predictive
    distribution at each generated position, averaged over the sequence --
    Pletenev et al.'s `Mean Token Entropy`. Perplexity uses the log-probability
    of the token actually emitted.
    """
    if answer_only:
        prompt_ids = answer_prompt_ids(hf_tok, question)
    else:
        prompt_ids = list(hf_tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        ))
    cache = make_prompt_cache(model)
    logits = _logits(model(mx.array([prompt_ids]), cache=cache))[:, -1, :]

    gen, logps, ents = [], [], []
    for _ in range(max_tokens):
        lg = logits[0].astype(mx.float32)
        logp = lg - mx.logsumexp(lg)
        nxt = int(mx.argmax(lg).item())
        if nxt in stops:
            break
        ent = float((-mx.sum(mx.exp(logp) * logp)).item())
        logps.append(float(logp[nxt].item()))
        ents.append(ent)
        gen.append(nxt)
        logits = _logits(model(mx.array([[nxt]]), cache=cache))[:, -1, :]

    if not logps:
        return "", None, None, 0
    mean_lp = sum(logps) / len(logps)
    return (hf_tok.decode(gen, skip_special_tokens=True).strip(),
            math.exp(-mean_lp), sum(ents) / len(ents), len(logps))


def balanced_subset(rows, n_per_class=N_PER_CLASS, seed=SEED):
    """Deterministic balanced sample, matching their 200/200 protocol."""
    rng = random.Random(seed)
    out = []
    for lab in (0, 1):
        pool = [r for r in rows if r["is_evergreen"] == lab]
        out.extend(pool if len(pool) <= n_per_class
                   else rng.sample(pool, n_per_class))
    out.sort(key=lambda r: r["idx"])
    return out


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", help="dir with their test.csv")
    ap.add_argument("--subset-from",
                    help="reuse the exact 400 questions of an earlier eg_*.jsonl run "
                         "(the original test.csv copy is no longer on disk)")
    ap.add_argument("--answer-only", action="store_true",
                    help="suppress reasoning preambles (see answer_prompt_ids)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if bool(args.data) == bool(args.subset_from):
        ap.error("pass exactly one of --data or --subset-from")

    if args.subset_from:
        sub = [{k: json.loads(l)[k] for k in ("idx", "question", "is_evergreen")}
               for l in open(args.subset_from)]
        sub.sort(key=lambda r: r["idx"])
    else:
        df = pd.read_csv(Path(args.data) / "test.csv")
        rows = [{"idx": i, "question": str(q), "is_evergreen": int(e)}
                for i, (q, e) in enumerate(zip(df["English"], df["is_evergreen"]))]
        sub = balanced_subset(rows)
    print(f"[eg] {len(sub)} questions "
          f"({sum(r['is_evergreen'] for r in sub)} evergreen)")

    model, tokenizer, _, _ = load_base_only(args.base)
    label = args.base.rstrip("/").split("/")[-1]
    hf_tok = _hf_tok(tokenizer)
    stops = set(_stop_ids(args.base, hf_tok) or [hf_tok.eos_token_id])

    out_rows = []
    t0 = time.time()
    for i, r in enumerate(sub):
        text, ppl, ent, ntok = generate_with_uncertainty(
            model, hf_tok, r["question"], stops, answer_only=args.answer_only)
        if ppl is None:
            continue
        out_rows.append({**r, "answer": text, "ppl": ppl,
                         "entropy": ent, "n_tokens": ntok, "model": label})
        if (i + 1) % 50 == 0:
            print(f"[eg] {i+1}/{len(sub)} ({time.time()-t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[eg] {label}: wrote {len(out_rows)} rows to {out}")


if __name__ == "__main__":
    main()
