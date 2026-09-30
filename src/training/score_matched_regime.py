"""Score the matched, within-entity regime set (B11-X).

Replaces `score_mixed_paragraphs.py`, whose input was shown to be 22 distinct
claims with volatility confounded against grammatical form. Here every claim
shares one frame -- `The ROLE of ENTITY is PERSON.` -- and each entity
contributes all four arms, so the contrasts are within-entity.

Two numbers per claim:

  `mean_logprob`      mean per-token log-probability of the claim
  `name_logprob`      the same person in a neutral frame, "<name> is a person."

The second exists because founders skew historically famous while sitting
office-holders can be obscure, so raw surprisal partly tracks name frequency
rather than volatility. `pmi = mean_logprob - name_logprob` nets that out. A
separation visible only in the raw figure is a name-frequency effect, not a
regime effect, and must be reported as such.

Usage:
    python3 -m src.training.score_matched_regime \
        --base /Users/edward/.oMLX/models/Qwen2.5-7B-Instruct-MLX \
        --out data/prep/predictions_7b/matched_qwen25_7b.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from .generate_predictions import _hf_tok, _logits, load_base_only, rebuild_adapter

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIMS = REPO_ROOT / "data" / "stress_tests" / "matched_regime_claims.jsonl"


def claim_logprob(model, hf_tok, text, bos_id=None):
    """Mean per-token log-probability of `text`, teacher-forced.

    `bos_id` is prepended when the tokenizer declares one and it is not already
    present. Gemma is trained with `<bos>` on every sequence, but `_hf_tok`
    unwraps to the raw `tokenizers.Tokenizer`, whose post-processor does not
    fire -- so `add_special_tokens=True` silently returns an unprefixed
    sequence. Without it the whole Gemma family scores 2-4x more surprised than
    Qwen on identical text, and the true/false ordering inverts on some pairs.
    Qwen and gpt-oss declare no BOS and are unaffected.
    """
    ids = hf_tok(text, add_special_tokens=True)["input_ids"]
    if bos_id is not None and (not ids or ids[0] != bos_id):
        ids = [bos_id] + ids
    if len(ids) < 2:
        return None
    logits = _logits(model(mx.array([ids])))[:, :-1, :].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tok_lp = mx.take_along_axis(
        logprobs, mx.array([ids[1:]])[..., None], axis=-1)[0, :, 0]
    mx.eval(tok_lp)
    vals = tok_lp.tolist()
    return sum(vals) / len(vals)


CHAT_USER = "State one fact."


def chat_claim_logprob(model, hf_tok, text):
    """Mean log-prob of `text` as the assistant turn after a fixed neutral user
    turn, scoring only the claim's own tokens (the prompt control for item 4).
    The template supplies any BOS/turn markers the model was trained with."""
    prefix = hf_tok.apply_chat_template([{"role": "user", "content": CHAT_USER}],
                                        tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
    if "<|start|>assistant" in prefix and prefix.rstrip().endswith("<|start|>assistant"):
        prefix = prefix.rstrip() + "<|channel|>final<|message|>"  # gpt-oss harmony
    pre = hf_tok(prefix, add_special_tokens=False)["input_ids"]
    body = hf_tok(text, add_special_tokens=False)["input_ids"]
    ids = pre + body
    logits = _logits(model(mx.array([ids])))[:, :-1, :].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tok_lp = mx.take_along_axis(logprobs, mx.array([ids[1:]])[..., None], axis=-1)[0, :, 0]
    vals = tok_lp.tolist()[len(pre) - 1:]
    return sum(vals) / len(vals)


def context_claim_logprob(model, hf_tok, context, text, bos_id=None):
    """Mean log-prob of the claim's own tokens after a context prefix, e.g.
    "As of 2026, the CEO of Nike is Elliott Hill." scored on the tokens after
    "As of 2026," (the date-conditioning probe). The claim's leading capital is
    lowered so the sentence reads naturally; the context tokens are not scored."""
    body = " " + text[0].lower() + text[1:]
    pre = hf_tok(context, add_special_tokens=True)["input_ids"]
    if bos_id is not None and (not pre or pre[0] != bos_id):
        pre = [bos_id] + pre
    full = hf_tok(context + body, add_special_tokens=True)["input_ids"]
    if bos_id is not None and (not full or full[0] != bos_id):
        full = [bos_id] + full
    # score from where the context's tokenization ends (robust to merges at the seam)
    n_pre = len(pre)
    while n_pre > 1 and full[:n_pre] != pre[:n_pre]:
        n_pre -= 1
    logits = _logits(model(mx.array([full])))[:, :-1, :].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tok_lp = mx.take_along_axis(logprobs, mx.array([full[1:]])[..., None], axis=-1)[0, :, 0]
    vals = tok_lp.tolist()[n_pre - 1:]
    return sum(vals) / len(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base")
    p.add_argument("--adapter")
    p.add_argument("--out", required=True)
    p.add_argument("--claims", default=str(CLAIMS),
                   help="claim file to score (default: the original matched set)")
    p.add_argument("--frame", choices=("raw", "chat"), default="raw",
                   help="raw: score the claim as plain text (the Aug 13 protocol); chat: score it "
                        "as the assistant turn after a fixed user turn (prompt control)")
    p.add_argument("--context", default=None,
                   help='date-conditioning probe: prefix each claim with this context, e.g. "As of 2026,", '
                        'and score only the claim tokens; the neutral name frame is scored without it')
    p.add_argument("--context-field", default=None,
                   help="per-claim context: prefix each claim with the value of this field of its row "
                        "(PLAN_context_control.md); the neutral name frame is scored without it")
    p.add_argument("--bos", choices=("auto", "off"), default="auto",
                   help="auto: prepend the tokenizer's declared BOS (the Aug 13 behaviour); "
                        "off: never prepend. gpt-oss declares <|startoftext|> but is not "
                        "trained with it on raw text (results_sep10/model_audit).")
    args = p.parse_args()
    if bool(args.base) == bool(args.adapter):
        p.error("pass exactly one of --base or --adapter")

    if args.base:
        model, tokenizer, _, _ = load_base_only(args.base)
        label = args.base.rstrip("/").split("/")[-1]
    else:
        model, tokenizer, _, _ = rebuild_adapter(args.adapter)
        label = args.adapter.rstrip("/").split("/")[-2]
    hf_tok = _hf_tok(tokenizer)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        bos_id = getattr(hf_tok, "bos_token_id", None)
    if args.bos == "off":
        bos_id = None
    print(f"[matched] {label}: bos_token_id={bos_id} (--bos {args.bos})", flush=True)

    claims = [json.loads(l) for l in open(args.claims)]
    # the neutral frame repeats per person; score each distinct one once
    name_cache = {}
    rows = []
    t0 = time.time()
    for i, c in enumerate(claims):
        score = ((lambda t: chat_claim_logprob(model, hf_tok, t)) if args.frame == "chat"
                 else (lambda t: claim_logprob(model, hf_tok, t, bos_id)))
        ctx = c.get(args.context_field) if args.context_field else args.context
        lp = (context_claim_logprob(model, hf_tok, ctx, c["text"], bos_id)
              if ctx else score(c["text"]))
        if lp is None:
            continue
        nt = c["name_only_text"]
        if nt not in name_cache:
            name_cache[nt] = score(nt)
        nlp = name_cache[nt]
        rows.append({**c, "mean_logprob": lp, "name_logprob": nlp,
                     "pmi": None if nlp is None else lp - nlp,
                     "model": label})
        if (i + 1) % 100 == 0:
            print(f"[matched] {i+1}/{len(claims)} ({time.time()-t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[matched] {label}: wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
