"""Follow-up diagnostics for the model audit.

1. Where the surprisal sits: per-token log-probs of the audit paragraph scored
   as raw text (the matched-set harness's format), and the same text placed in
   the model's chat template as an assistant turn.
2. Which BOS setting reproduces the stored Aug 13 matched-set scores: rescore
   the first 12 claims of matched_regime_claims.jsonl with and without the BOS
   prefix and compare to data/prep/predictions_7b/<stored>.

Usage (from repo root):
    python3 docs/orientation/results_sep10/model_audit/diag_format.py --base <path> --name <tag> --stored matched_<x>.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.training.generate_predictions import _hf_tok, _logits, load_base_only  # noqa: E402
from src.training.score_matched_regime import claim_logprob  # noqa: E402
from audit_models import PARAGRAPH  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs"


def token_lps(model, ids):
    logits = _logits(model(mx.array([ids])))[:, :-1, :].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return mx.take_along_axis(lp, mx.array([ids[1:]])[..., None], axis=-1)[0, :, 0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--stored")
    args = ap.parse_args()
    model, tokenizer, _, _ = load_base_only(args.base)
    hf = _hf_tok(tokenizer)
    bos = getattr(tokenizer, "bos_token_id", None) or getattr(hf, "bos_token_id", None)
    rec = {"name": args.name, "bos": bos}

    raw = hf(PARAGRAPH, add_special_tokens=True)["input_ids"]
    if bos is not None and raw[:1] != [bos]:
        raw = [bos] + raw
    lps = token_lps(model, raw)
    toks = [hf.decode([t]) for t in raw[1:]]
    rec["raw_mean_nll"] = -sum(lps) / len(lps)
    rec["raw_mean_nll_excl_first3"] = -sum(lps[3:]) / len(lps[3:])
    rec["raw_worst_tokens"] = sorted(zip(lps, toks))[:8]

    msgs = [{"role": "user", "content": "Write a short description of a river town."},
            {"role": "assistant", "content": PARAGRAPH}]
    try:
        full = hf.apply_chat_template(msgs, tokenize=False)
        prefix = hf.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        ids_full = hf(full, add_special_tokens=False)["input_ids"]
        n_pre = len(hf(prefix, add_special_tokens=False)["input_ids"])
        lps_c = token_lps(model, ids_full)[n_pre - 1:]
        # score only the paragraph tokens (drop the closing turn markers)
        n_par = len(hf(PARAGRAPH, add_special_tokens=False)["input_ids"])
        lps_c = lps_c[:n_par]
        rec["chat_mean_nll"] = -sum(lps_c) / len(lps_c)
    except Exception as exc:
        rec["chat_error"] = f"{type(exc).__name__}: {exc}"

    if args.stored:
        stored = [json.loads(l) for l in open(ROOT / "data/prep/predictions_7b" / args.stored)][:12]
        cmp = []
        for r in stored:
            a = claim_logprob(model, hf, r["text"], bos)
            b = claim_logprob(model, hf, r["text"], None)
            cmp.append({"stored": r["mean_logprob"], "with_bos": a, "no_bos": b})
        rec["stored_match"] = {
            "max_abs_diff_with_bos": max(abs(c["stored"] - c["with_bos"]) for c in cmp),
            "max_abs_diff_no_bos": max(abs(c["stored"] - c["no_bos"]) for c in cmp),
            "rows": cmp,
        }
    json.dump(rec, open(OUT / f"diag_{args.name}.json", "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("stored_match", "raw_worst_tokens")}
                     | ({"stored_diff": {k: v for k, v in rec["stored_match"].items() if k != "rows"}}
                        if "stored_match" in rec else {}), default=str))
    print("worst raw tokens:", rec["raw_worst_tokens"][:5])


if __name__ == "__main__":
    main()
