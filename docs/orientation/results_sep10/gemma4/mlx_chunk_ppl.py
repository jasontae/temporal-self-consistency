"""G3 (independent implementation): mlx side of the llama.cpp perplexity comparison.

Replicates llama-perplexity's procedure on plain_text.txt: tokenize the whole
text with BOS, cut it into chunks of n_ctx tokens, put BOS at the start of each
chunk, and score only the second half of each chunk (positions n_ctx/2 .. n_ctx-2
predicting the next token). Reports PPL = exp(mean NLL) over the scored
positions, to compare with `llama-perplexity -c <n_ctx>` on the Q8_0 GGUF.

Usage (from repo root):
    python3 docs/orientation/results_sep10/gemma4/mlx_chunk_ppl.py --base <mlx dir> --name <tag> [--ctx 128]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.training.generate_predictions import _hf_tok, _logits, load_base_only  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ctx", type=int, default=128)
    a = ap.parse_args()
    model, tok, _, _ = load_base_only(a.base)
    hf = _hf_tok(tok)
    bos = getattr(tok, "bos_token_id", None) or getattr(hf, "bos_token_id", None)
    text = (HERE / "plain_text.txt").read_text()
    ids = hf(text, add_special_tokens=False)["input_ids"]
    if bos is not None:
        ids = [bos] + ids
    n = a.ctx
    nll, cnt, per_chunk = 0.0, 0, []
    for c in range(len(ids) // n):
        chunk = ids[c * n:(c + 1) * n]
        if bos is not None:
            chunk = [bos] + chunk[1:]
        lg = _logits(model(mx.array([chunk]))).astype(mx.float32)[0]
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        first = n // 2
        tgt = mx.array(chunk[first + 1:n])
        vals = mx.take_along_axis(lp[first:n - 1], tgt[:, None], axis=-1)[:, 0].tolist()
        per_chunk.append(-sum(vals) / len(vals))
        nll += -sum(vals)
        cnt += len(vals)
    res = {"name": a.name, "n_tokens": len(ids), "ctx": n, "chunks": len(per_chunk),
           "ppl": math.exp(nll / cnt), "mean_nll": nll / cnt, "per_chunk_nll": per_chunk}
    (HERE / "runs").mkdir(exist_ok=True)
    json.dump(res, open(HERE / "runs" / f"chunk_ppl_{a.name}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "per_chunk_nll"}))


if __name__ == "__main__":
    main()
