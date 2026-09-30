"""Knowledge probe for the staleness result (PLAN_knowledge_probe.md, 2026-09-30).

For each entity in the matched set, ask the model who holds the office now, and
separately let it complete the scoring frame "The ROLE of ENTITY is". Greedy
decoding, one model in memory. The answers are labelled current / stale / other
later, by knowledge_analysis.py; this script only generates.

    python3 docs/orientation/results_sep10/knowledge_probe/knowledge_probe.py \
        --base ~/.oMLX/models/Qwen3.5-27B-4bit --tag qwen35_27b
    python3 .../knowledge_probe.py --adapter data/prep/tcl_mlx_7b/tsct_seed1/adapter_fixed --tag tsct
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from src.training.generate_predictions import (  # noqa: E402
    _hf_tok, _logits, _stop_ids, load_base_only, rebuild_adapter)

CLAIMS = REPO / "data" / "stress_tests" / "matched_regime_claims_disjoint.jsonl"
OUT_DIR = REPO / "data" / "prep" / "predictions_7b"
QUESTION = "Who is the current {role} of {entity}? Answer with only the name, no explanation."
# Two fixed examples (an immutable fact and a long-held office, neither in the set) so the
# completion is a name rather than prose; the probed line is the scoring frame itself.
FRAME_SHOTS = "The founder of Microsoft is Bill Gates.\nThe CEO of Tesla is Elon Musk.\n"
FRAME = "The {role} of {entity} is"


def qa_prompt_ids(hf_tok, question, allow_think=False):
    """Chat prompt that starts the model on its answer (as answer_prompt_ids in
    score_evergreen_uncertainty), plus an empty think block for checkpoints whose
    template opens one regardless of enable_thinking (Qwen3-4B-Thinking-2507)."""
    text = hf_tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
    if "<|start|>assistant" in text and text.rstrip().endswith("<|start|>assistant"):
        text = text.rstrip() + "<|channel|>final<|message|>"  # gpt-oss harmony
    if text.rstrip().endswith("<think>") and not allow_think:
        text = text.rstrip() + "\n\n</think>\n\n"
    return hf_tok(text, add_special_tokens=False)["input_ids"], text


def frame_prompt_ids(hf_tok, text, bos_id):
    ids = hf_tok(text, add_special_tokens=True)["input_ids"]
    if bos_id is not None and (not ids or ids[0] != bos_id):
        ids = [bos_id] + ids
    return ids


def greedy(model, hf_tok, ids, stops, max_tokens, keep_special=False):
    cache = make_prompt_cache(model)
    logits = _logits(model(mx.array([ids]), cache=cache))[:, -1, :]
    out = []
    for _ in range(max_tokens):
        nxt = int(mx.argmax(logits, axis=-1).item())
        if nxt in stops:
            break
        out.append(nxt)
        logits = _logits(model(mx.array([[nxt]]), cache=cache))[:, -1, :]
    return hf_tok.decode(out, skip_special_tokens=not keep_special).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base")
    p.add_argument("--adapter")
    p.add_argument("--tag", required=True)
    p.add_argument("--bos", choices=("auto", "off"), default="auto")
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--think-budget", type=int, default=0,
                   help="for checkpoints that always reason (Qwen3-4B-Thinking-2507): let the model "
                        "think for up to this many tokens and keep the text after </think>")
    args = p.parse_args()
    if bool(args.base) == bool(args.adapter):
        p.error("pass exactly one of --base or --adapter")

    if args.base:
        model, tokenizer, hedge_ids, meta = load_base_only(args.base)
        stops = set(meta["stop_ids"])
    else:
        model, tokenizer, hedge_ids, cfg = rebuild_adapter(args.adapter)
        stops = set(_stop_ids(cfg["model"], _hf_tok(tokenizer)))
    hf_tok = _hf_tok(tokenizer)
    if hf_tok.eos_token_id is not None:
        stops.add(int(hf_tok.eos_token_id))
    for t in ("<|im_end|>", "<end_of_turn>", "<|return|>", "<|end|>", "<|eot_id|>"):
        tid = hf_tok.get_vocab().get(t)
        if tid is not None:
            stops.add(int(tid))
    if hedge_ids:  # the TSCT adapter ends its answer with a hedge token
        stops.update(int(h) for h in hedge_ids)
    bos_id = getattr(tokenizer, "bos_token_id", None) or getattr(hf_tok, "bos_token_id", None)
    if args.bos == "off":
        bos_id = None

    claims = [json.loads(l) for l in open(CLAIMS)]
    ents = {}
    for c in claims:
        e = ents.setdefault(c["entity_id"], {"entity_id": c["entity_id"], "entity": c["entity"]})
        if c["arm"] == "volatile_current":
            e["role"], e["current"] = c["role"], c["person"]
        elif c["arm"] == "volatile_stale":
            e["stale"] = c["person"]
    rows = list(ents.values())[: args.limit]

    _, sample = qa_prompt_ids(hf_tok, QUESTION.format(role=rows[0]["role"], entity=rows[0]["entity"]))
    print(f"[kp] {args.tag}: bos={bos_id} stops={sorted(stops)[:8]}... prompt tail={sample[-80:]!r}", flush=True)

    out_path = OUT_DIR / f"knowledge_{args.tag}.jsonl"
    t0 = time.time()
    with open(out_path, "w") as f:
        for i, e in enumerate(rows):
            q = QUESTION.format(role=e["role"], entity=e["entity"])
            qa_ids, _ = qa_prompt_ids(hf_tok, q, allow_think=args.think_budget > 0)
            if args.think_budget:
                raw = greedy(model, hf_tok, qa_ids, stops, args.think_budget, keep_special=True)
                qa = raw.split("</think>", 1)[1].strip() if "</think>" in raw else ""
                qa = qa.replace("<|im_end|>", "").strip()
                extra = {"think_truncated": "</think>" not in raw, "think_chars": len(raw)}
            else:
                qa = greedy(model, hf_tok, qa_ids, stops, args.max_tokens)
                extra = {}
            fr_text = FRAME_SHOTS + FRAME.format(role=e["role"], entity=e["entity"])
            fr = greedy(model, hf_tok, frame_prompt_ids(hf_tok, fr_text, bos_id), stops, args.max_tokens)
            fr = fr.split("\n", 1)[0].strip()  # the probed line only
            f.write(json.dumps({**e, "tag": args.tag, "qa_answer": qa, "frame_completion": fr, **extra}) + "\n")
            if (i + 1) % 20 == 0:
                print(f"[kp] {args.tag} {i+1}/{len(rows)} ({time.time()-t0:.0f}s) | {e['entity']}: "
                      f"qa={qa!r} frame={fr!r}", flush=True)
    print(f"[kp] {args.tag}: wrote {len(rows)} rows to {out_path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
