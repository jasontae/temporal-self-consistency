"""TCL training run on a 7-8B model via MLX + LoRA (Apple Silicon).

Scale-up of `run_tcl_diagnostic.py` (0.5B, PyTorch/MPS, full fine-tune) to
the target model class, using MLX so 4-bit quantised weights work on
unified memory -- the HF/bitsandbytes 4-bit path is CUDA-only and `peft`
is not installed here.

Keeps the diagnostic's telemetry contract exactly: one CSV row per step
with `ce`, `l_over`, `l_under`, `r_hedge`, `tcl_total`, `n_volatile`,
`pred_confident_frac` and `c_hat_grad_norm` as SEPARATE columns. Per the
debugging notes, the summed total is what let the original bug hide, and
per `specs/tcl-fix-validation.md` aggregate `tcl_total` drift is explicitly
NOT a reliable gate-check -- the isolated `c_hat_grad_norm` and the
hedge-collapse check are the load-bearing signals.

## No embedding resize

Qwen2.5 ships `config.vocab_size` (151936) larger than `len(tokenizer)`
(151665), leaving 271 unused embedding rows. The 4 hedge tokens are added
to the tokenizer and land on ids 151665-151668, which already have rows in
the (quantised) embedding matrix. So there is no `resize_token_embeddings`
step -- which is what makes the 4-bit path usable at all, since resizing a
quantised matrix is not well-defined. The runner asserts the ids fit
rather than assuming it.

Usage (smoke test on a small cached model):
    python -m src.training.run_tcl_mlx --model mlx-community/Qwen2.5-3B-Instruct-4bit \
        --n-per-volatility 16 --epochs 1 --both-conditions

Usage (real run):
    python -m src.training.run_tcl_mlx --model mlx-community/Qwen2.5-7B-Instruct-4bit \
        --n-per-volatility 2000 --epochs 3
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.utils import load

from .data import load_slice
from .hedge_tokens import RESERVED_HEDGE_TOKENS, PLAIN_CHAT_TEMPLATE, HEDGE_TOKENS, HEDGE_TO_CONFIDENCE
from .mlx_tcl import (
    compute_c_hat,
    conf_scalar_array,
    global_grad_norm,
    masked_ce,
    tcl_terms,
)

# Revised hyperparameters from docs/tcl_debugging.md, same as the 0.5B run.
EPOCHS = 3
LEARNING_RATE = 5e-5
LAMBDA_OVER = 0.5
LAMBDA_UNDER = 0.5
LAMBDA_HEDGE = 0.3
BATCH_SIZE = 4
MAX_LEN = 128

# LoRA config. Base model stays frozen/quantised; only adapters train.
LORA_RANK = 8
LORA_SCALE = 20.0
LORA_LAYERS = 16


def attach_hedge_tokens(tokenizer, model, mode="added"):
    """Add hedge tokens to the tokenizer WITHOUT resizing model embeddings.

    Relies on the spare rows Qwen2.5 already carries (see module docstring).
    Asserts both that the new ids fit inside the model's embedding matrix
    and that each hedge token is exactly one id -- the TCL loss indexes a
    single logit position per hedge token, so a multi-piece tokenisation
    would silently break it.
    """
    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    # Round-trip guard: a mis-declared tokenizer class can silently drop spaces
    # (Meta-Llama-3 loaded as LlamaTokenizer under transformers 5.8.1 encodes
    # "Mark Parker" as Mark,P,arker), which corrupts every training example.
    probe = "Mark Parker and Susan"
    assert hf_tok.decode(hf_tok(probe, add_special_tokens=False)["input_ids"]) == probe, (
        f"tokenizer {type(hf_tok).__name__} does not round-trip {probe!r}; fix tokenizer_class before training")
    if mode == "reserved":
        # No spare rows (Meta-Llama-3): reuse four reserved special tokens.
        names = RESERVED_HEDGE_TOKENS
    else:
        existing = set(hf_tok.get_vocab().keys())
        new_tokens = [t for t in HEDGE_TOKENS if t not in existing]
        if new_tokens:
            hf_tok.add_special_tokens({"additional_special_tokens": new_tokens})
        names = HEDGE_TOKENS

    hedge_ids = [hf_tok.convert_tokens_to_ids(t) for t in names]
    for tok, tid in zip(names, hedge_ids):
        pieces = hf_tok(tok, add_special_tokens=False)["input_ids"]
        assert len(pieces) == 1, f"hedge token {tok!r} tokenized to {len(pieces)} ids, expected 1"
        assert tid is not None, f"hedge token {tok!r} has no id"

    model_vocab = model.args.vocab_size
    assert max(hedge_ids) < model_vocab, (
        f"hedge ids {hedge_ids} exceed model vocab_size {model_vocab} -- this model has no "
        "spare embedding rows, so it would need a resize (not supported on quantised weights)"
    )
    return hedge_ids


def assert_hedge_head_is_trainable(model, hedge_ids, batch):
    """Fail fast if the 4 hedge logits cannot differ from one another.

    Qwen2.5's spare embedding rows are not fresh random vectors -- every
    unused row (151665 onward) is the SAME padding row, byte for byte. With
    a frozen LM head that is fatal: the 4 hedge tokens would produce
    identical logits for every hidden state, softmax would be exactly
    uniform, c_hat would be a constant, and argmax would return
    [CONFIDENT] forever. That looks exactly like the collapse failure mode
    we are trying to study, but is a dead parameterisation rather than a
    training result -- so it must be caught before a long run, not
    diagnosed from its output afterwards.

    Applying LoRA to `lm_head` gives each hedge row its own trainable
    low-rank component, which is what makes them separable. This check
    verifies that empirically instead of trusting the wiring.
    """
    logits = model(batch["input_ids"])
    hl = hedge_logits_from(logits, batch["hedge_positions"], hedge_ids)
    mx.eval(hl)
    spread = float((hl.max(axis=-1) - hl.min(axis=-1)).max().item())

    # The quantity under test is whether the hedge logits can ever DIFFER,
    # not whether they move together. Their sum has nonzero gradient even
    # with identical frozen rows (all four track the hidden state in
    # lockstep), so summing would pass a dead head -- it measures the wrong
    # thing entirely.
    #
    # Use a LINEAR contrast, logit[0] - logit[i]. Linearity matters: any
    # smooth symmetric measure (variance, spread) sits at a stationary point
    # when the logits coincide, so its gradient vanishes at exactly the
    # degenerate configuration we need to detect, and it too would pass.
    #
    #   frozen identical rows -> logit[0]-logit[i] == 0 for every input and
    #     every parameter value -> gradient is exactly 0.
    #   lm_head under LoRA     -> difference is h @ (B_0 - B_i) @ A, whose
    #     gradient w.r.t. B is nonzero even though B is zero-initialised
    #     (so the contrast is still 0 at init -- which is why this tests the
    #     gradient rather than the spread).
    def contrast(m, b):
        h = hedge_logits_from(
            m(b["input_ids"]), b["hedge_positions"], mx.array(hedge_ids)
        ).astype(mx.float32)
        return (h[:, 0:1] - h[:, 1:]).sum()

    _, g = nn.value_and_grad(model, contrast)(model, batch)
    mx.eval(g)
    gnorm = global_grad_norm(g)
    print(
        f"[mlx] hedge-head check: logit spread={spread:.6f} contrast_grad_norm={gnorm:.6f}",
        flush=True,
    )
    assert gnorm > 0.0, (
        "hedge logit CONTRASTS have zero gradient w.r.t. trainable params -- the LM head rows "
        "for the 4 hedge tokens are frozen duplicates of one padding row, so they produce "
        "identical logits for every hidden state and the model can never learn to distinguish "
        "them. softmax would be exactly uniform, c_hat constant, and argmax pinned to "
        "[CONFIDENT]. That mimics the collapse failure mode but is a dead parameterisation, "
        "not a training result. Re-run without --no-lora-lm-head."
    )
    return spread, gnorm


def build_examples(raw_examples, tokenizer, hedge_ids, max_len=MAX_LEN):
    """Tokenise into the same shape the torch TCLDataset produces.

    Format and hedge_position semantics are identical to `data.TCLDataset`
    so the 7B numbers are comparable to the 0.5B diagnostic: logits at
    `hedge_position` are what predict the hedge token.
    """
    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    hedge_id_by_token = dict(zip(HEDGE_TOKENS, hedge_ids))
    out = []
    for ex in raw_examples:
        prompt_ids = list(hf_tok.apply_chat_template(
            [{"role": "user", "content": ex.question}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        ))
        answer_ids = hf_tok(ex.answer + " ", add_special_tokens=False)["input_ids"]
        hedge_id = hedge_id_by_token[ex.hedge_token]
        eos_id = hf_tok.eos_token_id

        input_ids = prompt_ids + answer_ids + [hedge_id, eos_id]
        hedge_position = len(prompt_ids) + len(answer_ids) - 1
        if len(input_ids) > max_len:
            continue  # drop rather than truncate: truncation can cut the hedge token off
        out.append({
            "input_ids": input_ids,
            "n_prompt": len(prompt_ids),
            "hedge_position": hedge_position,
            "c_gold": HEDGE_TO_CONFIDENCE[ex.hedge_token],
            "volatile": 1.0 if ex.volatility in ("fast", "slow") else 0.0,
            "gold_idx": HEDGE_TOKENS.index(ex.hedge_token),
        })
    return out


def collate(batch, pad_id):
    """Pad to the batch max and build the shifted CE mask.

    loss_mask is aligned to the SHIFTED targets (position i predicts
    input_ids[i+1]), and covers the assistant turn only -- the torch
    equivalent of labels=-100 on the prompt.
    """
    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    input_ids = [[pad_id] * max_len for _ in range(B)]
    loss_mask = [[0.0] * (max_len - 1) for _ in range(B)]

    for i, b in enumerate(batch):
        ids = b["input_ids"]
        input_ids[i][: len(ids)] = ids
        for j in range(max_len - 1):
            # target at shifted index j is input_ids[j+1]; supervise the assistant turn
            if b["n_prompt"] <= j + 1 < len(ids):
                loss_mask[i][j] = 1.0

    return {
        "input_ids": mx.array(input_ids),
        "loss_mask": mx.array(loss_mask),
        "hedge_positions": mx.array([b["hedge_position"] for b in batch]),
        "c_gold": mx.array([b["c_gold"] for b in batch], dtype=mx.float32),
        "volatile_mask": mx.array([b["volatile"] for b in batch], dtype=mx.float32),
        "gold_idx": [b["gold_idx"] for b in batch],
    }


def hedge_logits_from(logits, hedge_positions, hedge_ids):
    """(B, L, V) -> (B, 4) logits at each example's hedge position."""
    B = logits.shape[0]
    rows = logits[mx.arange(B), hedge_positions, :]  # (B, V)
    return rows[:, mx.array(hedge_ids)]


def make_loss_fns(hedge_ids, conf_scalars, mode, lambdas):
    """Build (full_loss, c_hat_only_loss). Both close over the same batch.

    `c_hat_only` isolates lambda_over*L_over + lambda_under*L_under -- the
    two terms that are pure functions of c_hat. R_hedge is built straight
    from hedge_probs and stays differentiable in BOTH modes, so including
    it would mask the very signal under test.

    NOTE: `c_hat_only` deliberately uses the module-level REFERENCE lambdas
    (0.5/0.5), not the run's training lambdas. It is a diagnostic of whether
    gradient can flow from c_hat at all, which is a property of the graph,
    not of the loss weighting. Under the lambda=0 (SFT-only) ablation the
    training lambdas are zero, so weighting the probe by them would make it
    read exactly 0.0 -- the same signature as the `broken` bug, but for a
    completely unrelated reason. Holding the probe weights fixed keeps the
    two situations distinguishable.
    """
    broken = mode == "broken"
    lam_over, lam_under, lam_hedge = lambdas

    def full_loss(model, batch):
        logits = model(batch["input_ids"])
        ce = masked_ce(logits[:, :-1, :], batch["input_ids"][:, 1:], batch["loss_mask"])
        hl = hedge_logits_from(logits, batch["hedge_positions"], hedge_ids)
        c_hat, hedge_probs = compute_c_hat(hl, conf_scalars, broken=broken)
        terms = tcl_terms(c_hat, batch["c_gold"], hedge_probs, batch["volatile_mask"])
        calib = (
            lam_over * terms["l_over"]
            + lam_under * terms["l_under"]
            + lam_hedge * terms["r_hedge"]
        )
        total = ce + calib
        return total, (ce, terms["l_over"], terms["l_under"], terms["r_hedge"], hedge_probs)

    def c_hat_only(model, batch):
        logits = model(batch["input_ids"])
        hl = hedge_logits_from(logits, batch["hedge_positions"], hedge_ids)
        c_hat, hedge_probs = compute_c_hat(hl, conf_scalars, broken=broken)
        terms = tcl_terms(c_hat, batch["c_gold"], hedge_probs, batch["volatile_mask"])
        return LAMBDA_OVER * terms["l_over"] + LAMBDA_UNDER * terms["l_under"]

    return full_loss, c_hat_only


def eval_hedge_distribution(model, batches, hedge_ids):
    """Predicted-hedge distribution, per-example accuracy, and confusion matrix.

    The aggregate distribution alone is a weak metric and can be actively
    misleading here: a model that learned nothing but the LABEL MARGINAL
    reproduces the target distribution exactly while being at chance on any
    individual example. Since the gold hedge token is part of the supervised
    sequence, plain CE is perfectly capable of fitting that marginal.

    So `accuracy` (did it pick the right hedge for THIS fact?) and the
    gold->pred confusion matrix are the metrics that separate discrimination
    from prior-matching. Compare accuracy against the majority-class rate,
    which is the score a model gets for learning the prior alone.
    """
    counts = {t: 0 for t in HEDGE_TOKENS}
    confusion = {g: {p: 0 for p in HEDGE_TOKENS} for g in HEDGE_TOKENS}
    correct = 0
    total = 0
    for batch in batches:
        logits = model(batch["input_ids"])
        hl = hedge_logits_from(logits, batch["hedge_positions"], hedge_ids)
        preds = mx.argmax(hl, axis=-1).tolist()
        mx.eval(logits)
        for p, g in zip(preds, batch["gold_idx"]):
            counts[HEDGE_TOKENS[p]] += 1
            confusion[HEDGE_TOKENS[g]][HEDGE_TOKENS[p]] += 1
            correct += int(p == g)
            total += 1
    dist = {k: v / max(total, 1) for k, v in counts.items()}
    gold_counts = {g: sum(row.values()) for g, row in confusion.items()}
    majority = max(gold_counts.values()) / max(total, 1)
    return {
        "dist": dist,
        "accuracy": correct / max(total, 1),
        "majority_class_rate": majority,
        "confusion": confusion,
        "n": total,
    }


def run_condition(mode, args, raw_examples, log_rows, condition_meta):
    print(f"\n[mlx] === condition: {mode} ===", flush=True)
    model, tokenizer = load(args.model)
    hf_tok_ = getattr(tokenizer, "_tokenizer", tokenizer)
    if args.plain_prompt and not getattr(hf_tok_, "chat_template", None):
        hf_tok_.chat_template = PLAIN_CHAT_TEMPLATE  # base checkpoint without a template
    hedge_ids = attach_hedge_tokens(tokenizer, model, mode=args.hedge_tokens)
    print(f"[mlx] hedge ids (no resize, mode={args.hedge_tokens}): {hedge_ids}", flush=True)

    model.freeze()
    linear_to_lora_layers(
        model, args.lora_layers, {"rank": LORA_RANK, "scale": LORA_SCALE, "dropout": 0.0}
    )
    if args.lora_lm_head:
        # Required, not optional -- see assert_hedge_head_is_trainable().
        # Without an adapter here the hedge rows are frozen duplicates of one
        # padding row and can never separate.
        #
        # Two architectures to handle. Qwen2.5-7B has tie_word_embeddings=False
        # and a real `lm_head`. The smaller Qwen2.5 models (0.5B/3B) tie input
        # and output embeddings, so there is no lm_head at all and the output
        # projection IS `model.embed_tokens` -- adapting that covers both roles
        # at once. Getting this wrong is an AttributeError at startup rather
        # than a silent wrong result, but the tied path is the one the small
        # smoke-test models take, so both are wired up.
        if hasattr(model, "lm_head"):
            model.lm_head = LoRALinear.from_base(
                model.lm_head, r=LORA_RANK, scale=LORA_SCALE, dropout=0.0
            )
            head_kind = "lm_head (untied)"
        else:
            model.model.embed_tokens = LoRAEmbedding.from_base(
                model.model.embed_tokens, r=LORA_RANK, scale=LORA_SCALE, dropout=0.0
            )
            head_kind = "embed_tokens (tied)"
        print(f"[mlx] adapted output projection: {head_kind}", flush=True)
    print_trainable_parameters(model)

    hf_tok = getattr(tokenizer, "_tokenizer", tokenizer)
    pad_id = hf_tok.pad_token_id if hf_tok.pad_token_id is not None else hf_tok.eos_token_id
    examples = build_examples(raw_examples, tokenizer, hedge_ids)
    print(f"[mlx] {len(examples)} examples after tokenisation (max_len={MAX_LEN})", flush=True)

    eval_batches = [
        collate(examples[i : i + args.batch_size], pad_id)
        for i in range(0, len(examples) - args.batch_size + 1, args.batch_size)
    ]

    head_spread, head_gnorm = assert_hedge_head_is_trainable(model, hedge_ids, eval_batches[0])

    conf_scalars = conf_scalar_array()
    lambdas = (args.lambda_over, args.lambda_under, args.lambda_hedge)
    full_loss, c_hat_only = make_loss_fns(hedge_ids, conf_scalars, mode, lambdas)
    loss_and_grad = nn.value_and_grad(model, full_loss)
    chat_grad = nn.value_and_grad(model, c_hat_only)
    optimizer = optim.AdamW(learning_rate=args.lr)

    model.train()
    pre = eval_hedge_distribution(model, eval_batches, hedge_ids)
    print(f"[mlx] {mode} pre-train (n={pre['n']}) acc={pre['accuracy']:.4f} "
          f"majority={pre['majority_class_rate']:.4f} dist={pre['dist']}", flush=True)

    rng = random.Random(args.seed)
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            batch = collate([examples[j] for j in order[i : i + args.batch_size]], pad_id)

            # isolated gradient-connectivity probe: must be exactly 0 in
            # "broken", nonzero in "fixed". Sampled, not every step -- it
            # costs a second forward+backward, which is material at 7B.
            if step % args.grad_check_every == 0:
                _, g = chat_grad(model, batch)
                mx.eval(g)
                c_hat_grad_norm = global_grad_norm(g)
            else:
                c_hat_grad_norm = ""

            (total, aux), grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, total)

            ce, l_over, l_under, r_hedge, hedge_probs = aux
            pred_confident_frac = float(
                (mx.argmax(hedge_probs, axis=-1) == 0).astype(mx.float32).mean().item()
            )

            log_rows.append({
                "mode": mode,
                "epoch": epoch,
                "step": step,
                "ce": float(ce.item()),
                "l_over": float(l_over.item()),
                "l_under": float(l_under.item()),
                "r_hedge": float(r_hedge.item()),
                "tcl_total": float(total.item()),
                "n_volatile": int(batch["volatile_mask"].sum().item()),
                "pred_confident_frac": pred_confident_frac,
                "c_hat_grad_norm": c_hat_grad_norm,
            })
            if step % args.log_every == 0:
                print(
                    f"[mlx] {mode} step {step} ce={float(ce.item()):.4f} "
                    f"total={float(total.item()):.4f} grad={c_hat_grad_norm} "
                    f"conf_frac={pred_confident_frac:.3f} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
            step += 1

    wall = time.time() - t0
    post = eval_hedge_distribution(model, eval_batches, hedge_ids)
    print(f"[mlx] {mode} post-train (n={post['n']}) acc={post['accuracy']:.4f} "
          f"majority={post['majority_class_rate']:.4f} dist={post['dist']}", flush=True)
    lift = post["accuracy"] - post["majority_class_rate"]
    print(f"[mlx] {mode} accuracy lift over majority-class baseline: {lift:+.4f} "
          f"({'discriminating' if lift > 0.02 else 'NOT clearly better than learning the prior'})",
          flush=True)
    print(f"[mlx] {mode} finished: {step} steps in {wall:.1f}s", flush=True)

    # Persist the adapter. Without this the trained model is discarded at the
    # end of the run, so the eval pipeline (specs/tsct-project-state.md next
    # action 3) could not be pointed at it without retraining from scratch.
    try:
        adapter_dir = Path(args.out_dir) / f"adapter_{mode}"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), weights)
        with open(adapter_dir / "adapter_config.json", "w") as f:
            json.dump({
                "model": args.model, "lora_rank": LORA_RANK, "lora_scale": LORA_SCALE,
                "lora_layers": args.lora_layers, "lm_head_adapted": args.lora_lm_head,
                "hedge_token_ids": hedge_ids, "hedge_tokens": HEDGE_TOKENS,
                "hedge_token_mode": args.hedge_tokens,
                "chat_template_override": PLAIN_CHAT_TEMPLATE if args.plain_prompt else None,
            }, f, indent=2)
        print(f"[mlx] saved adapter -> {adapter_dir}", flush=True)
    except Exception as exc:  # never lose a finished run over a save failure
        print(f"[mlx] WARNING: adapter save failed ({exc!r}) -- metrics below are still valid",
              flush=True)

    condition_meta[mode] = {
        "n_steps": step,
        "wall_seconds": wall,
        "n_examples": len(examples),
        "pre_train": pre,
        "post_train": post,
        "accuracy_lift_over_majority": lift,
        "hedge_head_logit_spread_at_init": head_spread,
        "hedge_head_grad_norm_at_init": head_gnorm,
    }
    del model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    p.add_argument("--out-dir", default="data/prep/tcl_mlx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--lora-layers", type=int, default=LORA_LAYERS)
    p.add_argument("--lambda-over", type=float, default=LAMBDA_OVER)
    p.add_argument("--lambda-under", type=float, default=LAMBDA_UNDER)
    p.add_argument(
        "--lambda-hedge", type=float, default=LAMBDA_HEDGE,
        help="Set all three lambdas to 0 for the SFT-only ablation (CE loss only, no TCL) -- "
             "paper_draft.md calls this 'the critical ablation isolating the contribution of TCL'.",
    )
    p.add_argument("--n-per-volatility", type=int, default=2000)
    p.add_argument("--grad-check-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--no-lora-lm-head", dest="lora_lm_head", action="store_false",
        help="Do NOT adapt lm_head. Only for demonstrating the dead-head failure -- "
             "a real run will trip assert_hedge_head_is_trainable() without it.",
    )
    p.set_defaults(lora_lm_head=True)
    p.add_argument("--hedge-tokens", choices=("added", "reserved"), default="added",
                   help="added: new tokenizer ids on spare embedding rows (Qwen2.5); reserved: "
                        "four reserved special tokens, for models with no spare rows (Meta-Llama-3)")
    p.add_argument("--plain-prompt", action="store_true",
                   help="for base checkpoints with no chat template: 'Question: ...\\nAnswer:'")
    p.add_argument(
        "--both-conditions", action="store_true",
        help="also run the 'broken' ablation arm. Off by default: at 7B it doubles "
             "cost, and the bug is already characterised at 0.5B.",
    )
    args = p.parse_args()

    mx.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_examples = load_slice(n_per_volatility=args.n_per_volatility, seed=args.seed, strict=True)
    print(f"[mlx] loaded {len(raw_examples)} raw examples", flush=True)

    log_rows, condition_meta = [], {}
    modes = ("broken", "fixed") if args.both_conditions else ("fixed",)
    for mode in modes:
        run_condition(mode, args, raw_examples, log_rows, condition_meta)

    csv_path = out_dir / "loss_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)
    print(f"\n[mlx] wrote {csv_path}")

    with open(out_dir / "run_meta.json", "w") as f:
        json.dump({
            "model": args.model,
            "framework": "mlx",
            "lora": {
                "rank": LORA_RANK, "scale": LORA_SCALE, "layers": args.lora_layers,
                "lm_head_adapted": args.lora_lm_head,
            },
            "hedge_token_mode": args.hedge_tokens,
            "plain_prompt": args.plain_prompt,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "lambda_over": args.lambda_over,
            "lambda_under": args.lambda_under,
            "lambda_hedge": args.lambda_hedge,
            "arm": "sft_only" if max(
                args.lambda_over, args.lambda_under, args.lambda_hedge
            ) == 0 else "tsct",
            "batch_size": args.batch_size,
            "seed": args.seed,
            "n_raw_examples": len(raw_examples),
            "n_per_volatility": args.n_per_volatility,
            "grad_check_every": args.grad_check_every,
            "conditions": condition_meta,
        }, f, indent=2)
    print(f"[mlx] wrote {out_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()
