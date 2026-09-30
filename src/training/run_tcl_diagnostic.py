"""Diagnostic TCL training run on a small proxy model.

Validates the differentiable-c_hat fix (docs/tcl_debugging.md) before anyone
commits to a full 8B run. Per the debugging doc's own diagnostic checklist:
run 50-200 steps, log L_over / L_under / R_hedge / CE / tcl_total
SEPARATELY every step (not just the summed total -- that's exactly what let
the original bug hide), and check the predicted hedge-token distribution
for collapse.

Runs two conditions back to back for direct contrast:
  - "broken": reproduces the original bug (argmax + lookup -> gradient cut)
  - "fixed":  the documented fix (softmax expectation, differentiable)

Usage:
    python -m src.training.run_tcl_diagnostic --model Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import TCLDataset, collate, load_slice
from .hedge_tokens import HEDGE_TOKENS, add_hedge_tokens
from .tcl_loss import compute_c_hat, conf_scalar_tensor, tcl_terms

# Revised hyperparameters from docs/tcl_debugging.md (post-debugging table).
EPOCHS = 3
LEARNING_RATE = 5e-5
LAMBDA_OVER = 0.5
LAMBDA_UNDER = 0.5
LAMBDA_HEDGE = 0.3

BATCH_SIZE = 4
N_PER_VOLATILITY = 48  # -> ~104 examples (fast/slow capped at 48, immutable ~8 available)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_condition(mode, model, tokenizer, hedge_token_ids, loader, device, conf_scalars, log_rows, epochs=EPOCHS):
    """mode: "broken" or "fixed". Trains for `epochs` passes over loader,
    appending one row per step to log_rows with every TCL term separated.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    step = 0
    t0 = time.time()

    for epoch in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            hedge_positions = batch["hedge_positions"].to(device)
            c_gold = batch["c_gold"].to(device)
            volatile_mask = batch["volatile_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ce_loss = out.loss

            batch_idx = torch.arange(input_ids.size(0), device=device)
            full_logits_at_hedge = out.logits[batch_idx, hedge_positions, :]  # (B, vocab)
            hedge_logits = full_logits_at_hedge[:, hedge_token_ids]  # (B, 4)

            c_hat, hedge_probs = compute_c_hat(hedge_logits, conf_scalars, broken=(mode == "broken"))
            terms = tcl_terms(c_hat, c_gold, hedge_probs, volatile_mask)
            calib_loss = terms.calib_loss(LAMBDA_OVER, LAMBDA_UNDER, LAMBDA_HEDGE)
            tcl_total = ce_loss + calib_loss

            # Isolated gradient-connectivity check. R_hedge is built from
            # hedge_probs directly and never touches c_hat, so it stays
            # differentiable in BOTH modes -- it is not a useful signal for
            # the bug. The bug is specifically about c_hat, so isolate
            # (L_over + L_under) -- which are pure functions of c_hat --
            # and backprop that alone into the LM head weight. In "broken"
            # mode this must be ~0 (argmax detaches c_hat from the graph);
            # in "fixed" mode it must be > 0.
            lm_head = model.get_output_embeddings()
            c_hat_loss = LAMBDA_OVER * terms.l_over + LAMBDA_UNDER * terms.l_under
            model.zero_grad(set_to_none=True)
            if c_hat_loss.requires_grad:
                c_hat_loss.backward(retain_graph=True)
            c_hat_grad_norm = (
                lm_head.weight.grad.norm().item()
                if lm_head.weight.grad is not None else 0.0
            )

            optimizer.zero_grad()
            tcl_total.backward()
            optimizer.step()

            with torch.no_grad():
                predicted_hedge_ids = hedge_probs.argmax(dim=-1)
                pred_confident_frac = (predicted_hedge_ids == 0).float().mean().item()  # index 0 == [CONFIDENT]

            log_rows.append({
                "mode": mode,
                "epoch": epoch,
                "step": step,
                "ce": ce_loss.item(),
                "l_over": terms.l_over.item(),
                "l_under": terms.l_under.item(),
                "r_hedge": terms.r_hedge.item(),
                "tcl_total": tcl_total.item(),
                "n_volatile": terms.n_volatile,
                "pred_confident_frac": pred_confident_frac,
                "c_hat_grad_norm": c_hat_grad_norm,
            })
            step += 1

    return step, time.time() - t0


def eval_hedge_distribution(model, tokenizer, hedge_token_ids, loader, device):
    """Full pass in eval mode: what fraction of predictions land on each
    hedge token? Used to check for collapse to 100% [CONFIDENT].
    """
    model.eval()
    counts = {t: 0 for t in HEDGE_TOKENS}
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hedge_positions = batch["hedge_positions"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_idx = torch.arange(input_ids.size(0), device=device)
            hedge_logits = out.logits[batch_idx, hedge_positions, :][:, hedge_token_ids]
            pred_ids = hedge_logits.argmax(dim=-1).tolist()
            for pid in pred_ids:
                counts[HEDGE_TOKENS[pid]] += 1
                total += 1
    model.train()
    return {k: v / total for k, v in counts.items()}, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out-dir", default="data/prep/tcl_diagnostic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--n-per-volatility", type=int, default=N_PER_VOLATILITY)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"[run] device={device} model={args.model}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_slice(n_per_volatility=args.n_per_volatility, seed=args.seed)
    print(f"[run] loaded {len(examples)} examples")

    log_rows = []
    dist_results = {}
    condition_meta = {}

    for mode in ("broken", "fixed"):
        print(f"\n[run] === condition: {mode} ===")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        hedge_token_ids = add_hedge_tokens(tokenizer, model)
        model.to(device)
        model.train()

        conf_scalars = conf_scalar_tensor(device, torch.float32)

        dataset = TCLDataset(examples, tokenizer)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        )
        eval_loader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        )

        pre_dist, pre_total = eval_hedge_distribution(model, tokenizer, hedge_token_ids, eval_loader, device)
        print(f"[run] {mode} pre-train hedge dist (n={pre_total}): {pre_dist}")

        n_steps, wall_s = run_condition(mode, model, tokenizer, hedge_token_ids, loader, device, conf_scalars, log_rows, epochs=args.epochs)

        post_dist, post_total = eval_hedge_distribution(model, tokenizer, hedge_token_ids, eval_loader, device)
        print(f"[run] {mode} post-train hedge dist (n={post_total}): {post_dist}")
        print(f"[run] {mode} finished: {n_steps} steps in {wall_s:.1f}s")

        condition_meta[mode] = {
            "n_steps": n_steps,
            "wall_seconds": wall_s,
            "pre_train_hedge_dist": pre_dist,
            "post_train_hedge_dist": post_dist,
        }

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    csv_path = out_dir / "loss_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n[run] wrote {csv_path}")

    meta_path = out_dir / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_examples": len(examples),
            "epochs": args.epochs,
            "learning_rate": LEARNING_RATE,
            "lambda_over": LAMBDA_OVER,
            "lambda_under": LAMBDA_UNDER,
            "lambda_hedge": LAMBDA_HEDGE,
            "batch_size": BATCH_SIZE,
            "conditions": condition_meta,
        }, f, indent=2)
    print(f"[run] wrote {meta_path}")


if __name__ == "__main__":
    main()
