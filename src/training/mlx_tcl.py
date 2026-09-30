"""Temporal Calibration Loss (TCL) -- MLX port of `src/training/tcl_loss.py`.

This is a direct, term-for-term translation of the validated PyTorch
implementation so that results at 7B (MLX/LoRA, Apple Silicon) are
comparable to the 0.5B proxy diagnostic (PyTorch/MPS, full fine-tune).
The *semantics* are intended to be identical; only the array library and
the parameter-update path differ.

Every assumption documented in `tcl_loss.py`'s module docstring carries
over unchanged -- in particular, `L_over`/`L_under`/`R_hedge` are inferred
forms, not recovered from the training lead's original repo. See that
docstring for the full rationale; it is not duplicated here so the two
cannot drift.

## Why a port rather than `mlx_lm.lora`

`mlx_lm`'s CLI trains plain cross-entropy. Its Python API
(`mlx_lm.tuner.trainer.train`) does accept a custom `loss` callable, but
the loss it hands you only receives `(model, batch, lengths)` -- there is
no channel for the per-example `hedge_position`, `c_gold`, and
`volatile_mask` that TCL needs, and its batch iterator sorts/packs by
length. Rather than fight both abstractions, this module supplies the loss
and `run_tcl_mlx.py` supplies its own loop, which also keeps the per-step
telemetry at exactly the resolution the debugging notes asked for.

## Gradient-connectivity check under LoRA

In the PyTorch diagnostic the check backprops the isolated c_hat loss into
the LM head weight. Under LoRA the LM head is frozen, so the MLX analog
measures the global norm over the *trainable* (adapter) parameters
instead. The test is unchanged in kind: gradient reaches the adapters
through the frozen-but-differentiable LM head, so `broken` must give
exactly 0 and `fixed` must give nonzero.
"""
from math import log
from typing import Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .hedge_tokens import HEDGE_TOKENS, HEDGE_TO_CONFIDENCE

MAX_HEDGE_ENTROPY = log(len(HEDGE_TOKENS))


def conf_scalar_array() -> mx.array:
    """HEDGE_TO_CONFIDENCE values in HEDGE_TOKENS order."""
    return mx.array([HEDGE_TO_CONFIDENCE[t] for t in HEDGE_TOKENS], dtype=mx.float32)


def compute_c_hat(
    hedge_logits: mx.array, conf_scalars: mx.array, broken: bool = False
) -> Tuple[mx.array, mx.array]:
    """Compute c_hat and the hedge-token probability distribution.

    hedge_logits: (batch, 4) logits restricted to the 4 hedge token ids,
        taken from the position whose next-token prediction is the hedge
        token (standard causal-LM convention).
    broken: reproduce the ORIGINAL BUG (argmax + discrete lookup) for the
        ablation arm. Intentionally non-differentiable.
    """
    hedge_probs = mx.softmax(hedge_logits.astype(mx.float32), axis=-1)

    if broken:
        hedge_id = mx.stop_gradient(mx.argmax(hedge_logits, axis=-1))
        c_hat = conf_scalars[hedge_id]  # discrete lookup -- gradient cut here, on purpose
        return c_hat, hedge_probs

    c_hat = (hedge_probs * conf_scalars).sum(axis=-1)  # fix: differentiable expectation
    return c_hat, hedge_probs


def tcl_terms(
    c_hat: mx.array,
    c_gold: mx.array,
    hedge_probs: mx.array,
    volatile_mask: mx.array,
) -> Dict[str, mx.array]:
    """L_over, L_under, R_hedge for one batch. Mirrors tcl_loss.tcl_terms.

    R_hedge is computed over volatile examples only, matching the torch
    version's `hedge_probs[vmask_bool].mean(0)`. Because MLX has no boolean
    fancy-indexing here, the same mean is expressed as a mask-weighted sum
    divided by the volatile count -- algebraically identical.
    """
    denom = mx.maximum(volatile_mask.sum(), 1.0)

    over = mx.maximum(c_hat - c_gold, 0.0) ** 2
    under = mx.maximum(c_gold - c_hat, 0.0) ** 2
    l_over = (over * volatile_mask).sum() / denom
    l_under = (under * volatile_mask).sum() / denom

    # mean hedge distribution over volatile examples only
    mean_probs = (hedge_probs * volatile_mask[:, None]).sum(axis=0) / denom
    entropy = -(mean_probs * mx.log(mx.maximum(mean_probs, 1e-8))).sum()
    r_hedge = MAX_HEDGE_ENTROPY - entropy

    return {"l_over": l_over, "l_under": l_under, "r_hedge": r_hedge}


def masked_ce(logits: mx.array, targets: mx.array, loss_mask: mx.array) -> mx.array:
    """Token-mean cross-entropy over the assistant span only.

    Matches the torch run's HF `labels=-100` masking: CE is averaged over
    supervised (assistant-turn) tokens, not over all positions.
    """
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    return (ce * loss_mask).sum() / mx.maximum(loss_mask.sum(), 1.0)


def global_grad_norm(grads) -> float:
    """L2 norm over every trainable-parameter gradient in the tree.

    The MLX/LoRA analog of the torch diagnostic's `lm_head.weight.grad.norm()`
    -- see module docstring for why the measured parameter set differs.
    """
    total = mx.zeros(())
    for _, g in tree_flatten(grads):
        if isinstance(g, mx.array):
            total = total + (g.astype(mx.float32) ** 2).sum()
    return float(mx.sqrt(total).item())
