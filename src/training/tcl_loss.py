"""Temporal Calibration Loss (TCL) -- reimplemented from the fix documented
in docs/tcl_debugging.md.

The original `src/training/tcl_loss.py` (owned by the training lead) lives in
a separate, unlinked repo and is not available here. This module is a
from-scratch, standalone reimplementation of just enough of TCL to validate
the gradient-path fix on a small proxy model. It is NOT a drop-in
replacement for the original -- treat every design choice below as an
explicit, documented assumption, not a recovered spec.

## The bug (already root-caused, see docs/tcl_debugging.md)

c_hat (the model's confidence estimate fed into the calibration loss) was
computed by argmax-selecting a hedge token and then doing a discrete lookup
into a fixed confidence table. argmax is non-differentiable, so the
calibration terms (L_over, L_under, R_hedge) never received gradient. CE
still trained normally (it doesn't touch c_hat), which is why SFT looked
fine while tcl_total drifted <1% across a full run.

## The fix

c_hat is the softmax-weighted expectation over the 4 hedge-token logits at
the position where the model emits its hedge token:

    hedge_probs = softmax(hedge_logits, dim=-1)
    c_hat = (hedge_probs * conf_scalars).sum(dim=-1)

This keeps every operation differentiable end to end -- there is no argmax,
sampling, or lookup table indexing anywhere in the c_hat path.

## Assumed loss terms (NOT specified anywhere in this repo -- inferred)

The original loss formula is not present in this repo (it lives in the
unlinked training repo). The following are standard, defensible forms for
an asymmetric calibration loss + anti-collapse regularizer, chosen because
they match the term *names* used in docs/tcl_debugging.md (L_over, L_under,
R_hedge) and the failure mode described (collapse to [CONFIDENT] on 100% of
predictions). If the real formulas differ, this reimplementation should be
swapped for the recovered original -- but the differentiability fix itself
(the actual bug) is formula-agnostic and applies regardless.

- **c_gold**: the target confidence, taken from HEDGE_TO_CONFIDENCE[gold_hedge_token]
  for each example (the label already encodes which hedge token -- and
  therefore which confidence -- is correct for that fact's volatility).
- **L_over** (overconfidence penalty): relu(c_hat - c_gold) ** 2, averaged
  over volatile (fast/slow) examples in the batch. Fires only when the
  model claims MORE confidence than the gold label warrants.
- **L_under** (underconfidence penalty): relu(c_gold - c_hat) ** 2, same
  masking. Fires only when the model claims LESS confidence than warranted.
  Kept as a separate term (not folded into L_over via abs()) because the
  debugging doc treats them as distinct loggable quantities, and because a
  real deployment may want asymmetric lambdas (over-claiming confidence is
  usually the worse failure mode).
- **R_hedge** (anti-collapse regularizer): max_entropy - entropy(mean
  hedge_probs over the batch), where mean is taken over volatile examples
  only. This is 0 when the batch's average hedge-token distribution is
  uniform across the 4 tokens (max entropy = log(4)) and grows as the
  batch's predictions collapse onto a single token -- directly penalizing
  the observed 100%-[CONFIDENT] collapse failure mode.

TCL only fires on time-sensitive (fast/slow) examples per the debugging
doc's diagnostic checklist ("TCL only fires on time-sensitive examples";
immutable facts have a single correct answer -- [CONFIDENT] -- and adding
calibration pressure there is redundant with CE). volatile_mask must be
passed in per-batch, and volatile-example count is logged separately so a
batch that's accidentally filtered down to ~0 volatile examples is visible
rather than silently producing near-zero gradient (exactly the false
positive the original debugging process had to rule out).
"""
from dataclasses import dataclass
from math import log

import torch
import torch.nn.functional as F

from .hedge_tokens import HEDGE_TOKENS, HEDGE_TO_CONFIDENCE

MAX_HEDGE_ENTROPY = log(len(HEDGE_TOKENS))


@dataclass
class TCLTermBreakdown:
    l_over: torch.Tensor
    l_under: torch.Tensor
    r_hedge: torch.Tensor
    n_volatile: int

    def calib_loss(self, lambda_over: float, lambda_under: float, lambda_hedge: float) -> torch.Tensor:
        return lambda_over * self.l_over + lambda_under * self.l_under + lambda_hedge * self.r_hedge


def compute_c_hat(hedge_logits: torch.Tensor, conf_scalars: torch.Tensor, broken: bool = False):
    """Compute c_hat and the hedge-token probability distribution.

    hedge_logits: (batch, 4) raw logits restricted to the 4 hedge token ids,
        taken from the position immediately preceding the hedge token
        (standard causal-LM next-token convention).
    conf_scalars: (4,) tensor, HEDGE_TO_CONFIDENCE values in HEDGE_TOKENS order.
    broken: if True, reproduce the ORIGINAL BUG for ablation purposes --
        argmax-select the hedge token, then do a discrete table lookup.
        This path is intentionally non-differentiable and is only wired up
        so the diagnostic run can demonstrate the before/after contrast
        (tcl_total frozen vs. tcl_total moving) in one script.
    """
    hedge_probs = F.softmax(hedge_logits, dim=-1)  # differentiable, always computed for logging

    if broken:
        with torch.no_grad():
            hedge_id = torch.argmax(hedge_logits, dim=-1)  # non-differentiable
        c_hat = conf_scalars[hedge_id]  # discrete lookup -- gradient cut here, on purpose
        return c_hat, hedge_probs

    c_hat = (hedge_probs * conf_scalars).sum(dim=-1)  # fix: differentiable expectation
    return c_hat, hedge_probs


def tcl_terms(
    c_hat: torch.Tensor,
    c_gold: torch.Tensor,
    hedge_probs: torch.Tensor,
    volatile_mask: torch.Tensor,
) -> TCLTermBreakdown:
    """Compute L_over, L_under, R_hedge for one batch. See module docstring
    for the definitions and the documented assumptions behind them.
    """
    n_volatile = int(volatile_mask.sum().item())
    denom = volatile_mask.sum().clamp(min=1)

    over = F.relu(c_hat - c_gold) ** 2
    under = F.relu(c_gold - c_hat) ** 2
    l_over = (over * volatile_mask).sum() / denom
    l_under = (under * volatile_mask).sum() / denom

    if n_volatile > 0:
        vmask_bool = volatile_mask.bool()
        mean_probs = hedge_probs[vmask_bool].mean(dim=0)
        entropy = -(mean_probs * mean_probs.clamp(min=1e-8).log()).sum()
        r_hedge = MAX_HEDGE_ENTROPY - entropy
    else:
        r_hedge = torch.zeros((), device=c_hat.device, dtype=c_hat.dtype)

    return TCLTermBreakdown(l_over=l_over, l_under=l_under, r_hedge=r_hedge, n_volatile=n_volatile)


def conf_scalar_tensor(device, dtype) -> torch.Tensor:
    return torch.tensor([HEDGE_TO_CONFIDENCE[t] for t in HEDGE_TOKENS], device=device, dtype=dtype)
