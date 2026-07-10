"""Training losses for the pocket-tts CALM model.

Implements the objectives of "Continuous Audio Language Models" (arXiv:2509.06926),
Appendix A (Lagrangian Self-Distillation, following Boffi et al. 2025):

- flow-matching loss (Eq. 5), trained on the diagonal F(c, t, t, x_t)
- LSD self-distillation loss (Eq. 6), enforcing consistency of the two-time flow
  map through a forward-mode derivative (jvp)
- an EOS binary classification loss for the `out_eos` head

Time convention follows `pocket_tts.models.flow_lm.lsd_decode`: time 0 is noise,
time 1 is data, and the flow map is f(x_a, a -> b) = x_a + (b - a) * F(c, a, b, x_a).
The linear interpolation path is x_t = (1 - t) * eps + t * x_1 with velocity
target x_1 - eps.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveWeight(nn.Module):
    """Learned adaptive weighting w_psi(s, t) from Eq. (5)-(6) (cf. Lu & Song 2025).

    The loss is weighted as exp(-w) * ||err||^2 + w. The final layer is
    zero-initialized so training starts as a plain MSE (w = 0).
    """

    def __init__(self, hidden: int = 128, n_freqs: int = 16):
        super().__init__()
        self.register_buffer("freqs", 2.0 * math.pi * torch.arange(1, n_freqs + 1).float())
        self.mlp = nn.Sequential(
            nn.Linear(4 * n_freqs, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """s, t: [M, 1] in [0, 1]. Returns w of shape [M, 1]."""
        feats = torch.cat(
            [
                torch.sin(s * self.freqs),
                torch.cos(s * self.freqs),
                torch.sin(t * self.freqs),
                torch.cos(t * self.freqs),
            ],
            dim=-1,
        )
        return self.mlp(feats)


def _weighted(per_sample: torch.Tensor, w: torch.Tensor | None) -> torch.Tensor:
    """Apply exp(-w) * err + w weighting (Eq. 5-6) and reduce to a scalar.

    w is clamped to [-4, 4]: near convergence w tracks ln(err), and exp(-w)
    would otherwise amplify the gradient of a late hard batch by ~1/err,
    destabilizing the end of training.
    """
    if w is None:
        return per_sample.mean()
    w = w.clamp(-4.0, 4.0)
    return (torch.exp(-w) * per_sample + w).mean()


def flow_matching_loss(
    flow_net: nn.Module,
    cond: torch.Tensor,
    x1: torch.Tensor,
    weight_fn: AdaptiveWeight | None = None,
    generator: torch.Generator | None = None,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Flow-matching loss (Eq. 5) on the diagonal s == t.

    Args:
        flow_net: SimpleMLPAdaLN with signature (cond, s, t, x) -> flow.
        cond: [M, D] backbone conditioning vectors.
        x1: [M, ldim] normalized target latents (data, time 1).
        return_aux: also return {"mse": raw unweighted MSE, "w": mean learned
            weight} (detached, for logging) — the weighted loss alone cannot
            distinguish error reduction from w drift.
    """
    m = x1.shape[0]
    t = torch.rand(m, 1, device=x1.device, dtype=x1.dtype, generator=generator)
    eps = torch.empty_like(x1).normal_(generator=generator)
    x_t = (1.0 - t) * eps + t * x1
    v_target = x1 - eps
    pred = flow_net(cond, t, t, x_t)
    per_sample = (pred - v_target).pow(2).mean(dim=-1, keepdim=True)
    w = weight_fn(t, t) if weight_fn is not None else None
    loss = _weighted(per_sample, w)
    if return_aux:
        aux = {
            "mse": per_sample.detach().mean(),
            "w": w.detach().mean() if w is not None else torch.zeros((), device=x1.device),
        }
        return loss, aux
    return loss


def lsd_loss(
    flow_net: nn.Module,
    cond: torch.Tensor,
    x1: torch.Tensor,
    weight_fn: AdaptiveWeight | None = None,
    generator: torch.Generator | None = None,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """Lagrangian Self-Distillation loss (Eq. 6).

    Enforces d/db f(x_a, a -> b) == F(c, b, b, f(x_a, a -> b)) with a
    stop-gradient teacher on the right-hand side. The derivative is computed
    with forward-mode AD (the custom LayerNorm in `pocket_tts.modules.mlp`
    exists precisely to support jvp).
    """
    m = x1.shape[0]
    a = torch.rand(m, 1, device=x1.device, dtype=x1.dtype, generator=generator)
    u = torch.rand(m, 1, device=x1.device, dtype=x1.dtype, generator=generator)
    b = a + (1.0 - a) * u
    eps = torch.empty_like(x1).normal_(generator=generator)
    x_a = (1.0 - a) * eps + a * x1

    def flow_map(b_: torch.Tensor) -> torch.Tensor:
        return x_a + (b_ - a) * flow_net(cond, a, b_, x_a)

    f_b, df_db = torch.func.jvp(flow_map, (b,), (torch.ones_like(b),))
    with torch.no_grad():
        v_at_b = flow_net(cond, b, b, f_b)
    per_sample = (df_db - v_at_b).pow(2).mean(dim=-1, keepdim=True)
    w = weight_fn(a, b) if weight_fn is not None else None
    loss = _weighted(per_sample, w)
    if return_aux:
        aux = {
            "mse": per_sample.detach().mean(),
            "w": w.detach().mean() if w is not None else torch.zeros((), device=x1.device),
        }
        return loss, aux
    return loss


def eos_loss(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, pos_weight: float = 30.0
) -> torch.Tensor:
    """Masked BCE for the EOS head.

    Args:
        logits: [B, S] raw outputs of `flow_lm.out_eos`.
        labels: [B, S] float, 1.0 on EOS frames.
        mask: [B, S] bool, True on valid (non-padding) frames.
    """
    per_pos = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=torch.tensor(pos_weight, device=logits.device),
        reduction="none",
    )
    return (per_pos * mask).sum() / mask.sum().clamp(min=1)
