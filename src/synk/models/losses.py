"""Loss functions with explicit class-imbalance handling.

``bce_weighted``
    Binary cross-entropy with a data-derived ``pos_weight`` (n_negative /
    n_positive of the TRAINING split).  The weight is computed from data, not
    guessed - see ``compute_pos_weight``.

``focal``
    Focal loss (Lin et al., 2017) down-weighting easy negatives; selected via
    ``loss.type: focal`` with a documented gamma.

``bce``
    Plain unweighted BCE (reference baseline).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_pos_weight(y: torch.Tensor | "np.ndarray") -> float:
    """pos_weight = n_negative / n_positive (clipped to a sane range)."""
    import numpy as np

    arr = np.asarray(y, dtype=float)
    n_pos = float((arr == 1).sum())
    n_neg = float((arr == 0).sum())
    if n_pos == 0:
        return 1.0
    return float(min(max(n_neg / n_pos, 0.2), 20.0))


def weighted_bce_with_logits(logits: torch.Tensor, y: torch.Tensor, pos_weight: float) -> torch.Tensor:
    weight = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight)


def focal_loss_with_logits(logits: torch.Tensor, y: torch.Tensor, gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    """Binary focal loss. ``alpha`` weights the positive class."""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * y + (1 - p) * (1 - y)
    alpha_t = alpha * y + (1 - alpha) * (1 - y)
    loss = alpha_t * (1 - p_t).pow(gamma) * bce
    return loss.mean()


def build_loss(config, y_train: "np.ndarray"):
    """Return a callable (logits, y) -> scalar loss plus descriptive metadata."""
    kind = config.loss.type

    if kind == "bce_weighted":
        pos_weight = compute_pos_weight(y_train)

        def loss_fn(logits, y):
            return weighted_bce_with_logits(logits, y, pos_weight)

        return loss_fn, {"type": kind, "pos_weight": round(pos_weight, 4)}
    if kind == "focal":
        gamma = config.loss.focal_gamma

        def loss_fn(logits, y):
            return focal_loss_with_logits(logits, y, gamma=gamma)

        return loss_fn, {"type": kind, "focal_gamma": gamma}
    if kind == "bce":

        def loss_fn(logits, y):
            return F.binary_cross_entropy_with_logits(logits, y)

        return loss_fn, {"type": kind}
    raise ValueError(f"Unknown loss type: {kind}")
