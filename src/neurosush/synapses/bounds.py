"""Weight-dependent gates of learning (how much potentiation and depression a weight allows)."""

from __future__ import annotations

from collections.abc import Callable

import torch

Gates = tuple[torch.Tensor, torch.Tensor]


def soft_bound(w: torch.Tensor, w_min: float, w_max: float) -> Gates:
    """Soft weight-dependent gates.

    Args:
        w: Weights tensor.
        w_min: Minimum weight.
        w_max: Maximum weight.

    Returns:
        A tuple of (ltp_gate, ltd_gate) tensors.
    """
    return (w_max - w).clamp(min=0), (w - w_min).clamp(min=0)


def hard_bound(w: torch.Tensor, w_min: float, w_max: float) -> Gates:
    """Hard weight-dependent gates.

    Args:
        w: Weights tensor.
        w_min: Minimum weight.
        w_max: Maximum weight.

    Returns:
        A tuple of (ltp_gate, ltd_gate) tensors.
    """
    return (w < w_max).to(w.dtype), (w > w_min).to(w.dtype)


def no_bound(w: torch.Tensor, w_min: float, w_max: float) -> Gates:
    """No weight-dependent gates.

    Args:
        w: Weights tensor.
        w_min: Minimum weight.
        w_max: Maximum weight.

    Returns:
        A tuple of (ltp_gate, ltd_gate) tensors.
    """
    return torch.ones_like(w), torch.ones_like(w)


BOUNDS: dict[str, Callable[[torch.Tensor, float, float], Gates]] = {
    "soft": soft_bound,
    "hard": hard_bound,
    "none": no_bound,
}
