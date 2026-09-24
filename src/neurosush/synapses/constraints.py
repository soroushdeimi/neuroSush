"""Constraints on weights and currents: clipping and per-neuron normalization."""

from __future__ import annotations

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import SynapseGroup
from neurosush.core.order import Order


def incoming_weight_sum(syn: SynapseGroup) -> torch.Tensor:
    """Sum of the weights reaching each destination neuron, shape ``(dst.size,)``."""
    w, kind = syn.weights, getattr(syn, "connectivity", None)
    assert w is not None  # Weight normalization requires initialized weights.
    if kind == "dense":
        return w.sum(0)
    if kind == "one_to_one":
        return w.clone()
    if kind == "sparse":
        return torch.zeros(syn.dst.size, dtype=w.dtype, device=w.device).index_add_(
            0, syn.dst_idx, w
        )
    if kind == "conv2d":
        return w.sum(dim=(1, 2, 3)).repeat_interleave(syn.dst.height * syn.dst.width)
    if kind == "local2d":
        return w.sum(-1).flatten()
    raise ValueError(f"{syn.name}: normalization does not support connectivity {kind!r}")


def _scale_weights(syn: SynapseGroup, factor: torch.Tensor) -> torch.Tensor:
    """Multiply the weights reaching destination neuron ``j`` by ``factor[j]``."""
    w, kind = syn.weights, syn.connectivity
    assert w is not None  # Weight normalization requires initialized weights.
    if kind == "dense":
        return w * factor.unsqueeze(0)
    if kind == "one_to_one":
        return w * factor
    if kind == "sparse":
        return w * factor[syn.dst_idx]
    if kind == "conv2d":
        per_channel = factor.view(syn.dst.depth, -1)[:, 0]
        return w * per_channel.view(-1, 1, 1, 1)
    return w * factor.view(w.shape[0], w.shape[1], 1)


def _normalizing_factor(sums: torch.Tensor, norm: float) -> torch.Tensor:
    return torch.where(sums == 0, torch.ones_like(sums), norm / sums)


def _check_input(behavior: Behavior, syn: SynapseGroup) -> None:
    if not hasattr(syn, "connectivity"):
        raise RuntimeError(f"{type(behavior).__name__} on {syn.name} needs an input behavior")


class WeightClip(Behavior):
    """Clamps the weights to ``[w_min, w_max]`` after learning.

    Args:
        w_min: Lower bound (may be negative).
        w_max: Upper bound, above ``w_min``.
    """

    order = Order.WEIGHT_CLIP

    def __init__(self, *, w_min: float = 0.0, w_max: float = 1.0) -> None:
        if w_min >= w_max:
            raise ValueError(f"w_min ({w_min}) must be below w_max ({w_max})")
        self.w_min, self.w_max = w_min, w_max

    def forward(self, syn: SynapseGroup) -> None:
        """Clamp the weights."""
        assert syn.weights is not None  # Weight clipping requires initialized weights.
        syn.weights = syn.weights.clamp(self.w_min, self.w_max)


class WeightNormalization(Behavior):
    """Rescales weights so each destination neuron receives a total weight of ``norm``.

    Neurons with no incoming weight are left unchanged.

    Args:
        norm: Target sum of incoming weights.
    """

    order = Order.WEIGHT_NORMALIZATION

    def __init__(self, *, norm: float = 1.0) -> None:
        self.norm = norm

    def initialize(self, syn: SynapseGroup) -> None:
        """Check that an input behavior set the connectivity."""
        _check_input(self, syn)

    def forward(self, syn: SynapseGroup) -> None:
        """Normalize the weights."""
        syn.weights = _scale_weights(syn, _normalizing_factor(incoming_weight_sum(syn), self.norm))


class CurrentNormalization(Behavior):
    """Scales ``syn.I`` by ``norm / incoming weight sum`` of each destination neuron.

    Args:
        norm: Current of a neuron whose inputs all spike.
    """

    order = Order.CURRENT_NORMALIZATION

    def __init__(self, *, norm: float = 1.0) -> None:
        self.norm = norm

    def initialize(self, syn: SynapseGroup) -> None:
        """Check that an input behavior set the connectivity."""
        _check_input(self, syn)

    def forward(self, syn: SynapseGroup) -> None:
        """Normalize the current."""
        syn.I = syn.I * _normalizing_factor(incoming_weight_sum(syn), self.norm)
