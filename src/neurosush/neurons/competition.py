"""competition and noise acting on the membrane between integration and firing."""

from __future__ import annotations

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import NeuronGroup
from neurosush.core.order import Order


def kwta_losers(
    v: torch.Tensor,
    threshold: torch.Tensor | float,
    k: int,
    *,
    shape: tuple[int, ...] | None = None,
    dim: int | None = None,
) -> torch.Tensor:
    """Pure k-winners-take-all losers identification."""
    if dim is None:
        if shape is not None:
            raise ValueError("shape must be None if dim is None")
        v_flat = v.reshape(-1)
        candidates = v_flat >= threshold
        masked = v_flat.masked_fill(~candidates, -float("inf"))
        rank = masked.argsort(dim=0, descending=True, stable=True).argsort(dim=0, stable=True)
        return candidates & (rank >= k)
    if shape is None:
        raise ValueError("shape must be provided if dim is not None")
    candidates = v >= threshold
    masked = v.view(shape).masked_fill(~candidates.view(shape), -float("inf"))
    rank = masked.argsort(dim=dim, descending=True, stable=True).argsort(dim=dim, stable=True)
    losers = candidates.view(shape) & (rank >= k)
    return losers.reshape(-1)


class KWTA(Behavior):
    """K-winners-take-all competition."""

    order = Order.COMPETITION

    def __init__(self, k: int, *, dim: int | None = None) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if dim is not None and dim not in (0, 1, 2):
            raise ValueError(f"dim must be None or in (0, 1, 2), got {dim}")
        self.k = k
        self.dim = dim

    def initialize(self, group: NeuronGroup) -> None:
        """Check that the group has a neuron model."""
        if not hasattr(group, "model"):
            raise RuntimeError(f"KWTA on {group.name} needs a neuron model such as LIF")

    def forward(self, group: NeuronGroup) -> None:
        """Apply k-winners-take-all competition."""
        if self.dim is None:
            losers = kwta_losers(group.v, group.threshold, self.k)
        else:
            losers = kwta_losers(group.v, group.threshold, self.k, shape=group.shape, dim=self.dim)
        group.v = torch.where(losers.reshape(group.v.shape), group.v_reset, group.v)


class InherentNoise(Behavior):
    """Inherent noise acting on the membrane."""

    order = Order.NOISE

    def __init__(
        self, *, scale: float = 1.0, offset: float = 0.0, distribution: str = "uniform"
    ) -> None:
        if distribution not in ("uniform", "normal"):
            raise ValueError(f"distribution must be 'uniform' or 'normal', got {distribution}")
        self.scale = scale
        self.offset = offset
        self.distribution = distribution

    def forward(self, group: NeuronGroup) -> None:
        """Emit noise on the membrane."""
        sample = group.rand() if self.distribution == "uniform" else group.randn()
        group.v = group.v + self.scale * sample + self.offset
