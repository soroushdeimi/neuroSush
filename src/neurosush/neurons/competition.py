"""Competition and noise acting on the membrane between integration and firing."""

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
    """Neurons above threshold that are not among the ``k`` highest voltages.

    Args:
        v: Membrane voltages, one per neuron.
        threshold: Spike threshold, scalar or per neuron.
        k: Number of winners.
        shape: ``(depth, height, width)`` of the group; required when ``dim`` is given.
        dim: Compete separately along this axis of ``shape`` (``0``: across feature maps at
            every position); ``None`` runs one competition over all neurons.

    Returns:
        Flat bool mask of the neurons that must not spike. Ties keep the lower index.
    """
    candidates = v >= threshold
    if dim is None:
        view_v, view_c, axis = v, candidates, 0
    elif shape is None:
        raise ValueError(f"shape is required when dim is given (dim={dim})")
    else:
        view_v, view_c, axis = v.view(shape), candidates.view(shape), dim
    masked = view_v.masked_fill(~view_c, -float("inf"))
    rank = masked.argsort(dim=axis, descending=True, stable=True).argsort(dim=axis, stable=True)
    return (view_c & (rank >= k)).reshape(-1)


class KWTA(Behavior):
    """k-winners-take-all: only the ``k`` highest voltages above threshold may spike.

    Losers are set to ``v_reset`` before :class:`~neurosush.neurons.models.Fire` runs.

    Args:
        k: Number of winners.
        dim: Axis of the group shape ``(depth, height, width)`` to compete along, or ``None``
            for one competition over the group.
    """

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
        """Reset the losers of the competition."""
        losers = kwta_losers(group.v, group.threshold, self.k, shape=group.shape, dim=self.dim)
        group.v = group.v.masked_fill(losers, group.v_reset)


class InherentNoise(Behavior):
    """Adds ``scale * sample + offset`` to the membrane every step.

    Samples come from the network generator: uniform in ``[0, 1)`` or standard normal.

    Args:
        scale: Multiplier of the random sample.
        offset: Constant added every step.
        distribution: ``"uniform"`` or ``"normal"``.
    """

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
        """Perturb the membrane."""
        sample = group.rand() if self.distribution == "uniform" else group.randn()
        group.v = group.v + self.scale * sample + self.offset
