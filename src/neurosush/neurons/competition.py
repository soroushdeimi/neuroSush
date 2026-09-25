"""Competition and noise acting on the membrane between integration and firing."""

from __future__ import annotations

import math

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
    lead = v.shape[:-1]  # batch dimensions: every sample competes on its own
    if dim is None:
        view_v, view_c, axis = v, candidates, -1
    elif shape is None:
        raise ValueError(f"shape is required when dim is given (dim={dim})")
    else:
        view_v, view_c = v.view(*lead, *shape), candidates.view(*lead, *shape)
        axis = len(lead) + dim
    masked = view_v.masked_fill(~view_c, -float("inf"))
    # the first k of a stable descending order win (ties keep the lower index)
    order = masked.argsort(dim=axis, descending=True, stable=True)
    winners = torch.zeros_like(view_c).scatter_(
        axis, order.narrow(axis, 0, min(k, order.shape[axis])), True
    )
    return (view_c & ~winners).reshape(v.shape)


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


def minicolumn_inhibition(
    v: torch.Tensor,
    threshold: torch.Tensor | float,
    v_reset: float,
    inhibition: torch.Tensor,
    *,
    cells_per_column: int,
    duration: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of fast inhibition inside minicolumns of ``cells_per_column`` cells.

    Cells of an inhibited minicolumn are held at ``v_reset``. In every other minicolumn the
    cells that reach ``threshold`` this step are left to fire together, and the minicolumn
    becomes inhibited for the next ``duration`` steps: cells that would cross later stay
    silent.

    Args:
        v: Membrane voltages ``(..., columns * cells_per_column)``.
        threshold: Spike threshold, scalar or per cell.
        v_reset: Voltage of inhibited cells.
        inhibition: Steps of inhibition left per minicolumn ``(..., columns)``.
        cells_per_column: Cells in every minicolumn (consecutive cells).
        duration: Steps of inhibition after a minicolumn fires.

    Returns:
        The new voltages and the new inhibition counters.
    """
    lead, columns = v.shape[:-1], inhibition.shape[-1]
    blocked = (inhibition > 0).unsqueeze(-1).expand(*lead, columns, cells_per_column)
    v = v.masked_fill(blocked.reshape(v.shape), v_reset)
    fires = (v >= threshold).view(*lead, columns, cells_per_column).any(-1)
    inhibition = torch.where(fires, duration, (inhibition - 1).clamp(min=0))
    return v, inhibition


class MinicolumnInhibition(Behavior):
    """The first cells of a minicolumn to reach threshold silence the rest of it.

    Temporal memory in spiking form: a cell depolarized by a dendritic plateau (predicted)
    reaches threshold before its neighbors and fires alone; when no cell of an active
    minicolumn is predicted, all of them reach threshold in the same step and fire together
    (a burst). Consecutive groups of ``cells_per_column`` cells form the minicolumns.

    Args:
        cells_per_column: Cells in every minicolumn.
        duration: How long the inhibition lasts after a minicolumn fires, in the unit of
            ``dt``; it lasts ``ceil(duration / dt)`` steps.
    """

    order = Order.COMPETITION

    def __init__(self, *, cells_per_column: int, duration: float) -> None:
        if cells_per_column < 1:
            raise ValueError(f"cells_per_column must be positive, got {cells_per_column}")
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")
        self.cells_per_column, self.duration = cells_per_column, duration

    def initialize(self, group: NeuronGroup) -> None:
        """Check the group and allocate the inhibition counters."""
        if not hasattr(group, "model"):
            raise RuntimeError(
                f"MinicolumnInhibition on {group.name} needs a neuron model such as LIF"
            )
        if group.size % self.cells_per_column:
            raise ValueError(
                f"{group.name} has {group.size} cells, not a multiple of "
                f"cells_per_column={self.cells_per_column}"
            )
        net = group.net
        columns = group.size // self.cells_per_column
        batch = () if net.batch_size is None else (net.batch_size,)
        group.column_inhibition = torch.zeros(
            (*batch, columns), dtype=torch.long, device=net.device
        )
        self.steps = max(1, math.ceil(self.duration / net.dt - 1e-9))

    def forward(self, group: NeuronGroup) -> None:
        """Hold inhibited minicolumns down and start inhibition where cells cross."""
        group.v, group.column_inhibition = minicolumn_inhibition(
            group.v,
            group.threshold,
            group.v_reset,
            group.column_inhibition,
            cells_per_column=self.cells_per_column,
            duration=self.steps,
        )


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
