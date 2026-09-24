"""Grid cell modules: a location code built from periodic phases on a torus.

A module has a hexagonal lattice with basis ``A = scale * [e1, e2]`` where ``e1`` points at
``orientation`` and ``e2`` 60 degrees further. Its state is a phase ``phi`` in ``[0, 1)^2``;
a position ``x`` has phase ``A^-1 x mod 1`` and a displacement ``d`` moves it to
``phi + A^-1 d mod 1`` (path integration). Each cell prefers one phase and fires as a
Gaussian bump of the world-space distance to it, taken over all lattice translates, so its
firing field is a hexagonal grid. Modules with different scales and orientations together
give locations a code that is unique over ranges far larger than any single scale.

See Hawkins et al. (2019), "A framework for intelligence and cortical function based on
grid cells in the neocortex", Front. Neural Circuits 12:121, and Lewis et al. (2019),
"Locations in the neocortex", Front. Neural Circuits 13:22.
"""

from __future__ import annotations

import math

import torch


def hexagonal_rate(
    position: torch.Tensor,
    scale: float,
    orientation: float = 0.0,
    offset: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Idealized grid firing rate in ``[0, 1]``: three plane waves 60 degrees apart.

    ``(1 + 2 / 3 * sum_i cos(k_i . (x - offset))) / 3`` with ``|k_i| = 4 pi / (sqrt(3) scale)``
    peaks (value 1) on a hexagonal lattice with spacing ``scale`` (Solstad et al. 2006).
    """
    k = 4 * math.pi / (math.sqrt(3) * scale)
    angles = torch.tensor(
        [orientation + math.pi / 6 + i * math.pi / 3 for i in range(3)], dtype=position.dtype
    )
    waves = k * torch.stack([angles.cos(), angles.sin()], -1)  # (3, 2)
    shifted = position - torch.tensor(offset, dtype=position.dtype)
    return (1 + 2 / 3 * torch.cos(shifted @ waves.T).sum(-1)) / 3


class GridCellModule:
    """One grid module: phase arithmetic and the activity of its cells.

    Args:
        scale: Distance between neighboring firing fields.
        orientation: Angle of the first lattice vector, in radians.
        cells_per_axis: Preferred phases form a ``cells_per_axis x cells_per_axis`` grid.
        bump_width: Standard deviation of a cell's bump, as a fraction of ``scale``.
    """

    def __init__(
        self,
        scale: float,
        orientation: float = 0.0,
        *,
        cells_per_axis: int = 10,
        bump_width: float = 0.1,
    ) -> None:
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if cells_per_axis < 1:
            raise ValueError(f"cells_per_axis must be positive, got {cells_per_axis}")
        if bump_width <= 0:
            raise ValueError(f"bump_width must be positive, got {bump_width}")
        self.scale, self.orientation = scale, orientation
        self.bump_width = bump_width
        e1 = (math.cos(orientation), math.sin(orientation))
        e2 = (math.cos(orientation + math.pi / 3), math.sin(orientation + math.pi / 3))
        self.basis = scale * torch.tensor([[e1[0], e2[0]], [e1[1], e2[1]]], dtype=torch.float64)
        self.inverse = torch.linalg.inv(self.basis)
        ticks = (torch.arange(cells_per_axis, dtype=torch.float64) + 0.5) / cells_per_axis
        self.preferred = torch.cartesian_prod(ticks, ticks)  # (cells, 2)

    @property
    def cells(self) -> int:
        """Number of cells in the module."""
        return len(self.preferred)

    def phase(self, position: torch.Tensor) -> torch.Tensor:
        """Phase ``A^-1 x mod 1`` of positions of shape ``(..., 2)``."""
        return torch.remainder(position.to(torch.float64) @ self.inverse.T, 1.0)

    def move(self, phase: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        """Path integration: the phase after moving by ``displacement``."""
        return torch.remainder(phase + displacement.to(torch.float64) @ self.inverse.T, 1.0)

    def activity(self, phase: torch.Tensor) -> torch.Tensor:
        """Cell activity ``exp(-d^2 / (2 (bump_width * scale)^2))`` of shape ``(..., cells)``.

        ``d`` is the world-space distance between the phase and a cell's preferred phase,
        minimized over the lattice translates (the nearest of the 9 neighboring wraps).
        """
        delta = phase.unsqueeze(-2) - self.preferred  # (..., cells, 2)
        delta = delta - torch.round(delta)  # to [-0.5, 0.5]
        shifts = torch.cartesian_prod(*(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64),) * 2)
        world = (delta.unsqueeze(-2) + shifts) @ self.basis.T  # (..., cells, 9, 2)
        distance_sq = (world**2).sum(-1).min(-1).values
        sigma = self.bump_width * self.scale
        return torch.exp(-distance_sq / (2 * sigma**2))


class GridCellModules:
    """Several modules that together encode positions.

    Args:
        scales: Scale of every module.
        orientations: Orientation of every module (radians); defaults to 0 for all.
        cells_per_axis: Cells per phase axis in every module.
        bump_width: Bump width of every module, as a fraction of its scale.
    """

    def __init__(
        self,
        scales: list[float],
        orientations: list[float] | None = None,
        *,
        cells_per_axis: int = 10,
        bump_width: float = 0.1,
    ) -> None:
        orientations = orientations if orientations is not None else [0.0] * len(scales)
        if len(orientations) != len(scales) or not scales:
            raise ValueError("give one orientation per scale and at least one module")
        self.modules = [
            GridCellModule(s, o, cells_per_axis=cells_per_axis, bump_width=bump_width)
            for s, o in zip(scales, orientations, strict=True)
        ]

    def phases(self, position: torch.Tensor) -> torch.Tensor:
        """Phases of every module, shape ``(..., modules, 2)``."""
        return torch.stack([m.phase(position) for m in self.modules], -2)

    def move(self, phases: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        """Path-integrate every module's phase by the same world displacement."""
        return torch.stack(
            [m.move(phases[..., i, :], displacement) for i, m in enumerate(self.modules)], -2
        )

    def encode(self, phases: torch.Tensor) -> torch.Tensor:
        """Concatenated cell activity of all modules."""
        return torch.cat([m.activity(phases[..., i, :]) for i, m in enumerate(self.modules)], -1)

    def decode(self, phases: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """The candidate position whose phases are closest (on every torus) to ``phases``.

        Args:
            phases: Phases ``(modules, 2)`` to decode.
            candidates: Positions ``(n, 2)`` to choose from.
        """
        delta = self.phases(candidates) - phases
        delta = delta - torch.round(delta)
        return candidates[(delta**2).sum((-1, -2)).argmin()]
