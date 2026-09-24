"""Sensorimotor object recognition by many columns that vote.

Follows Hawkins, Ahmad and Cui (2017), "A theory of how columns in the neocortex enable
learning the structure of the world", Front. Neural Circuits 11:81, and Lewis et al.
(2019), "Locations in the neocortex: a theory of sensorimotor object recognition using
cortical grid cells", Front. Neural Circuits 13:22.

An object is a feature at every location of a ``height x width`` grid; locations wrap
around like the phases of a grid cell module. A column touches one location at a time.
It keeps every hypothesis ``(object, location)`` consistent with all features it has
sensed, and moves every hypothesis with the sensor (path integration). Each column
proposes the objects that still have a hypothesis; the objects supported by enough
columns form the consensus, and every column drops hypotheses outside it.
"""

from __future__ import annotations

import torch


class ObjectLibrary:
    """Learned objects: ``features[object, y, x]`` is the feature id at a location.

    Args:
        features: Integer tensor of shape ``(objects, height, width)`` with ids ``>= 0``.
    """

    def __init__(self, features: torch.Tensor) -> None:
        if features.dim() != 3 or features.dtype.is_floating_point:
            raise ValueError("features must be an integer tensor (objects, height, width)")
        if (features < 0).any():
            raise ValueError("feature ids must be non-negative")
        self.features = features

    @classmethod
    def random(
        cls,
        objects: int,
        shape: tuple[int, int],
        features: int,
        *,
        generator: torch.Generator | None = None,
    ) -> ObjectLibrary:
        """Objects whose features are drawn uniformly from ``features`` ids."""
        return cls(torch.randint(features, (objects, *shape), generator=generator))

    @property
    def objects(self) -> int:
        """Number of objects."""
        return self.features.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """Grid of locations of every object."""
        return self.features.shape[1], self.features.shape[2]


class SensorColumn:
    """One column's hypotheses about which object it touches and where.

    ``hypotheses[object, y, x]`` is true while "the sensor is at ``(y, x)`` on ``object``"
    agrees with everything sensed so far.
    """

    def __init__(self, library: ObjectLibrary) -> None:
        self.library = library
        self.reset()

    def reset(self) -> None:
        """Forget all sensations: every hypothesis is possible."""
        self.hypotheses = torch.ones_like(self.library.features, dtype=torch.bool)

    def sense(self, feature: int) -> None:
        """Keep the hypotheses whose location holds ``feature``."""
        self.hypotheses &= self.library.features == feature

    def move(self, displacement: tuple[int, int]) -> None:
        """Path integration: shift every hypothesis by the sensor's displacement."""
        self.hypotheses = torch.roll(self.hypotheses, shifts=displacement, dims=(1, 2))

    def candidates(self) -> torch.Tensor:
        """Bool ``(objects,)``: objects that still have a hypothesis."""
        return self.hypotheses.any((1, 2))

    def restrict(self, objects: torch.Tensor) -> None:
        """Drop the hypotheses on objects outside ``objects`` (bool ``(objects,)``)."""
        self.hypotheses &= objects[:, None, None]


def vote(candidates: torch.Tensor, min_support: int | None = None) -> torch.Tensor:
    """Objects proposed by at least ``min_support`` columns (default: all of them).

    Args:
        candidates: Bool ``(columns, objects)``, one row per column.
        min_support: Columns that must agree; fewer than all tolerates faulty columns.
    """
    columns = candidates.shape[0]
    needed = columns if min_support is None else min_support
    if not 1 <= needed <= columns:
        raise ValueError(f"min_support must be in [1, {columns}], got {min_support}")
    return candidates.sum(0) >= needed


class ColumnEnsemble:
    """Columns that sense in parallel and vote after every sensation.

    Args:
        library: Objects every column has learned.
        columns: Number of columns (sensors).
        min_support: Votes an object needs to stay in the consensus.
    """

    def __init__(
        self, library: ObjectLibrary, columns: int, *, min_support: int | None = None
    ) -> None:
        if columns < 1:
            raise ValueError(f"columns must be positive, got {columns}")
        self.columns = [SensorColumn(library) for _ in range(columns)]
        self.min_support = min_support
        self.consensus = torch.ones(library.objects, dtype=torch.bool)

    def reset(self) -> None:
        """Start a new object."""
        for column in self.columns:
            column.reset()
        self.consensus = torch.ones_like(self.consensus)

    def sense(self, features: list[int]) -> torch.Tensor:
        """Every column senses its feature, then all vote; returns the consensus."""
        if len(features) != len(self.columns):
            raise ValueError(f"expected {len(self.columns)} features, got {len(features)}")
        for column, feature in zip(self.columns, features, strict=True):
            column.sense(feature)
        candidates = torch.stack([column.candidates() for column in self.columns])
        self.consensus = vote(candidates, self.min_support)
        for column in self.columns:
            column.restrict(self.consensus)
        return self.consensus

    def move(self, displacements: list[tuple[int, int]]) -> None:
        """Move every column's sensor by its own displacement."""
        for column, displacement in zip(self.columns, displacements, strict=True):
            column.move(displacement)

    def recognized(self) -> int | None:
        """The object id once the consensus holds exactly one object."""
        remaining = self.consensus.nonzero().flatten()
        return int(remaining[0]) if len(remaining) == 1 else None
