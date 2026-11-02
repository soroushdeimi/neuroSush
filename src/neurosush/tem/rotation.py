"""Structured torus transitions for the square-world action interface."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Rotation(nn.Module):
    """Rotate independent 2D planes with tied inverse actions and identity stay.

    This builds in the group structure TEM is supposed to learn. Any grid-like
    periodicity is imposed by the torus, rather than emergent spatial structure.
    Actions follow GraphWorld: right, left, up, down, stay (0 through 4).
    """

    directions: Tensor

    def __init__(self, size: int, generator: torch.Generator) -> None:
        super().__init__()
        if not isinstance(size, int) or isinstance(size, bool) or size < 2 or size % 2:
            raise ValueError(f"size must be a positive even integer, got {size}")
        self.size = size
        self.angles = nn.Parameter(
            torch.empty(2, size // 2, device=generator.device).uniform_(
                -math.pi, math.pi, generator=generator
            )
        )
        self.register_buffer(
            "directions",
            torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]], device=generator.device),
        )

    def forward(self, g: Tensor, action: Tensor) -> Tensor:
        """Apply an orthogonal map without clamping or changing plane radii."""
        if g.ndim != 2 or g.shape[-1] != self.size:
            raise ValueError(f"g must have shape (batch, {self.size}), got {g.shape}")
        if action.shape != g.shape[:1]:
            raise ValueError(f"action shape must be {g.shape[:1]}, got {action.shape}")
        if action.dtype != torch.long:
            raise ValueError(f"action dtype must be torch.int64, got {action.dtype}")
        if bool(((action < 0) | (action > 4)).any()):
            raise ValueError(f"action must be in [0, 4], got {action}")
        angle = self.directions[action].to(g.dtype) @ self.angles
        x, y = g.reshape(-1, self.size // 2, 2).unbind(-1)
        cosine, sine = angle.cos(), angle.sin()
        return torch.stack((cosine * x - sine * y, sine * x + cosine * y), -1).flatten(1)
