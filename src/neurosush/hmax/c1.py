"""C1 pooling (Serre et al. 2007, Sec. 2): local max over a spatial grid and two adjacent scales."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F


class _BandSpec(Protocol):
    """Duck-typed band spec: anything with ``c1_grid`` and ``c1_stride``."""

    c1_grid: int
    c1_stride: int


def c1_pool(s1: list[torch.Tensor], bands: list[tuple[int, int, int, int]]) -> list[torch.Tensor]:
    """Pool S1 maps into C1 maps: elementwise max of two adjacent scales, then a local max.

    Args:
        s1: One tensor ``(B, T, H, W)`` per S1 scale; all scales share H and W.
        bands: One ``(scale_a, scale_b, grid, stride)`` tuple per band.

    Returns:
        One ``(B, T, floor((H - grid) / stride) + 1, floor((W - grid) / stride) + 1)`` tensor
        per band. Orientations (dim 1) are never mixed.

    Raises:
        ValueError: If s1 or bands is empty, an index is out of range, the two scales have
            different shapes, an input is not 4-D, grid or stride is below 1, or grid is
            larger than H or W.
    """
    if not s1:
        raise ValueError("s1 must not be empty")
    if not bands:
        raise ValueError("bands must not be empty")

    for i, t in enumerate(s1):
        if t.dim() != 4:
            raise ValueError(f"s1[{i}] must be 4-D, got {t.dim()}-D")

    out: list[torch.Tensor] = []
    for j, (a, b, grid, stride) in enumerate(bands):
        if not 0 <= a < len(s1) or not 0 <= b < len(s1):
            raise ValueError(f"band {j}: scale index {a}/{b} out of range for {len(s1)} scales")
        if s1[a].shape != s1[b].shape:
            raise ValueError(
                f"band {j}: scale shapes differ: {tuple(s1[a].shape)} vs {tuple(s1[b].shape)}"
            )
        if grid < 1:
            raise ValueError(f"band {j}: grid must be at least 1, got {grid}")
        if stride < 1:
            raise ValueError(f"band {j}: stride must be at least 1, got {stride}")
        h, w = s1[a].shape[-2:]
        if grid > h or grid > w:
            raise ValueError(f"band {j}: grid {grid} larger than input {h}x{w}")

        m = torch.maximum(s1[a], s1[b])
        out.append(F.max_pool2d(m, kernel_size=grid, stride=stride, padding=0, ceil_mode=False))
    return out


def c1_bands_from_spec(bands: list[_BandSpec]) -> list[tuple[int, int, int, int]]:
    """Build C1 band tuples from band specs carrying ``c1_grid`` and ``c1_stride``.

    Band i (0-based) pools the two adjacent S1 scales (2i, 2i + 1).

    Args:
        bands: Sequence of objects with ``c1_grid`` and ``c1_stride`` attributes.

    Returns:
        One ``(2i, 2i + 1, c1_grid, c1_stride)`` tuple per band.

    Raises:
        ValueError: If bands is empty.
    """
    if not bands:
        raise ValueError("bands must not be empty")
    return [(2 * i, 2 * i + 1, spec.c1_grid, spec.c1_stride) for i, spec in enumerate(bands)]
