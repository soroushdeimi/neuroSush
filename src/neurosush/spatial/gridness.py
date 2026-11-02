"""Spatial autocorrelogram and gridness metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F


def spatial_autocorrelogram(rate: torch.Tensor) -> torch.Tensor:
    """Full nonperiodic, overlap-wise Pearson autocorrelation, masking NaNs.

    Output is (2H-1, 2W-1); undefined/constant overlaps and fewer than three
    overlapping occupied bins return NaN. FFT computes the six sufficient sums.
    """
    r = rate.detach().double().cpu()
    if r.ndim != 2 or min(r.shape) < 2 or torch.isinf(r).any():
        raise ValueError("rate must be a 2D map at least 2x2, with finite values or NaNs")
    mask = torch.isfinite(r).double()
    r = torch.nan_to_num(r)
    h, w = r.shape
    shape = (2 * h - 1, 2 * w - 1)

    mask_spectrum = torch.fft.rfft2(mask, s=shape)
    rate_spectrum = torch.fft.rfft2(r, s=shape)
    square_spectrum = torch.fft.rfft2(r.square(), s=shape)

    def correlate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        result = torch.fft.irfft2(a * b.conj(), s=shape)
        return torch.roll(result, (h - 1, w - 1), (0, 1))

    count = correlate(mask_spectrum, mask_spectrum).round().clamp_min(0)
    n = count.clamp_min(1)
    sx, sy = correlate(rate_spectrum, mask_spectrum), correlate(mask_spectrum, rate_spectrum)
    vx = (correlate(square_spectrum, mask_spectrum) - sx.square() / n).clamp_min(0)
    vy = (correlate(mask_spectrum, square_spectrum) - sy.square() / n).clamp_min(0)
    denominator = (vx * vy).sqrt()
    result = (correlate(rate_spectrum, rate_spectrum) - sx * sy / n) / denominator.clamp_min(1e-12)
    result[(count < 3) | (denominator < 1e-10)] = float("nan")
    return result.clamp(-1, 1)


@dataclass(frozen=True)
class Gridness:
    """Both rotation scores, annulus/correlations, and conservative size eligibility.

    resolvable=False below 16 bins on either axis; True does not validate geometry.
    """

    hexagonal: float
    square: float
    annulus: tuple[float, float]
    correlations: dict[int, float]
    resolvable: bool


@lru_cache(maxsize=32)
def _rotation_geometry(
    shape: tuple[int, int], annulus: tuple[float, float]
) -> tuple[torch.Tensor, tuple[tuple[int, torch.Tensor], ...]]:
    """Reuse private CPU geometry; callers must never mutate these tensors."""
    h, w = shape
    inner, outer = annulus
    y, x = torch.meshgrid(
        torch.arange(h, device="cpu"), torch.arange(w, device="cpu"), indexing="ij"
    )
    x, y = (x.double() - (w - 1) / 2), (y.double() - (h - 1) / 2)
    ring = (x.square() + y.square() >= inner**2) & (x.square() + y.square() <= outer**2)
    grids: list[tuple[int, torch.Tensor]] = []
    for angle in (30, 45, 60, 90, 120, 135, 150):
        a = math.radians(angle)
        grid = torch.stack(
            (
                (x * math.cos(a) - y * math.sin(a)) / ((w - 1) / 2),
                (x * math.sin(a) + y * math.cos(a)) / ((h - 1) / 2),
            ),
            -1,
        )[None]
        grids.append((angle, grid))
    return ring, tuple(grids)


def gridness(rate: torch.Tensor, *, annulus: tuple[float, float] | None = None) -> Gridness:
    """Score SAC rotations: min(r60,r120)-max(r30,r90,r150), r90-max(r45,r135).

    Bilinear rotation excludes any pixel touching an undefined SAC bin. Defaults
    use 0.2--0.8 of the smaller map extent; supply an annulus for known scales.
    Undefined correlations/scores are NaN, never evidence for periodicity.
    """
    sac = spatial_autocorrelogram(rate)
    extent = min(rate.shape) - 1
    inner, outer = annulus if annulus is not None else (0.2 * extent, 0.8 * extent)
    if not 0 <= inner < outer <= extent:
        raise ValueError(
            f"annulus must satisfy 0 <= inner < outer <= {extent}, got {(inner, outer)}"
        )
    ring, grids = _rotation_geometry((sac.shape[0], sac.shape[1]), (inner, outer))
    valid = torch.isfinite(sac)
    values = torch.nan_to_num(sac)[None, None]
    correlations: dict[int, float] = {}
    for angle, grid in grids:
        rotated = F.grid_sample(values, grid, align_corners=True)[0, 0]
        support = F.grid_sample(valid.double()[None, None], grid, align_corners=True)[0, 0]
        use = ring & valid & (support > 1 - 1e-8)
        u, v = sac[use], rotated[use]
        u, v = u - u.mean(), v - v.mean()
        norm = u.norm() * v.norm()
        correlations[angle] = (
            float((u * v).sum() / norm) if use.sum() >= 3 and norm > 1e-12 else math.nan
        )
    c = correlations
    hex_score = min(c[60], c[120]) - max(c[30], c[90], c[150])
    square_score = c[90] - max(c[45], c[135])
    if not all(math.isfinite(v) for v in c.values()):
        hex_score = square_score = math.nan
    resolvable = min(rate.shape) >= 16 and math.isfinite(hex_score)
    return Gridness(hex_score, square_score, (inner, outer), c, resolvable)
