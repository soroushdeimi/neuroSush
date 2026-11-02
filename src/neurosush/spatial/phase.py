"""Theta phase regression and phase precession classification."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

LOCKED_MAX: float = 0.2
PRECESSION_MAX: float = 0.6


@dataclass(frozen=True)
class PhaseRegression:
    """Signed slope in rad/bin along running direction, intercept and concentration."""

    slope: float
    intercept: float
    concentration: float
    at_bound: bool


def circular_linear_regression(
    position: torch.Tensor,
    theta_phase: torch.Tensor,
    *,
    direction: int = 1,
    max_slope: float = 1.0,
    candidates: int = 2001,
) -> PhaseRegression:
    """Fit wrapped theta phase = slope * direction * position + intercept.

    Pass one direction and one field segment at a time. A negative fitted slope
    means earlier theta phases along travel. Search bounds limit aliasing; boundary
    optima and low concentration need caution. No p-value or field detection is inferred.

    Shapes/Units:
        position: 1D tensor of positions in spatial units;
        theta_phase: 1D tensor of phases in radians [0, 2pi);
        slope: radians per position unit.
    """
    x, phase = position.detach().double().cpu(), theta_phase.detach().double().cpu()
    if (
        x.ndim != 1
        or phase.shape != x.shape
        or x.numel() < 3
        or not torch.isfinite(x).all()
        or not torch.isfinite(phase).all()
        or x.max() == x.min()
    ):
        raise ValueError(
            "need at least three finite paired phases/positions with position variation"
        )
    if direction not in (-1, 1) or not math.isfinite(max_slope) or max_slope <= 0 or candidates < 3:
        raise ValueError("direction must be +/-1, max_slope positive, candidates >=3")
    slopes = torch.linspace(-max_slope, max_slope, candidates, dtype=torch.float64)
    residual = phase[None] - slopes[:, None] * (direction * x[None])
    cosine, sine = residual.cos().mean(-1), residual.sin().mean(-1)
    concentration = torch.hypot(cosine, sine)
    best = int(concentration.argmax())
    return PhaseRegression(
        float(slopes[best]),
        float(torch.atan2(sine[best], cosine[best])),
        float(concentration[best]),
        best in (0, candidates - 1),
    )


def phase_slope_class(slope: float | torch.Tensor) -> str:
    """Classify phase precession slope magnitude: 'locked', 'precession', or 'independent'."""
    val = float(slope.item()) if isinstance(slope, torch.Tensor) else float(slope)
    if not math.isfinite(val):
        raise ValueError(f"slope must be finite, got {val}")
    abs_slope = abs(val)
    if abs_slope <= LOCKED_MAX:
        return "locked"
    if abs_slope < PRECESSION_MAX:
        return "precession"
    return "independent"
