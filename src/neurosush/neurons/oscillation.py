"""Imposed oscillatory inhibitory input current."""

from __future__ import annotations

import math

import torch


def theta_current(
    t: float | torch.Tensor, alpha: float = 10.0, freq: float = 8.0
) -> float | torch.Tensor:
    """Return theta modulation current: -alpha * (sin(2 * pi * freq * t) + 1).

    Args:
        t: Time in seconds (scalar or tensor).
        alpha: Modulation amplitude.
        freq: Frequency in Hz (> 0).

    Returns:
        Inhibitory current (float or tensor).
    """
    if freq <= 0:
        raise ValueError(f"freq must be positive, got {freq}")
    if isinstance(t, torch.Tensor):
        return -alpha * (torch.sin(2 * math.pi * freq * t) + 1.0)
    return -alpha * (math.sin(2 * math.pi * freq * t) + 1.0)
