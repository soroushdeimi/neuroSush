"""Bernoulli spiking sampling and group averaging helpers."""

from __future__ import annotations

import torch


def bernoulli_group_means(
    prob: torch.Tensor,
    group_size: int = 2,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample Bernoulli spikes and average over consecutive groups of group_size along last dim.

    Args:
        prob: Probability tensor of shape (*batch, units).
        group_size: Number of consecutive units to group and average (>= 1).
        generator: Optional random number generator.

    Returns:
        Grouped mean tensor of shape (*batch, units // group_size).
    """
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}")
    units = prob.shape[-1]
    if units % group_size != 0:
        raise ValueError(f"units ({units}) must be divisible by group_size ({group_size})")
    spikes = torch.bernoulli(prob, generator=generator)
    shape = list(spikes.shape)
    shape[-1:] = [units // group_size, group_size]
    grouped = spikes.reshape(shape)
    return grouped.mean(dim=-1)
