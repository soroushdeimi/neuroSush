"""Poisson and latency spike encoders."""

from __future__ import annotations

import torch


def _validate_poisson(x: torch.Tensor, steps: int, ratio: float) -> None:
    """Validate Poisson encoder arguments.

    Args:
        x: Input tensor.
        steps: Number of steps.
        ratio: Scaling ratio.

    Raises:
        ValueError: If arguments are invalid.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if ratio < 0:
        raise ValueError(f"ratio must be >= 0, got {ratio}")
    if (x < 0).any():
        raise ValueError("x must be non-negative")


def rate_poisson(
    x: torch.Tensor, steps: int, *, ratio: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Bernoulli spikes: every step, each element spikes with probability ``x * ratio``.

    Args:
        x: Input tensor of rates.
        steps: Number of steps.
        ratio: Scaling ratio.
        generator: Random number generator.

    Returns:
        A boolean tensor of shape (steps, *x.shape).
    """
    _validate_poisson(x, steps, ratio)
    noise = torch.rand((steps, *x.shape), generator=generator, device=x.device)
    return noise < x * ratio


def interval_poisson(
    x: torch.Tensor, steps: int, *, ratio: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Spikes whose inter-spike intervals are Poisson with mean ``1 / (x * ratio)`` steps.

    Zero intervals are raised to one step; zero intensities never spike.

    Args:
        x: Input tensor of intensities.
        steps: Number of steps.
        ratio: Scaling ratio.
        generator: Random number generator.

    Returns:
        A boolean tensor of shape (steps, *x.shape).
    """
    _validate_poisson(x, steps, ratio)
    n = x.numel()
    x_flat = x.flatten().float()
    rate = x_flat * ratio
    active = rate > 0

    mean_interval = torch.where(active, 1 / rate.clamp_min(1e-12), torch.zeros_like(rate))
    intervals = torch.poisson(mean_interval.expand(steps, -1), generator=generator)
    intervals = torch.where(
        active.unsqueeze(0) & (intervals == 0), torch.ones_like(intervals), intervals
    )

    times = torch.cumsum(intervals, dim=0)

    # A time t (1-based) is row t - 1; late times and inactive elements go to row 0,
    # which is dropped.
    rows = torch.where(times <= steps, times, torch.zeros_like(times)).long()
    spikes_plus_one = torch.zeros((steps + 1, n), dtype=torch.bool, device=x.device)
    spikes_plus_one.scatter_(0, rows, True)
    return spikes_plus_one[1:].reshape(steps, *x.shape)


def intensity_to_latency(
    x: torch.Tensor,
    steps: int,
    *,
    threshold: float | None = None,
    sparsity: float | None = None,
    value_range: tuple[float, float] = (0.0, 1.0),
    trim_low: bool = True,
    trim_high: bool = True,
) -> torch.Tensor:
    """One spike per active element; stronger values spike earlier.

    Args:
        x: Input tensor of intensities.
        steps: Number of steps.
        threshold: Threshold for activity.
        sparsity: Fraction of elements to keep active.
        value_range: Range of input values.
        trim_low: Whether to trim the lowest active level.
        trim_high: Whether to trim the highest active level.

    Returns:
        A boolean tensor of shape (steps, *x.shape).
    """
    low, high = value_range
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if high <= low:
        raise ValueError(f"value_range high must be > low, got {value_range}")
    if threshold is not None and sparsity is not None:
        raise ValueError("cannot provide both threshold and sparsity")
    if sparsity is not None and (sparsity <= 0 or sparsity > 1):
        raise ValueError(f"sparsity must be in (0, 1], got {sparsity}")

    if sparsity is not None:
        threshold = torch.quantile(x.flatten().float(), 1 - sparsity)
    elif threshold is None:
        threshold = low

    active = x >= threshold
    level = (x - low) / (high - low)

    if active.any():
        if trim_low:
            w = level[active].min()
            level = (level - w) / (1 - w) if w < 1 else level - w
        if trim_high:
            max_level = level[active].max()
            if max_level > 0:
                level = level / max_level

    time = torch.round((1 - level.clamp(0, 1)) * (steps - 1)).long()
    spikes = torch.zeros((steps, *x.shape), dtype=torch.bool, device=x.device)
    spikes.scatter_(0, time.unsqueeze(0), active.unsqueeze(0))
    return spikes
