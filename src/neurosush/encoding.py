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
    """Draw spikes using a rate Poisson encoder.

    Args:
        x: Input tensor of rates.
        steps: Number of steps.
        ratio: Scaling ratio.
        generator: Random number generator.

    Returns:
        A boolean tensor of shape (steps, *x.shape).
    """
    _validate_poisson(x, steps, ratio)
    noise = torch.rand((steps, *x.shape), generator=generator, device=x.device, dtype=x.dtype)
    return noise < x * ratio


def interval_poisson(
    x: torch.Tensor, steps: int, *, ratio: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Draw spikes using an interval Poisson encoder.

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

    spikes_plus_one = torch.zeros((steps + 1, n), dtype=torch.bool, device=x.device)
    indices = torch.where((times > 0) & (times <= steps), times, torch.zeros_like(times))
    # We need to use scatter_ with indices and active elements.
    # But the spec says "send them (and the zeros of inactive elements) to an extra row 0"
    # This implies we scatter for all i, e.
    # For inactive elements, times is 0, so indices is 0.
    # For active elements with times > steps, indices is 0.

    # We need to flatten indices and create a flat version of elements to scatter.
    # Actually, scatter_ works on the dimension we specify.
    # We want to scatter at (indices[i, e], e)
    # indices is (steps, n), we want to scatter into (steps+1, n)

    # Let's use the 2D version of scatter_ if possible, or flatten.
    # scatter_(dim, index, src)
    # If dim=0, it scatters along rows.
    # index must have same shape as src.

    # We want to scatter True.
    # src = torch.ones((steps, n), dtype=torch.bool, device=x.device)
    # But we only want to scatter where it's valid?
    # No, the spec says "send them ... to an extra row 0".
    # If we scatter True at (0, e) for inactive elements, it's fine, they get dropped.

    # Wait, if we scatter True at (0, e) for an inactive element, it's fine.
    # If we scatter True at (0, e) for an active element with times > steps, it's fine.

    # Let's use scatter_
    # We need to make sure indices is long.
    indices = indices.long()
    # We need a source tensor of the same shape as indices.
    src = torch.ones((steps, n), dtype=torch.bool, device=x.device)
    spikes_plus_one.scatter_(0, indices, src)

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
