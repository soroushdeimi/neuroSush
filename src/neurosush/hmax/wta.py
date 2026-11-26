"""Convolutional winner-take-all selection and simplified STDP (Kheradpisheh et al. 2018).

``select_conv_winners`` picks at most ``k`` winning neurons from a set of feature maps, one per
map, no two within a Chebyshev radius. ``simplified_stdp`` applies the simplified STDP rule of
Kheradpisheh et al. 2018, Eq. 3, and ``stdp_convergence`` reports the Eq. 4 convergence measure.
"""

from __future__ import annotations

import math

import torch


def select_conv_winners(
    first_spike_time: torch.Tensor,
    potential: torch.Tensor,
    k: int,
    radius: int,
) -> torch.Tensor:
    """Select convolutional winners in acceptance order.

    Candidates are the entries with a finite ``first_spike_time``, sorted by time ascending,
    potential descending, then c, y, x ascending. Walking that order, a candidate is accepted if
    its feature map has no accepted winner yet and no accepted winner lies within Chebyshev
    distance ``radius`` (any map). Selection stops after ``k`` winners.

    Args:
        first_spike_time: First spike time per entry, shape (C, H, W). ``+inf`` means never spiked.
        potential: Potential per entry, shape (C, H, W).
        k: Maximum number of winners to accept.
        radius: Chebyshev exclusion radius between accepted winners.

    Returns:
        A long tensor of shape (n, 3) with rows (c, y, x) in acceptance order; n <= k. Shape
        (0, 3) when nothing is accepted.

    Raises:
        ValueError: If k < 1, radius < 0, the inputs are not 3-D, or their shapes differ.
    """
    if first_spike_time.dim() != 3 or potential.dim() != 3:
        raise ValueError("first_spike_time and potential must be 3-D")
    if first_spike_time.shape != potential.shape:
        raise ValueError(
            "shape mismatch: first_spike_time "
            f"{tuple(first_spike_time.shape)} != potential {tuple(potential.shape)}"
        )
    if k < 1:
        raise ValueError("k must be at least 1")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    c, h, w = first_spike_time.shape
    candidates = []
    for ci in range(c):
        for yi in range(h):
            for xi in range(w):
                t = first_spike_time[ci, yi, xi].item()
                if math.isinf(t):
                    continue
                p = potential[ci, yi, xi].item()
                candidates.append((t, -p, ci, yi, xi))
    candidates.sort()

    winners: list[tuple[int, int, int]] = []
    for _, _, ci, yi, xi in candidates:
        if len(winners) >= k:
            break
        if any(wc == ci for wc, _, _ in winners):
            continue
        if any(max(abs(yi - wy), abs(xi - wx)) <= radius for _, wy, wx in winners):
            continue
        winners.append((ci, yi, xi))

    if not winners:
        return torch.empty((0, 3), dtype=torch.long)
    return torch.tensor(winners, dtype=torch.long)


def simplified_stdp(
    w: torch.Tensor,
    pre_time: torch.Tensor,
    post_time: float,
    a_plus: float,
    a_minus: float,
) -> torch.Tensor:
    """Simplified STDP weight update (Kheradpisheh et al. 2018, Eq. 3).

    For each synapse: if ``pre_time <= post_time`` then ``dw = +a_plus * w * (1 - w)`` (LTP),
    otherwise ``dw = -a_minus * w * (1 - w)`` (LTD). A presynaptic time of ``+inf`` (never spiked)
    takes the LTD branch.

    Args:
        w: Synaptic weights in [0, 1], any shape.
        pre_time: Presynaptic first spike time, same shape as ``w``. ``+inf`` = never spiked.
        post_time: Postsynaptic first spike time (a single float).
        a_plus: LTP learning rate (non-negative).
        a_minus: LTD learning rate magnitude (non-negative).

    Returns:
        A new tensor ``w + dw`` with the same shape and dtype as ``w``.

    Raises:
        ValueError: If ``w`` and ``pre_time`` shapes differ, any element of ``w`` is outside
            [0, 1], or ``a_plus`` / ``a_minus`` is negative.
    """
    if w.shape != pre_time.shape:
        raise ValueError(f"shape mismatch: w {tuple(w.shape)} != pre_time {tuple(pre_time.shape)}")
    out_of_range = (w < 0) | (w > 1)
    if bool(out_of_range.any()):
        raise ValueError(f"w must be in [0, 1], got {w[out_of_range].flatten()[0].item()}")
    if a_plus < 0:
        raise ValueError(f"a_plus must be non-negative, got {a_plus}")
    if a_minus < 0:
        raise ValueError(f"a_minus must be non-negative, got {a_minus}")

    ltp = pre_time <= post_time
    dw = torch.where(ltp, a_plus * w * (1 - w), -a_minus * w * (1 - w))
    return w + dw


def stdp_convergence(w: torch.Tensor) -> torch.Tensor:
    """STDP convergence measure (Kheradpisheh et al. 2018, Eq. 4).

    Args:
        w: Synaptic weights in [0, 1], any shape.

    Returns:
        A 0-dim tensor ``mean(w * (1 - w))`` over all elements.

    Raises:
        ValueError: If any element of ``w`` is outside [0, 1].
    """
    out_of_range = (w < 0) | (w > 1)
    if bool(out_of_range.any()):
        raise ValueError(f"w must be in [0, 1], got {w[out_of_range].flatten()[0].item()}")
    return torch.mean(w * (1 - w))
