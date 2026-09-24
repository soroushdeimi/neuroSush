"""Sparse distributed representations (SDRs) and their matching statistics.

An SDR is a boolean tensor whose last dimension has ``n`` bits, of which a small number
``w`` are active. Leading dimensions are batches. The probability functions follow
Ahmad and Hawkins (2016), "How do neurons operate on sparse distributed representations?
A mathematical theory of sparsity, neurons and active dendrites", arXiv:1601.00720.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

import torch


def random_sdr(
    n: int,
    w: int,
    *,
    batch: tuple[int, ...] = (),
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Uniformly random SDRs with exactly ``w`` of ``n`` bits active, shape ``(*batch, n)``."""
    _check_n_w(n, w)
    keys = torch.rand((*batch, n), generator=generator, device=device)
    active = keys.topk(w, dim=-1).indices
    sdr = torch.zeros((*batch, n), dtype=torch.bool, device=device)
    return sdr.scatter_(-1, active, True)


def overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Number of bits active in both SDRs (broadcast over leading dimensions)."""
    return (a & b).sum(-1)


def sparsity(a: torch.Tensor) -> torch.Tensor:
    """Fraction of active bits."""
    return a.float().mean(-1)


def union(sdrs: torch.Tensor) -> torch.Tensor:
    """Union of a stack of SDRs along the first dimension."""
    return sdrs.any(0)


def subsample(a: torch.Tensor, k: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Keep ``k`` randomly chosen active bits of each SDR (all of them if it has fewer)."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    keys = torch.rand(a.shape, generator=generator, device=a.device).masked_fill(~a, -1.0)
    k = min(k, a.shape[-1])
    kept = keys.topk(k, dim=-1).indices
    out = torch.zeros_like(a).scatter_(-1, kept, True)
    return out & a


def _check_n_w(n: int, w: int) -> None:
    if n < 1 or not 0 <= w <= n:
        raise ValueError(f"need n >= 1 and 0 <= w <= n, got n={n}, w={w}")


def overlap_count(n: int, w_x: int, w: int, b: int) -> int:
    """Number of ``w``-bit SDRs sharing exactly ``b`` bits with a fixed ``w_x``-bit SDR.

    ``C(w_x, b) * C(n - w_x, w - b)`` (Ahmad and Hawkins 2016, eq. 2).
    """
    _check_n_w(n, w_x)
    _check_n_w(n, w)
    return comb(w_x, b) * comb(n - w_x, w - b)


def match_probability(n: int, w_x: int, w: int, theta: int) -> float:
    """Exact probability that a random ``w``-bit SDR overlaps a fixed one in >= ``theta`` bits.

    The fixed SDR has ``w_x`` active bits (Ahmad and Hawkins 2016, eq. 3). With ``theta`` as
    a neuron's dendritic threshold this is its false-positive rate.
    """
    _check_n_w(n, w_x)
    _check_n_w(n, w)
    if theta < 0:
        raise ValueError(f"theta must be non-negative, got {theta}")
    hits = sum(overlap_count(n, w_x, w, b) for b in range(theta, min(w, w_x) + 1))
    return float(Fraction(hits, comb(n, w)))


def expected_union_size(n: int, w: int, m: int) -> float:
    """Expected active bits in the union of ``m`` random ``w``-bit SDRs.

    ``n * (1 - (1 - w / n) ** m)``.
    """
    _check_n_w(n, w)
    if m < 0:
        raise ValueError(f"m must be non-negative, got {m}")
    return n * (1 - (1 - w / n) ** m)


def union_match_probability(n: int, w: int, m: int, theta: int) -> float:
    """Approximate probability that a random SDR matches a union of ``m`` stored SDRs.

    All SDRs have ``w`` active bits; a match is an overlap of at least ``theta``. Each bit of
    the union is active independently with probability ``p = 1 - (1 - w / n) ** m``, so the
    overlap is ``Binomial(w, p)``
    (Ahmad and Hawkins 2016, section on unions).
    """
    _check_n_w(n, w)
    p = 1 - (1 - w / n) ** m
    return sum(comb(w, b) * p**b * (1 - p) ** (w - b) for b in range(max(theta, 0), w + 1))
