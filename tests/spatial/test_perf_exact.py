"""Frozen pre-optimization reference and exact performance regression tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from neurosush.spatial import gridness, rate_maps, shuffled_position_control
from neurosush.spatial.gridness import spatial_autocorrelogram


def _old_sac(rate: torch.Tensor) -> torch.Tensor:
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

    def correlate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        result = torch.fft.irfft2(
            torch.fft.rfft2(a, s=shape) * torch.fft.rfft2(b, s=shape).conj(), s=shape
        )
        return torch.roll(result, (h - 1, w - 1), (0, 1))

    count = correlate(mask, mask).round().clamp_min(0)
    n = count.clamp_min(1)
    sx, sy = correlate(r, mask), correlate(mask, r)
    vx = (correlate(r.square(), mask) - sx.square() / n).clamp_min(0)
    vy = (correlate(mask, r.square()) - sy.square() / n).clamp_min(0)
    denominator = (vx * vy).sqrt()
    result = (correlate(r, r) - sx * sy / n) / denominator.clamp_min(1e-12)
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


def _old_gridness(rate: torch.Tensor, *, annulus: tuple[float, float] | None = None) -> Gridness:
    """Score SAC rotations: min(r60,r120)-max(r30,r90,r150), r90-max(r45,r135).

    Bilinear rotation excludes any pixel touching an undefined SAC bin. Defaults
    use 0.2--0.8 of the smaller map extent; supply an annulus for known scales.
    Undefined correlations/scores are NaN, never evidence for periodicity.
    """
    sac = _old_sac(rate)
    extent = min(rate.shape) - 1
    inner, outer = annulus if annulus is not None else (0.2 * extent, 0.8 * extent)
    if not 0 <= inner < outer <= extent:
        raise ValueError(
            f"annulus must satisfy 0 <= inner < outer <= {extent}, got {(inner, outer)}"
        )
    h, w = sac.shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    x, y = (x.double() - (w - 1) / 2), (y.double() - (h - 1) / 2)
    ring = (x.square() + y.square() >= inner**2) & (x.square() + y.square() <= outer**2)
    valid = torch.isfinite(sac)
    values = torch.nan_to_num(sac)[None, None]
    correlations: dict[int, float] = {}
    for angle in (30, 45, 60, 90, 120, 135, 150):
        a = math.radians(angle)
        grid = torch.stack(
            (
                (x * math.cos(a) - y * math.sin(a)) / ((w - 1) / 2),
                (x * math.sin(a) + y * math.cos(a)) / ((h - 1) / 2),
            ),
            -1,
        )[None]
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


def _old_rate_maps(
    activity: torch.Tensor, nodes: torch.Tensor, shape: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit-first occupancy means and counts; unseen bins remain NaN.

    Activity is (samples, units); each sample has equal duration. For spikes,
    divide by sample duration in seconds first to obtain Hz.
    """
    activity = activity.detach().to(device="cpu", dtype=torch.float64)
    nodes = nodes.detach().to(device="cpu", dtype=torch.long)
    size = math.prod(shape)
    if not shape or min(shape) < 1 or activity.ndim != 2 or nodes.shape != activity.shape[:1]:
        raise ValueError("expected positive shape, activity (samples, units), nodes (samples,)")
    if not torch.isfinite(activity).all() or (nodes < 0).any() or (nodes >= size).any():
        raise ValueError("activity must be finite and nodes must lie within shape")
    counts = torch.bincount(nodes, minlength=size)
    total = torch.zeros(size, activity.shape[1], dtype=activity.dtype)
    total.index_add_(0, nodes, activity)
    means = total / counts[:, None].clamp_min(1)
    means[counts == 0] = float("nan")
    return means.T.reshape(activity.shape[1], *shape), counts.reshape(shape)


def _old_information(rate: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
    """Skaggs information in bits/spike for nonnegative unit-first rate maps.

    Unoccupied bins are excluded; silent cells return zero. Finite rates are
    required at occupied bins. This is not a bias-corrected significance test.
    """
    counts = occupancy.detach().double().cpu().flatten()
    if (counts < 0).any() or not torch.isfinite(counts).all() or counts.sum() <= 0:
        raise ValueError("occupancy must be finite, nonnegative and nonempty")
    if rate.ndim != occupancy.ndim + 1 or rate.shape[1:] != occupancy.shape:
        raise ValueError("rate must have shape (units, *occupancy.shape)")
    r = rate.detach().double().cpu().reshape(-1, occupancy.numel())
    r = r[:, counts > 0]
    if not torch.isfinite(r).all() or (r < 0).any():
        raise ValueError("occupied rates must be finite and nonnegative")
    p = counts[counts > 0] / counts.sum()
    mean = (r * p).sum(-1, keepdim=True)
    ratio = r / mean.clamp_min(torch.finfo(r.dtype).tiny)
    terms = torch.where(ratio > 0, ratio * ratio.clamp_min(1e-300).log2(), 0)
    return (p * terms).sum(-1)


@dataclass(frozen=True)
class ShuffleNull:
    """Position-shuffle null arrays of shape (repeats, units); grid scores NaN in 1D."""

    hexagonal: torch.Tensor
    square: torch.Tensor
    information: torch.Tensor


def _old_shuffle(
    activity: torch.Tensor,
    nodes: torch.Tensor,
    shape: tuple[int, ...],
    *,
    repeats: int = 32,
    seed: int = 0,
    annulus: tuple[float, float] | None = None,
) -> ShuffleNull:
    """Shuffle sample-wise node labels, then recompute maps and both spatial nulls.

    This preserves occupancy and per-unit activity distributions. Permuting a
    finished rate map would preserve spatial information at uniform occupancy
    and is NOT this control. IID label shuffling destroys temporal correlations;
    it is a simple diagnostic, not a substitute for lap/block-preserving controls.
    """
    if repeats < 1 or len(shape) not in (1, 2):
        raise ValueError(f"need positive repeats and 1D/2D shape, got {repeats}, {shape}")
    _old_rate_maps(activity, nodes, shape)
    labels = nodes.detach().cpu()
    generator = torch.Generator().manual_seed(seed)
    hex_scores: list[list[float]] = []
    square_scores: list[list[float]] = []
    information: list[torch.Tensor] = []
    for _ in range(repeats):
        shuffled = labels[torch.randperm(len(labels), generator=generator)]
        maps, occupancy = _old_rate_maps(activity, shuffled, shape)
        information.append(_old_information(maps, occupancy))
        scores = [_old_gridness(m, annulus=annulus) for m in maps] if len(shape) == 2 else []
        hex_scores.append([s.hexagonal for s in scores] if scores else [math.nan] * len(maps))
        square_scores.append([s.square for s in scores] if scores else [math.nan] * len(maps))
    return ShuffleNull(
        torch.tensor(hex_scores, dtype=torch.float64),
        torch.tensor(square_scores, dtype=torch.float64),
        torch.stack(information),
    )


def _exact(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(actual[~torch.isnan(actual)], expected[~torch.isnan(expected)])


@pytest.mark.parametrize("shape", [(4, 4), (5, 8), (16, 16)])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("annulus", [None, (1.0, 2.0)])
def test_frozen_maps(shape, missing, annulus):
    generator = torch.Generator().manual_seed(41)
    rate = torch.rand(shape, generator=generator, dtype=torch.float64)
    if missing:
        rate[0, ::2] = math.nan
    original = rate.clone()
    _exact(spatial_autocorrelogram(rate), _old_sac(rate))
    actual, expected = gridness(rate, annulus=annulus), _old_gridness(rate, annulus=annulus)
    _exact(
        torch.tensor(list(actual.correlations.values()), dtype=torch.float64),
        torch.tensor(list(expected.correlations.values()), dtype=torch.float64),
    )
    _exact(
        torch.tensor([actual.hexagonal, actual.square], dtype=torch.float64),
        torch.tensor([expected.hexagonal, expected.square], dtype=torch.float64),
    )
    assert actual.annulus == expected.annulus
    assert actual.resolvable == expected.resolvable
    _exact(rate, original)


@pytest.mark.parametrize("shape", [(7,), (4, 4), (5, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("annulus", [None, (0.5, 2.0)])
def test_frozen_shuffle(shape, dtype, annulus):
    generator = torch.Generator().manual_seed(19)
    activity = torch.rand((173, 3), generator=generator, dtype=dtype)
    nodes = torch.randint(math.prod(shape), (173,), generator=generator).float()
    actual = shuffled_position_control(activity, nodes, shape, repeats=3, seed=27, annulus=annulus)
    expected = _old_shuffle(activity, nodes, shape, repeats=3, seed=27, annulus=annulus)
    for name in ("hexagonal", "square", "information"):
        _exact(getattr(actual, name), getattr(expected, name))
    for a, b in zip(
        rate_maps(activity, nodes, shape), _old_rate_maps(activity, nodes, shape), strict=True
    ):
        _exact(a, b)


def test_three_forward_spectra():
    with patch("torch.fft.rfft2", wraps=torch.fft.rfft2) as forward:
        spatial_autocorrelogram(torch.arange(16).reshape(4, 4))
    assert forward.call_count == 3


def test_shuffle_counts_once():
    with patch("torch.bincount", wraps=torch.bincount) as counts:
        shuffled_position_control(torch.ones(12, 2), torch.arange(12) % 4, (4,), repeats=3)
    assert counts.call_count == 1


def test_geometry_reused():
    with patch("torch.meshgrid", wraps=torch.meshgrid) as mesh:
        for _ in range(2):
            gridness(torch.arange(77).reshape(7, 11), annulus=(1.125, 3.25))
    assert mesh.call_count == 1


@pytest.mark.parametrize("value", [0.0, math.nan])
def test_undefined_maps(value):
    rate = torch.full((4, 6), value)
    _exact(spatial_autocorrelogram(rate), _old_sac(rate))
    assert math.isnan(gridness(rate).hexagonal)


@pytest.mark.parametrize(
    ("activity", "nodes", "shape"),
    [
        (torch.ones(3, 1), torch.tensor([0, 1, 4]), (2, 2)),
        (torch.tensor([[math.nan]]), torch.tensor([0]), (2,)),
        (torch.ones(3, 1), torch.tensor([0, 1]), (2,)),
        (torch.ones(3, 1), torch.tensor([0, 0, 0]), (0,)),
    ],
)
def test_shuffle_invalid_before_randomness(activity, nodes, shape):
    with (
        patch("torch.randperm", wraps=torch.randperm) as permutation,
        pytest.raises(ValueError, match=r"shape|finite"),
    ):
        shuffled_position_control(activity, nodes, shape)
    assert permutation.call_count == 0


def test_empty_samples_and_hand_counts():
    activity = torch.empty(0, 2)
    nodes = torch.empty(0, dtype=torch.long)
    maps, counts = rate_maps(activity, nodes, (2,))
    assert counts.tolist() == [0, 0]
    assert maps.isnan().all()
    with pytest.raises(ValueError, match="nonempty"):
        shuffled_position_control(activity, nodes, (2,))


def test_geometry_cache_bound_and_keys():
    from neurosush.spatial.gridness import _rotation_geometry

    _rotation_geometry.cache_clear()
    first = _rotation_geometry((9, 11), (1.0, 3.0))
    assert first is _rotation_geometry((9, 11), (1.0, 3.0))
    assert first is not _rotation_geometry((11, 9), (1.0, 3.0))
    assert first is not _rotation_geometry((9, 11), (1.5, 3.0))
    for index in range(40):
        _rotation_geometry((9, 11), (index / 100, 3.0))
    assert _rotation_geometry.cache_info().currsize == 32
    assert first[0].device.type == "cpu"
    assert all(grid.dtype == torch.float64 for _, grid in first[1])
