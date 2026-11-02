"""Tests for spatial field evaluation helpers."""

from __future__ import annotations

import math

import pytest
import torch

from neurosush.spatial.fields import (
    grid_fraction,
    spatial_information_vs_shuffle,
    summarize_classes,
    track_phase_slopes,
)


def synthetic_spikes(
    n: int,
    kind: str,
    *,
    slope: float = 0.0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic spike positions, theta phases, and directions for testing."""
    generator = torch.Generator().manual_seed(seed)
    positions = torch.rand(n, generator=generator) * 10.0
    rand_vals = torch.rand(n, generator=generator)
    directions = torch.where(rand_vals > 0.5, torch.tensor(1), torch.tensor(-1))

    if kind == "locked":
        phases = torch.full((n,), 1.2) + torch.randn(n, generator=generator) * 0.05
    elif kind == "precession":
        noise = torch.randn(n, generator=generator) * 0.05
        effective_slope = slope * directions.double()
        phases = (effective_slope * positions + 1.2 + noise) % (2 * math.pi)
    elif kind == "independent":
        phases = torch.rand(n, generator=generator) * 2 * math.pi
    else:
        raise ValueError(f"unknown synthetic spike kind: {kind}")

    return positions, phases, directions


def test_grid_fraction_small():
    axis = torch.arange(8, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    rate = 2.0 + torch.cos(2 * math.pi * x / 4.0) + torch.cos(2 * math.pi * y / 4.0)
    nodes = torch.arange(64)
    activity = rate.flatten().unsqueeze(1)
    res = grid_fraction(activity, nodes, (8, 8), threshold=0.0)
    assert "hex_fraction" in res
    assert "square_fraction" in res
    assert "label" in res
    assert "caveat" in res
    assert res["n_units"] == 1
    assert res["undefined_count"] == 0


def test_spatial_information_vs_shuffle():
    nodes = torch.arange(64).repeat(40)
    activity = torch.stack(((nodes == 20).double(), torch.ones(len(nodes))), -1)
    res = spatial_information_vs_shuffle(activity, nodes, (8, 8), n_shuffles=20, seed=42)
    assert "mean_info" in res
    assert "shuffle_mean" in res
    assert "shuffle_p95" in res
    assert "fraction_above_p95" in res
    assert res["fraction_above_p95"] > 0.0


def test_track_phase_slopes_and_empty_path():
    pos, phases, dirs = synthetic_spikes(400, "precession", slope=-0.4, seed=42)
    field_edges = [(0.0, 5.0), (9.99, 10.0)]
    records = track_phase_slopes(pos, phases, dirs, [1, -1], field_edges=field_edges)
    assert len(records) > 0
    short_rec = next(r for r in records if r["field"] == (9.99, 10.0))
    assert short_rec["class"] is None

    summary = summarize_classes(records)
    assert "counts" in summary
    assert "fractions" in summary


def test_synthetic_spikes_kinds():
    p_l, ph_l, d_l = synthetic_spikes(50, "locked", seed=1)
    assert p_l.shape == (50,)
    assert ph_l.shape == (50,)
    assert d_l.shape == (50,)

    p_p, ph_p, d_p = synthetic_spikes(400, "precession", slope=-0.4, seed=1)
    records = track_phase_slopes(p_p, ph_p, d_p, [1, -1], field_edges=[(0.0, 10.0)])
    assert len(records) > 0

    p_i, _ph_i, _d_i = synthetic_spikes(50, "independent", seed=1)
    assert p_i.shape == (50,)

    with pytest.raises(ValueError, match="unknown synthetic spike kind"):
        synthetic_spikes(10, "unknown_kind")


def test_seeded_determinism():
    p1, ph1, d1 = synthetic_spikes(100, "precession", slope=-0.3, seed=123)
    p2, ph2, d2 = synthetic_spikes(100, "precession", slope=-0.3, seed=123)
    torch.testing.assert_close(p1, p2)
    torch.testing.assert_close(ph1, ph2)
    torch.testing.assert_close(d1, d2)
