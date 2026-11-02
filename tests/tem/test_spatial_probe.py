"""Analytical spatial controls and frozen-model extraction."""

import pytest
import torch
from experiments.tem_spatial_probe import corr, decoder, extract, information, measure, shifted

from neurosush.tem import TEM, TEMConfig


def test_position_tuning_and_nulls():
    rng = torch.Generator().manual_seed(23)
    pos = torch.randint(64, (1800,), generator=rng)
    activity = torch.nn.functional.one_hot(pos, 64).double() * torch.arange(1, 65)
    counts = pos.bincount(minlength=64).double()
    expected = -(counts / counts.sum()).log2()
    torch.testing.assert_close(information(activity, pos), expected)
    acc, null = decoder(activity, pos, rng, 100)
    assert acc == pytest.approx(1)
    assert null < 0.04
    shuffled = shifted(pos, rng)
    assert torch.equal(pos.sort().values, shuffled.sort().values)
    # Independently compute a shifted unit's contingency-table information.
    joint = shuffled[pos == 0].bincount(minlength=64).double()
    occupancy = shuffled.bincount(minlength=64).double()
    ratio = joint / occupancy / (pos == 0).double().mean()
    expected_shift = (
        (occupancy / len(pos)) * torch.where(ratio > 0, ratio * ratio.clamp_min(1e-300).log2(), 0)
    ).sum()
    torch.testing.assert_close(information(activity, shuffled)[0], expected_shift)
    result = measure([(activity, pos, pos), (activity, pos, pos)], repeats=10, si_repeats=8)
    assert result["frac_circ"] == 1
    assert result["frac_perm"] == 1
    assert result["splithalf"] == pytest.approx(1)
    assert result["map_corr"] == pytest.approx(1)
    assert result["rsa"] == pytest.approx(1)


def test_random_units_and_failure_cases():
    rng = torch.Generator().manual_seed(19)
    pos = torch.randint(64, (2500,), generator=rng)
    activity = torch.rand(2500, 12, generator=rng)
    result = measure([(activity, pos, pos), (activity, pos, pos)], repeats=10, si_repeats=32)
    assert result["frac_perm"] < 0.25
    assert abs(result["splithalf"]) < 0.15
    assert result["dec"] < 0.04
    assert information(torch.ones_like(activity), pos).abs().max() < 1e-10
    assert torch.isnan(torch.tensor(corr(torch.ones(3), torch.arange(3))))
    with pytest.raises(ValueError, match="200"):
        shifted(pos[:200], rng)
    with pytest.raises(ValueError, match="two worlds"):
        measure([(activity, pos, pos)])


def test_random_model_control():
    model = TEM(TEMConfig(modules=(4,), widths=(8,)), generator=torch.Generator().manual_seed(0))
    streams = extract(model, worlds=2, steps=520, seed=1000000)
    for worlds in streams.values():
        assert len(worlds) == 2
        for activity, pos, sensory in worlds:
            assert activity.shape[0] == pos.numel() == sensory.numel() == 221
            assert torch.isfinite(activity).all()
            for node in pos.unique():
                assert sensory[pos == node].unique().numel() == 1
    with pytest.raises(ValueError, match="200"):
        extract(model, worlds=2, steps=500, seed=0)
