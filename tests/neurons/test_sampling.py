"""Tests for Bernoulli group sampling helper."""

from __future__ import annotations

import pytest
import torch

from neurosush.neurons.sampling import bernoulli_group_means


def test_bernoulli_group_means():
    prob = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    gen = torch.Generator().manual_seed(42)
    res = bernoulli_group_means(prob, group_size=2, generator=gen)
    assert res.shape == (2, 2)

    with pytest.raises(ValueError, match="group_size"):
        bernoulli_group_means(prob, group_size=3)
    with pytest.raises(ValueError, match="group_size"):
        bernoulli_group_means(prob, group_size=0)

    gen1 = torch.Generator().manual_seed(123)
    gen2 = torch.Generator().manual_seed(123)
    r1 = bernoulli_group_means(prob, group_size=2, generator=gen1)
    r2 = bernoulli_group_means(prob, group_size=2, generator=gen2)
    torch.testing.assert_close(r1, r2)

    p0 = torch.zeros(10, 4)
    r0 = bernoulli_group_means(p0, group_size=2)
    torch.testing.assert_close(r0, torch.zeros(10, 2))

    p1 = torch.ones(10, 4)
    r1_all = bernoulli_group_means(p1, group_size=2)
    torch.testing.assert_close(r1_all, torch.ones(10, 2))

    p_target = 0.3
    p_tensor = torch.full((1, 20000), p_target)
    samples = bernoulli_group_means(
        p_tensor, group_size=1000, generator=torch.Generator().manual_seed(99)
    )
    assert abs(float(samples.mean().item()) - p_target) < 0.02
