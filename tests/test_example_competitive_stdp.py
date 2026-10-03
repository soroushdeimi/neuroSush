"""Smoke test of examples/competitive_stdp.py with tiny sizes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "competitive_stdp", Path(__file__).resolve().parents[1] / "examples" / "competitive_stdp.py"
)
assert SPEC is not None
assert SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def test_run_keeps_weights_bounded() -> None:
    result = example.run(10.0, 2, 1, n_exc=50, n_inh=10, correlated=10, seed=0)
    w = result["weights"]
    assert w.shape == (50,)
    assert float(w.min()) >= 0.0
    assert float(w.max()) <= 1.0
    assert len(result["rates"]) == 2


def test_weights_move_under_stdp() -> None:
    result = example.run(10.0, 2, 1, n_exc=1000, n_inh=10, seed=0)
    assert float(result["weights"].mean()) < 1.0


def test_correlated_input_keeps_marginal_rate_and_correlates() -> None:
    from neurosush.core.network import Network, NeuronGroup

    net = Network(seed=0)
    group = NeuronGroup(net, 40, [example.CorrelatedPoissonInput(0.05, 20, 0.5)], name="x")
    net.initialize()
    spikes = []
    for _ in range(4000):
        net.run(1)
        spikes.append(group.spikes.clone())
    s = torch.stack(spikes).float()
    assert abs(float(s.mean()) - 0.05) < 0.01
    assert float(s[:, :20].mean()) > 0.04
    assert float(s[:, 20:].mean()) > 0.04
    corr = torch.corrcoef(s.T)
    assert float(corr[:20, :20].mean()) > float(corr[20:, 20:].mean()) + 0.05


def test_histogram_and_bimodality() -> None:
    w = torch.tensor([0.0, 0.05, 0.95, 1.0, 0.5])
    assert example.bimodality(w) == pytest.approx((0.4, 0.4))
    assert len(example.histogram(w).splitlines()) == 10
