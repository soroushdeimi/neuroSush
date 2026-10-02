"""Smoke tests of the examples that need no dataset."""

import importlib.util
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_stdp_frequency_triplet_potentiates_at_high_frequency():
    ex = load("stdp_frequency")
    result = ex.run(frequencies=(10.0, 50.0), pairings=20)
    assert result[("pair", 10.0, 10)] > 0 > result[("pair", 10.0, -10)]
    assert result[("triplet", 10.0, -10)] < 0
    assert result[("triplet", 50.0, 10)] > 0
    assert result[("triplet", 50.0, -10)] > 0
    assert result[("pair", 50.0, -10)] < 0


def test_balanced_network_rate_moves_towards_target():
    ex = load("balanced_network")
    result = ex.run(seconds=8, window=2)
    rates = result["rates"]
    assert rates[0] > 2 * ex.TARGET_HZ
    assert rates[-1] < rates[0] / 2
    assert result["weights"].mean() > 0.02


def test_predictive_coding_learns_a_synthetic_task():
    import torch

    ex = load("predictive_coding_mnist")
    gen = torch.Generator().manual_seed(0)
    prototypes = torch.rand(10, 784, generator=gen)
    labels = torch.arange(400) % 10
    images = (prototypes[labels] + 0.1 * torch.randn(400, 784, generator=gen)).clamp(0, 1)
    scores = ex.run(
        images[:300], labels[:300], images[300:], labels[300:], hidden=32, epochs=3, rate=0.02
    )
    assert len(scores) == 3
    assert scores[-1] > 0.9
