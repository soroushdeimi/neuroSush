"""The Diehl and Cook MNIST example trains and evaluates on tiny random data."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "diehl_cook_mnist.py"


def load():
    spec = importlib.util.spec_from_file_location("diehl_cook_mnist", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["diehl_cook_mnist"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rule", ["pair", "triplet"])
def test_trains_and_evaluates_on_random_images(tmp_path, rule):
    dc = load()
    gen = torch.Generator().manual_seed(0)
    images = (torch.rand(40, 784, generator=gen) * 255).to(torch.uint8)
    labels = torch.randint(0, 10, (40,), generator=gen)
    model = dc.build_network(10, "cpu", seed=0, rule=rule)
    before = model.syn.weights.clone()
    stats = dc.train(
        model, images, labels, samples=20, steps=60, log_every=10, checkpoint=tmp_path / "c.pt"
    )
    assert (model.syn.weights != before).any()
    assert model.exc.theta.max() > 0
    assert stats["sim_steps"] >= 20 * 60
    assert (tmp_path / "c.pt").exists()
    result = dc.evaluate(
        model, images[:20], labels[:20], images[20:], labels[20:], batch=10, steps=60
    )
    assert 0.0 <= result["accuracy"] <= 1.0
    assert sum(result["assignment_histogram"]) == 10
    json.dumps(result)


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="rule"):
        load().build_network(4, "cpu", rule="hebb")


def test_rest_relaxes_theta_after_every_presentation():
    dc = load()
    images = (torch.rand(5, 784, generator=torch.Generator().manual_seed(1)) * 255).to(torch.uint8)
    labels = torch.zeros(5, dtype=torch.long)
    theta = {}
    for rest in (0, 150):
        model = dc.build_network(6, "cpu", seed=0)
        dc.train(model, images, labels, samples=5, steps=60, log_every=5, rest=rest)
        theta[rest] = model.exc.theta.clone()
    assert theta[0].sum() > 0
    assert theta[150].sum() < theta[0].sum()
    model = dc.build_network(3, "cpu")
    model.exc.theta.fill_(2.0)
    dc.relax_theta(model, 100)
    expected = 2.0 * (1 - 1 / dc.THETA_TAU) ** 100
    assert torch.allclose(model.exc.theta, torch.full((3,), expected))
    assert torch.allclose(model.exc.threshold, model.exc.base_threshold + model.exc.theta)
