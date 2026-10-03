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


@pytest.mark.parametrize("rule", ["pair", "triplet"])
def test_independent_members_train_and_evaluate(tmp_path, rule):
    dc = load()
    gen = torch.Generator().manual_seed(0)
    images = (torch.rand(40, 784, generator=gen) * 255).to(torch.uint8)
    labels = torch.randint(0, 10, (40,), generator=gen)
    model = dc.build_network(10, "cpu", seed=0, rule=rule, members=2)
    assert model.net.independent
    assert model.syn.weights.shape == (2, 784, 10)
    assert not torch.equal(model.syn.weights[0], model.syn.weights[1])
    before = model.syn.weights.clone()
    stats = dc.train(
        model, images, labels, samples=20, steps=60, log_every=10, checkpoint=tmp_path / "c.pt"
    )
    assert (model.syn.weights != before).flatten(1).any(1).all()
    assert (model.exc.theta.max(-1).values > 0).all()
    assert len(stats["running_accuracy"]) >= 1
    assert all(len(point) == 2 for point in stats["running_accuracy"])
    sums = model.syn.weights.sum(-2)
    assert torch.allclose(sums, torch.full_like(sums, dc.WEIGHT_SUM))
    result = dc.evaluate(
        model, images[:20], labels[:20], images[20:], labels[20:], batch=10, steps=60
    )
    assert len(result["accuracies"]) == 2
    assert result["accuracy"] == pytest.approx(sum(result["accuracies"]) / 2)
    assert result["accuracy_std"] >= 0
    json.dumps(result)
    json.dumps(stats)


def test_members_and_batch_cannot_be_combined():
    with pytest.raises(ValueError, match="members"):
        load().build_network(4, "cpu", batch=2, members=2)


def test_repeat_leaves_satisfied_members_alone():
    dc = load()
    model = dc.build_network(6, "cpu", seed=0, members=2)
    image = torch.zeros(784, dtype=torch.uint8)
    image[:200] = 255
    # member 0 sees a bright image, member 1 a black one that can never reach 5 spikes
    images = torch.stack([image, torch.zeros_like(image)])
    before = model.syn.weights.clone()
    counts, info = dc.present(model, images, 60, max_intensity=4.0, rest=150)
    assert info["repeats"] >= 1
    assert counts.shape == (2, 6)
    assert torch.equal(model.syn.weights[1], before[1])  # the black image changes nothing
