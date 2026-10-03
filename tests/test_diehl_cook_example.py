"""The Diehl and Cook MNIST example trains and evaluates on tiny random data."""

import importlib.util
import json
import sys
from itertools import pairwise
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


def test_dt_half_smoke_run_and_winner_diagnostics():
    dc = load()
    gen = torch.Generator().manual_seed(0)
    images = (torch.rand(12, 784, generator=gen) * 255).to(torch.uint8)
    labels = torch.randint(0, 10, (12,), generator=gen)
    model = dc.build_network(6, "cpu", seed=0, dt=0.5)
    assert model.net.dt == 0.5
    stats = dc.train(model, images, labels, samples=8, steps=120, log_every=4, rest=300, block=5)
    assert stats["sim_steps"] >= 8 * 120
    diag = stats["winner_diagnostics"]
    assert diag["samples"] == 8
    assert 0 <= diag["distinct_neurons_mean"] <= 6
    assert 0 <= diag["top_share_mean"] <= 1
    assert [(b["start"], b["end"]) for b in diag["blocks"]] == [(0, 5), (5, 8)]
    result = dc.evaluate(
        model, images[:6], labels[:6], images[6:], labels[6:], batch=6, steps=120, dt=0.5
    )
    assert 0.0 <= result["accuracy"] <= 1.0
    json.dumps(diag)


def test_winner_diagnostics_per_member():
    dc = load()
    images = (torch.rand(6, 784, generator=torch.Generator().manual_seed(2)) * 255).to(torch.uint8)
    model = dc.build_network(5, "cpu", seed=0, members=2)
    stats = dc.train(model, images, torch.zeros(6, dtype=torch.long), samples=6, steps=60)
    diag = stats["winner_diagnostics"]
    assert len(diag["distinct_neurons_mean"]) == 2
    json.dumps(diag)


def test_exc_block_reproduces_the_original_timer():
    dc = load()
    plain = dc.build_network(3, "cpu")
    assert not hasattr(plain.exc, "fire_blocked")
    model = dc.build_network(3, "cpu", exc_block=50.0)
    refractory = next(b for b in model.exc.behaviors if type(b).__name__ == "Refractory")
    assert (refractory.period, refractory.block) == (5.0, 50.0)
    # voltage held far above threshold: a blocked neuron fires at most every 51 steps
    times = []
    for _ in range(300):
        model.exc.v.copy_(torch.full_like(model.exc.v, 0.0))
        model.net.step()
        if bool(model.exc.spikes[0]):
            times.append(model.net.iteration)
    assert len(times) > 2
    assert all(b - a >= 51 for a, b in pairwise(times))
    # the same drive without the block fires every 6 steps
    free = []
    for _ in range(300):
        plain.exc.v.copy_(torch.full_like(plain.exc.v, 0.0))
        plain.net.step()
        if bool(plain.exc.spikes[0]):
            free.append(plain.net.iteration)
    assert {b - a for a, b in pairwise(free)} == {6}


def test_exc_block_is_recorded_and_used_in_evaluation():
    dc = load()
    images = (torch.rand(8, 784, generator=torch.Generator().manual_seed(1)) * 255).to(torch.uint8)
    labels = torch.randint(0, 10, (8,), generator=torch.Generator().manual_seed(1))
    model = dc.build_network(4, "cpu", seed=0, exc_block=50.0)
    dc.train(model, images, labels, samples=4, steps=60)
    result = dc.evaluate(
        model, images[:4], labels[:4], images[4:], labels[4:], batch=4, steps=60, exc_block=50.0
    )
    assert 0.0 <= result["accuracy"] <= 1.0
    assert "--exc-block" in dc.__doc__
    assert "timer" in dc.__doc__


def test_black_image_hits_the_cap_and_is_counted():
    dc = load()
    model = dc.build_network(6, "cpu", seed=0)
    images = torch.zeros(3, 784, dtype=torch.uint8)
    labels = torch.zeros(3, dtype=torch.long)
    stats = dc.train(model, images, labels, samples=3, steps=20, log_every=3, rest=10)
    assert stats["cap_hits"] == 3.0
    assert stats["presentations_per_sample"] == pytest.approx(dc.MAX_INTENSITY - 1)
    assert stats["max_intensity_used"] == dc.MAX_INTENSITY
    json.dumps(stats["cap_hits"])


def test_no_cap_raises_at_the_safety_limit(monkeypatch):
    dc = load()
    monkeypatch.setattr(dc, "SAFETY_INTENSITY", 6.0)
    model = dc.build_network(6, "cpu", seed=0)
    black = torch.zeros(784, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="safety limit"):
        dc.present(model, black, 20, max_intensity=None)
    with pytest.raises(RuntimeError, match="safety limit"):
        dc.present(model, black, 20, max_intensity=0)


def test_no_cap_with_members_and_rest_restores_satisfied_members():
    dc = load()
    model = dc.build_network(6, "cpu", seed=0, members=2)
    image = torch.zeros(784, dtype=torch.uint8)
    image[:200] = 255
    # member 1 sees a dim image that needs a higher intensity than member 0's bright one
    dim = (image // 12).to(torch.uint8)
    images = torch.stack([image, dim])
    before = model.syn.weights.clone()
    counts, info = dc.present(model, images, 60, max_intensity=None, rest=150)
    assert (counts.sum(-1) >= dc.MIN_SPIKES).all()
    assert not info["cap_hit"].any()
    assert info["presentations"][1] >= info["presentations"][0]
    assert counts.shape == (2, 6)
    assert (model.syn.weights != before).any()
