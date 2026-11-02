import copy
import json

import pytest
import torch

from neurosush.spatial import Walk
from neurosush.tem import TEMConfig, evaluate, train
from neurosush.tem.train import _additional_controls


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    return TEMConfig(
        sensory=4,
        compressed=4,
        modules=(3, 2),
        hidden=5,
        batch=2,
        bptt=10,
        walk_len=40,
        widths=(2,),
        lr_max=0.01,
        decay_ramp=2,
        rate_ramp=4,
        latent_ramp=4,
    )


def test_tiny_training_loss_decreases():
    result = train(tiny_config(), steps=400, seed=7)
    early = sum(row["loss"] for row in result.history[:4]) / 4
    late = sum(row["loss"] for row in result.history[-4:]) / 4
    assert late < early
    assert len(result.history) == 40
    assert result.history[0]["memory_rate"] == 0
    assert result.history[4]["memory_rate"] == 0.5
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in result.history)


def test_training_determinism_and_evaluation_preserves_weights_rng_and_mode():
    config = tiny_config()
    rng = torch.random.get_rng_state().clone()
    first = train(config, steps=40, seed=3)
    second = train(config, steps=40, seed=3)
    assert first.history == second.history
    before = copy.deepcopy(first.model.state_dict())
    for name, value in before.items():
        assert torch.equal(value, second.model.state_dict()[name])
    first.model.train()
    a = evaluate(first.model, config, environments=2, steps=80, seed=1234)
    b = evaluate(first.model, config, environments=2, steps=80, seed=1234)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert first.model.training
    assert a["events"] > 0
    assert sum(row["events"] for row in a["by_fraction"].values()) == a["events"]
    assert all(torch.equal(value, first.model.state_dict()[name]) for name, value in before.items())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_online_extra_controls_hand_values_no_future_access():
    walk = Walk(
        torch.tensor([0, 1, 0, 2, 0]),
        torch.tensor([1, 2, 1, 0, 1]),
        torch.tensor([0, 1, 0, 1]),
        torch.zeros(4, dtype=torch.bool),
        torch.zeros(4),
    )
    result = _additional_controls(walk, sensory=3, size=3)
    # At t=2, both controls recall the completed 0->1 transition's label 2.
    assert result["sensory_only"].tolist() == [1, 1, 2, 1]
    assert result["node_conditional"].tolist() == [1, 1, 2, 1]
    changed = Walk(
        walk.nodes,
        torch.tensor([1, 2, 1, 2, 2]),
        walk.actions,
        walk.novel_edge_to_seen,
        walk.fraction_seen,
    )
    altered = _additional_controls(changed, sensory=3, size=3)
    assert torch.equal(result["sensory_only"][:3], altered["sensory_only"][:3])
    assert torch.equal(result["node_conditional"][:3], altered["node_conditional"][:3])


def test_oracle_position_memory_generalises_to_new_labels():
    config = TEMConfig(
        sensory=20,
        compressed=10,
        modules=(36,),
        hidden=20,
        batch=1,
        bptt=10,
        walk_len=400,
        widths=(4,),
        lr_max=0.03,
        lr_min=0.003,
        decay_ramp=1,
        rate_ramp=1,
        latent_ramp=1,
    )
    result = train(config, steps=400, seed=0, oracle=True)
    scores = evaluate(result.model, config, environments=4, steps=400, seed=1234, oracle=True)
    assert scores["events"] >= 100
    assert scores["accuracy"] > scores["frequency"] + 0.10
    assert scores["position_r2"] > scores["shift_null_p95"]


def test_invalid_training_and_evaluation_arguments():
    config = tiny_config()
    with pytest.raises(ValueError, match="0"):
        train(config, steps=0, seed=0)
    result = train(config, steps=10, seed=0)
    with pytest.raises(ValueError, match="9"):
        evaluate(result.model, config, environments=1, steps=9, seed=0)
    with pytest.raises(ValueError, match="lattice size"):
        train(config, steps=10, seed=0, oracle=True)
