import json
import re

import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup
from neurosush.neurons.models import LIF, Fire
from neurosush.structure.spec import (
    BehaviorSpec,
    ColumnSpec,
    GroupSpec,
    LayerSpec,
    SynapseSpec,
    from_json,
    to_json,
)


def model(threshold: float | torch.Tensor = 1.0) -> LIF:
    return LIF(tau=10, threshold=threshold, v_reset=0, v_rest=0)


@pytest.mark.parametrize("shape", [(2, 1), (1, 2), (3,)])
@pytest.mark.parametrize("batch", [None, 3])
def test_shared_threshold_shape(shape: tuple[int, ...], batch: int | None) -> None:
    net = Network(seed=0, batch_size=batch)
    NeuronGroup(net, 2, [model(torch.ones(shape)), Fire()])
    with pytest.raises(ValueError, match=re.escape(f"got {shape}")) as error:
        net.initialize()
    assert "(2,)" in str(error.value)


@pytest.mark.parametrize("threshold", [1.0, torch.tensor(1.0), torch.tensor([1.0, 2.0])])
def test_valid_threshold_values(threshold: float | torch.Tensor) -> None:
    net = Network(seed=0)
    group = NeuronGroup(net, 2, [model(threshold), Fire()])
    net.initialize()
    group.I.fill_(10)
    net.run(1)
    assert torch.equal(
        group.v,
        torch.tensor(
            [0.0, 1.0] if isinstance(threshold, torch.Tensor) and threshold.ndim else [0.0, 0.0]
        ),
    )


def spec() -> ColumnSpec:
    synapse = SynapseSpec("cells", "cells", (BehaviorSpec("WeightInit"),))
    return ColumnSpec(
        {"L": LayerSpec({"cells": GroupSpec(2, (BehaviorSpec("Fire"),))}, (synapse,))}, (synapse,)
    )


@pytest.mark.parametrize(
    ("path", "key", "hint"),
    [
        ((), "layres", "layers"),
        (("layers", "L"), "gruops", "groups"),
        (("layers", "L", "groups", "cells"), "shpae", "shape"),
        (("layers", "L", "groups", "cells", "behaviors", 0), "knid", "kind"),
        (("synapses", 0), "scr", "src"),
        (("layers", "L", "synapses", 0), "compartmnt", "compartment"),
        (("synapses", 0, "behaviors", 0), "prams", "params"),
        ((), "unrelated", None),
    ],
)
def test_unknown_spec_keys(path: tuple[str | int, ...], key: str, hint: str | None) -> None:
    data = json.loads(to_json(spec()))
    target = data
    location = "$"
    for part in path:
        target = target[part]
        location += f"[{part}]" if isinstance(part, int) else f".{part}"
    target[key] = None
    with pytest.raises(ValueError, match=re.escape(f"{location}.{key}: unknown key")) as error:
        from_json(json.dumps(data))
    if hint:
        assert f"did you mean {hint}?" in str(error.value)


def test_spec_round_trip() -> None:
    assert from_json(to_json(spec())) == spec()


@pytest.mark.parametrize("action", ["step", "run", "initialize"])
def test_failed_initialize_keeps_original_cause(action: str) -> None:
    class Fails(Behavior):
        order = 0
        calls = 0

        def initialize(self, group: NeuronGroup) -> None:
            self.calls += 1
            torch.rand(2, generator=group.net.generator)
            raise ValueError("bad setting 7; use 8")

    failure = Fails()
    net = Network(seed=4)
    NeuronGroup(net, 2, [failure])
    with pytest.raises(ValueError, match="bad setting 7; use 8") as first:
        net.initialize()
    rng = net.generator.get_state().clone()
    retry = (lambda: net.run(1)) if action == "run" else getattr(net, action)
    with pytest.raises(ValueError, match="bad setting 7; use 8") as again:
        retry()
    assert again.value.__cause__ is first.value
    assert failure.calls == 1
    assert net.iteration == 0
    assert torch.equal(rng, net.generator.get_state())


def test_fire_explains_late_model() -> None:
    late = model()
    late.order = Fire.order + 1
    net = Network(seed=0)
    NeuronGroup(net, 2, [Fire(), late], name="cells")
    with pytest.raises(RuntimeError, match="LIF on cells runs after Fire; set order"):
        net.initialize()


def test_successful_initialization_preserves_state_and_rng() -> None:
    class Draw(Behavior):
        order = 0

        def initialize(self, group: NeuronGroup) -> None:
            group.I = torch.rand(group.size, generator=group.net.generator)

    net = Network(seed=17)
    group = NeuronGroup(net, 2, [Draw(), model(), Fire()])
    expected_generator = torch.Generator().manual_seed(17)
    current = torch.rand(2, generator=expected_generator)
    expected = torch.zeros(2)
    for _ in range(3):
        expected = expected + (current - expected) / 10
    net.run(3)
    assert torch.equal(group.v, expected)
    assert torch.equal(net.generator.get_state(), expected_generator.get_state())
    assert not group.spikes.any()
    assert net.iteration == 3
