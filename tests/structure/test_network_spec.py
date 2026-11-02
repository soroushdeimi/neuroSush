import json
from pathlib import Path

import pytest
import torch

from neurosush.checkpoint import state_dict
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.structure.network_spec import (
    NetSettings,
    NetworkSpec,
    RecordingSpec,
    build_network,
    dump_spec,
    load_spec,
)
from neurosush.structure.spec import _REGISTRY, ColumnSpec, GroupSpec, LayerSpec, to_json
from neurosush.viz.schedule import schedule_table
from neurosush.viz.trace import TraceRecorder, pack_array


def behavior(kind, **params):
    defaults = {
        "LIF": {"tau": 10.0, "threshold": -50.0, "v_reset": -70.0, "v_rest": -65.0},
        "Traces": {"tau_pre": 20.0, "tau_post": 20.0},
        "STDP": {"a_plus": 0.01, "a_minus": 0.012},
    }
    return {"kind": kind, "params": {**defaults.get(kind, {}), **params}}


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


def test_v1_loads_unchanged():
    column = ColumnSpec({"L4": LayerSpec({"exc": GroupSpec(3)})})
    assert load_spec(to_json(column)) == column
    assert load_spec(dump_spec(column)) == column
    assert build_network(NetworkSpec(columns={"c1": column})).groups["c1.L4.exc"].size == 3


@pytest.mark.parametrize("path", sorted(Path("tests/golden/spec_v2").glob("*.json")))
def test_canonical_golden(path):
    text = path.read_text()
    assert dump_spec(load_spec(text)) == text


@pytest.mark.parametrize("with_poisson", [False, True])
def test_ei_stdp_matches_manual_build(with_poisson):
    if with_poisson and "PoissonInput" not in _REGISTRY:
        pytest.skip("PoissonInput registration belongs to G1")
    kinds = ["LIF", "Fire", "Axon", "DendriteStructure", "DendriteIntegration"]
    group_behaviors = [behavior(kind) for kind in kinds]
    syn_behaviors = [
        behavior("WeightInit", mode="uniform"),
        behavior("DenseInput"),
        behavior("SpikeGather"),
        behavior("Traces"),
        behavior("STDP"),
    ]
    groups = {
        "exc": {"shape": 4, "behaviors": group_behaviors},
        "inh": {"shape": 2, "behaviors": group_behaviors, "inhibitory": True},
    }
    spec = NetworkSpec(
        network=NetSettings(seed=7),
        groups=groups,
        synapses=({"src": "exc", "dst": "inh", "behaviors": syn_behaviors},),
        recordings=(RecordingSpec("exc", ("v", "spikes"), max_steps=50),),
    )
    if with_poisson:
        groups["input"] = {
            "shape": 4,
            "behaviors": [
                behavior("PoissonInput", rates={"$tensor": pack_array(torch.full((4,), 0.3))}),
                behavior("Axon"),
            ],
        }
        spec = NetworkSpec(
            network=spec.network,
            groups=groups,
            synapses=(*spec.synapses, {"src": "input", "dst": "exc", "behaviors": syn_behaviors}),
            recordings=spec.recordings,
        )
    built = build_network(spec)
    net = Network(seed=7)
    rec = TraceRecorder("v", "spikes", max_steps=50)
    exc = NeuronGroup(
        net, 4, [_REGISTRY[k](**behavior(k)["params"]) for k in kinds] + [rec], name="exc"
    )
    inh = NeuronGroup(
        net, 2, [_REGISTRY[k](**behavior(k)["params"]) for k in kinds], name="inh", inhibitory=True
    )
    SynapseGroup(
        net, exc, inh, [_REGISTRY[b["kind"]](**b["params"]) for b in syn_behaviors], name="exc->inh"
    )
    if with_poisson:
        source = NeuronGroup(
            net,
            4,
            [_REGISTRY["PoissonInput"](rates=torch.full((4,), 0.3)), _REGISTRY["Axon"]()],
            name="input",
        )
        SynapseGroup(
            net,
            source,
            exc,
            [_REGISTRY[b["kind"]](**b["params"]) for b in syn_behaviors],
            name="input->exc",
        )
    net.initialize()
    assert schedule_table(built.net) == schedule_table(net)
    built.net.run(50)
    net.run(50)
    assert_equal(state_dict(built.net), state_dict(net))
    assert torch.equal(built.recorders["exc"].trace().series["v"], rec.trace().series["v"])


def test_poisson_tensor_matches_manual_when_registered():
    if "PoissonInput" not in _REGISTRY:
        pytest.skip("PoissonInput registration belongs to G1")
    rates = torch.tensor([0.1, 0.3])
    spec = NetworkSpec(
        network=NetSettings(seed=7),
        groups={
            "input": {
                "shape": 2,
                "behaviors": [behavior("PoissonInput", rates={"$tensor": pack_array(rates)})],
            }
        },
        recordings=(RecordingSpec("input", ("spikes",), max_steps=50),),
    )
    built = build_network(spec)
    net = Network(seed=7)
    recorder = TraceRecorder("spikes", max_steps=50)
    NeuronGroup(net, 2, [_REGISTRY["PoissonInput"](rates=rates), recorder], name="input")
    net.initialize()
    assert schedule_table(built.net) == schedule_table(net)
    built.net.run(50)
    net.run(50)
    assert_equal(state_dict(built.net), state_dict(net))
    assert torch.equal(
        built.recorders["input"].trace().series["spikes"], recorder.trace().series["spikes"]
    )


def test_assets_enabled_and_recording_targets():
    spec = load_spec(
        json.dumps(
            {
                "version": 2,
                "network": {"seed": 7},
                "groups": {
                    "input": {
                        "shape": 2,
                        "behaviors": [behavior("SpikeInput", frames={"$asset": "frames"})],
                    },
                    "exc": {
                        "shape": 2,
                        "behaviors": [behavior("LIF"), {"kind": "Fire", "enabled": False}],
                    },
                },
                "synapses": [
                    {
                        "src": "input",
                        "dst": "exc",
                        "tags": ["test"],
                        "behaviors": [
                            behavior(
                                "WeightInit",
                                weights={"$init": "constant", "shape": [2, 2], "value": 1},
                            )
                        ],
                    }
                ],
                "recordings": [
                    {"target": "network", "attributes": ["iteration"]},
                    {"target": "input->exc", "attributes": ["weights"]},
                    {"target": "input", "attributes": ["spikes"], "interval": 2},
                ],
            }
        )
    )
    assert isinstance(spec, NetworkSpec)
    built = build_network(spec, {"frames": [torch.tensor([True, False])] * 2})
    assert built.groups["exc"].behaviors[1].enabled is False
    assert built.synapses["input->exc"].tags == {"test"}
    built.net.run(2)
    assert built.recorders["input"].trace().steps == [2]
    assert built.recorders["network"].trace().steps == [1, 2]
    with pytest.raises(RuntimeError, match="ran out of frames"):
        built.net.step()


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"synapses": ({"src": "missing", "dst": "exc"},)}, "unknown group"),
        ({"recordings": (RecordingSpec("missing", ("v",)),)}, "unknown recording"),
        ({"recordings": (RecordingSpec("exc", ("v",)),) * 2}, "duplicate recording"),
        ({"synapses": ({"src": "exc", "dst": "exc"},) * 2}, "duplicate"),
        ({"groups": {"exc": {"shape": 2, "behaviors": [behavior("unknown")]}}}, "unknown behavior"),
    ],
)
def test_build_failures(changes, match):
    options = {"groups": {"exc": {"shape": 2, "behaviors": [behavior("LIF")]}}}
    options.update(changes)
    with pytest.raises(ValueError, match=match):
        build_network(NetworkSpec(**options))


@pytest.mark.parametrize(
    "text", ['{"version": 3}', '{"version": 2, "network": {"dtype": "int64"}}']
)
def test_invalid_version_and_dtype(text):
    with pytest.raises(ValueError, match=r"/version|/network/dtype"):
        load_spec(text)
