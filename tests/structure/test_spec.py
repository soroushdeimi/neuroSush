import json

import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network
from neurosush.core.order import Order
from neurosush.structure.spec import (
    BehaviorSpec,
    ColumnSpec,
    GroupSpec,
    LayerSpec,
    SynapseSpec,
    build_column,
    from_json,
    register,
    registered,
    to_json,
)

LIF = BehaviorSpec("LIF", {"tau": 10.0, "threshold": -50.0, "v_reset": -70.0, "v_rest": -65.0})


def column_spec():
    exc = GroupSpec(4, (LIF, BehaviorSpec("Fire"), BehaviorSpec("Axon")))
    inh = GroupSpec(2, (LIF, BehaviorSpec("Fire"), BehaviorSpec("Axon")), inhibitory=True)
    dense = (
        BehaviorSpec("WeightInit", {"mode": "uniform"}),
        BehaviorSpec("DenseInput"),
        BehaviorSpec("SpikeGather"),
    )
    layer = LayerSpec(
        groups={"exc": exc, "inh": inh},
        synapses=(SynapseSpec("exc", "inh", dense), SynapseSpec("inh", "exc", dense)),
        outputs={"out": ("exc",)},
    )
    return ColumnSpec(
        layers={"L4": layer, "L23": layer},
        synapses=(SynapseSpec("L4.exc", "L23.exc", dense, compartment="distal"),),
        inputs={"in": "L4.in"},
        outputs={"out": "L23.out"},
    )


class TestBuild:
    def test_builds_groups_synapses_and_ports(self):
        net = Network(seed=0)
        column = build_column(net, "C1", column_spec())
        names = [g.name for g in net.groups]
        assert names == ["C1.L4.exc", "C1.L4.inh", "C1.L23.exc", "C1.L23.inh"]
        assert len(net.synapses) == 5
        assert net.groups[1].inhibitory is True
        assert column.output_port("out") == (net.groups[2],)
        assert column.input_port("in") == (net.groups[0], net.groups[1])
        assert net.synapses[-1].compartment.value == "distal"

    def test_every_object_gets_its_own_behaviors(self):
        net = Network()
        build_column(net, "C1", column_spec())
        behaviors = [b for g in net.groups for b in g.behaviors]
        assert len({id(b) for b in behaviors}) == len(behaviors)

    def test_built_network_runs(self):
        net = Network(seed=0)
        build_column(net, "C1", column_spec())
        net.run(3)
        assert net.iteration == 3

    def test_two_columns_from_one_spec(self):
        net = Network()
        build_column(net, "A", column_spec())
        build_column(net, "B", column_spec())
        assert len(net.groups) == 8

    def test_unknown_group_in_synapse(self):
        spec = LayerSpec(groups={"a": GroupSpec(1)}, synapses=(SynapseSpec("a", "b", ()),))
        with pytest.raises(ValueError, match="b"):
            build_column(Network(), "C", ColumnSpec(layers={"L": spec}))

    def test_unknown_behavior(self):
        with pytest.raises(ValueError, match="Nope"):
            BehaviorSpec("Nope").build()

    def test_bad_parameters_name_the_behavior(self):
        with pytest.raises(TypeError, match="LIF"):
            BehaviorSpec("LIF", {"tau": 1.0}).build()


class TestJson:
    def test_round_trip(self):
        spec = column_spec()
        text = to_json(spec)
        assert json.loads(text)["layers"]["L4"]["groups"]["exc"]["shape"] == 4
        assert from_json(text) == spec

    def test_tuple_shapes_survive(self):
        spec = ColumnSpec(layers={"L": LayerSpec(groups={"g": GroupSpec((2, 3, 3))})})
        assert from_json(to_json(spec)) == spec

    def test_tensor_parameters_are_rejected(self):
        spec = ColumnSpec(
            layers={
                "L": LayerSpec(
                    groups={
                        "g": GroupSpec(1, (BehaviorSpec("WeightInit", {"weights": torch.ones(1)}),))
                    }
                )
            }
        )
        with pytest.raises(ValueError, match="JSON"):
            to_json(spec)

    def test_loading_never_executes_code(self):
        text = to_json(
            ColumnSpec(layers={"L": LayerSpec(groups={"g": GroupSpec(1, (BehaviorSpec("Fire"),))})})
        )
        evil = text.replace('"Fire"', "\"__import__('os').system('false')\"")
        spec = from_json(evil)
        with pytest.raises(ValueError, match="unknown behavior"):
            spec.layers["L"].groups["g"].behaviors[0].build()

    @pytest.mark.parametrize("shape", [[2, 3], [1, 2, 3, 4]])
    def test_malformed_shape_is_a_value_error(self, shape):
        data = json.loads(to_json(ColumnSpec(layers={"L": LayerSpec(groups={"g": GroupSpec(1)})})))
        data["layers"]["L"]["groups"]["g"]["shape"] = shape
        with pytest.raises(ValueError, match="three ints"):
            from_json(json.dumps(data))


class TestRegistry:
    def test_library_behaviors_are_registered(self):
        names = registered()
        for name in (
            "LIF",
            "ELIF",
            "AdaptiveELIF",
            "Fire",
            "KWTA",
            "Axon",
            "DendriteStructure",
            "WeightInit",
            "DenseInput",
            "Conv2dInput",
            "SpikeGather",
            "Traces",
            "STDP",
            "RSTDP",
            "ISTDP",
            "WeightClip",
            "Dopamine",
        ):
            assert name in names

    def test_register_a_custom_behavior(self):
        @register
        class Custom(Behavior):
            order = Order.FIRE

            def __init__(self, *, gain=1.0):
                self.gain = gain

        assert isinstance(BehaviorSpec("Custom", {"gain": 2.0}).build(), Custom)

    def test_name_clash_is_rejected(self):
        class LIF(Behavior):
            order = Order.FIRE

        with pytest.raises(ValueError, match="LIF"):
            register(LIF)
