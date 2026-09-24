import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons.axon import Axon
from neurosush.structure.column import CorticalColumn
from neurosush.structure.connect import connect
from neurosush.structure.layer import CorticalLayer, Layer
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.traces import SpikeGather


class AlwaysFire(Behavior):
    order = Order.FIRE

    def initialize(self, group):
        group.spikes = group.state(True, dtype=torch.bool)


def groups(net, *names, inhibitory=()):
    return {n: NeuronGroup(net, 2, name=n, inhibitory=n in inhibitory) for n in names}


class TestLayer:
    def test_default_ports_expose_every_group(self):
        net = Network()
        g = groups(net, "a", "b")
        layer = Layer("L", g)
        assert layer.input_port() == (g["a"], g["b"])
        assert layer.output_port() == (g["a"], g["b"])
        assert layer["a"] is g["a"]

    def test_named_ports(self):
        net = Network()
        g = groups(net, "a", "b")
        layer = Layer("L", g, inputs={"x": ["b"]}, outputs={"y": ["a", "b"]})
        assert layer.input_port("x") == (g["b"],)
        assert layer.output_port("y") == (g["a"], g["b"])

    def test_unknown_port(self):
        layer = Layer("L", groups(Network(), "a"))
        with pytest.raises(KeyError, match="nope"):
            layer.input_port("nope")

    def test_port_must_name_existing_groups(self):
        with pytest.raises(ValueError, match="c"):
            Layer("L", groups(Network(), "a"), inputs={"in": ["c"]})

    def test_needs_groups(self):
        with pytest.raises(ValueError, match="group"):
            Layer("L", {})

    def test_groups_are_copied(self):
        net = Network()
        g = groups(net, "a")
        layer = Layer("L", g)
        g["b"] = NeuronGroup(net, 1)
        assert list(layer.groups) == ["a"]


class TestCorticalLayer:
    def test_ports(self):
        net = Network()
        g = groups(net, "e", "i", inhibitory=("i",))
        layer = CorticalLayer("L4", exc=g["e"], inh=g["i"])
        assert layer.exc is g["e"]
        assert layer.inh is g["i"]
        assert layer.input_port() == (g["e"], g["i"])
        assert layer.output_port() == (g["e"],)

    def test_populations_must_have_the_right_sign(self):
        net = Network()
        g = groups(net, "e", "i")
        with pytest.raises(ValueError, match="inhibitory"):
            CorticalLayer("L4", exc=g["e"], inh=g["i"])


def dense_factory():
    return [WeightInit(mode="ones"), DenseInput(), SpikeGather()]


class TestConnect:
    def test_one_synapse_group_per_pair_with_fresh_behaviors(self):
        net = Network()
        g = groups(net, "a", "b", "c")
        synapses = connect(net, [g["a"], g["b"]], [g["c"]], dense_factory, compartment="distal")
        assert [(s.src.name, s.dst.name) for s in synapses] == [("a", "c"), ("b", "c")]
        assert synapses[0].behaviors[0] is not synapses[1].behaviors[0]
        assert all(s.compartment.value == "distal" for s in synapses)

    def test_factory_must_be_callable(self):
        net = Network()
        g = groups(net, "a", "b")
        with pytest.raises(TypeError, match="callable"):
            connect(net, [g["a"]], [g["b"]], [DenseInput()])

    def test_connected_network_runs(self):
        net = Network()
        a, b = NeuronGroup(net, 2, [AlwaysFire(), Axon()]), NeuronGroup(net, 3)
        (syn,) = connect(net, [a], [b], dense_factory)
        net.run(2)
        # spikes gathered in step 1 drive step 2: two sources with weight 1 each
        assert syn.I.tolist() == [2.0, 2.0, 2.0]


class TestCorticalColumn:
    def test_ports_resolve_through_layers(self):
        net = Network()
        l4 = Layer("L4", groups(net, "a4"))
        l23 = Layer("L23", groups(net, "a23"), outputs={"up": ["a23"]})
        column = CorticalColumn(
            "C", {"L4": l4, "L23": l23}, inputs={"in": "L4.in"}, outputs={"out": "L23.up"}
        )
        assert column.input_port("in") == (l4["a4"],)
        assert column.output_port("out") == (l23["a23"],)
        assert column.layers["L4"] is l4

    def test_port_reference_must_exist(self):
        net = Network()
        with pytest.raises(ValueError, match="L5"):
            CorticalColumn("C", {"L4": Layer("L4", groups(net, "a"))}, inputs={"in": "L5.in"})
        with pytest.raises(ValueError, match=r"layer\.port"):
            CorticalColumn("C", {"L4": Layer("L4", groups(net, "b"))}, inputs={"in": "L4"})
