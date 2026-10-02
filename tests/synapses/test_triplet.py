import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.graph import GraphStepper
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order
from neurosush.neurons.axon import Axon
from neurosush.neurons.inputs import PoissonInput
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.traces import SpikeGather
from neurosush.synapses.triplet import TripletSTDP, trace_decay, trace_increment


class ScriptedSpikes(Behavior):
    """Fires the neurons listed for each step (1-based)."""

    order = Order.FIRE

    def __init__(self, schedule):
        self.schedule = schedule

    def initialize(self, group):
        group.spikes = group.state(False, dtype=torch.bool)

    def forward(self, group):
        spikes = group.state(False, dtype=torch.bool)
        spikes[..., self.schedule.get(group.net.iteration, [])] = True
        group.spikes = spikes


ARGS = {
    "a2_plus": 0.005,
    "a3_plus": 0.006,
    "a2_minus": 0.007,
    "a3_minus": 0.0023,
    "tau_plus": 16.8,
    "tau_minus": 33.7,
    "tau_x": 101.0,
    "tau_y": 125.0,
}


def init(net, src, dst, behaviors):
    syn = SynapseGroup(net, src, dst, behaviors)
    net.initialize()
    return syn


def make(behaviors=None, dt=1.0, **kwargs):
    net = Network(dt=dt, dtype=torch.float64)
    src = NeuronGroup(net, 2, [ScriptedSpikes({1: [0]}), Axon()])
    dst = NeuronGroup(net, 3, [ScriptedSpikes({2: [1]}), Axon()])
    rule = TripletSTDP(**{**ARGS, **kwargs})
    behaviors = behaviors or [WeightInit(mode=0.5), DenseInput(), SpikeGather(), rule]
    syn = SynapseGroup(net, src, dst, [*behaviors[:-1], rule])
    net.initialize()
    return net, syn, rule


class TestArguments:
    @pytest.mark.parametrize("name", ["a2_plus", "a3_plus", "a2_minus", "a3_minus"])
    def test_negative_amplitude_raises(self, name):
        with pytest.raises(ValueError, match=name):
            TripletSTDP(**{**ARGS, name: -0.1})

    @pytest.mark.parametrize("name", ["tau_plus", "tau_minus", "tau_x", "tau_y"])
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_nonpositive_tau_raises(self, name, value):
        with pytest.raises(ValueError, match=name):
            TripletSTDP(**{**ARGS, name: value})

    def test_unknown_interaction_raises(self):
        with pytest.raises(ValueError, match="interaction"):
            TripletSTDP(**ARGS, interaction="some")

    def test_bad_bounds_raise(self):
        with pytest.raises(ValueError, match="w_min"):
            TripletSTDP(**ARGS, w_min=1.0, w_max=1.0)
        with pytest.raises(ValueError, match="bound"):
            TripletSTDP(**ARGS, bound="sharp")  # type: ignore[arg-type]

    def test_tau_below_dt_raises_at_initialize(self):
        with pytest.raises(ValueError, match="tau_plus"):
            make(dt=20.0, tau_plus=10.0)

    def test_missing_spike_gather_raises(self):
        net = Network(dtype=torch.float64)
        src = NeuronGroup(net, 2, [Axon()])
        dst = NeuronGroup(net, 2, [Axon()])
        with pytest.raises(RuntimeError, match="TripletSTDP"):
            TripletSTDP(**ARGS).initialize(SynapseGroup(net, src, dst, []))

    def test_destination_without_axon_raises(self):
        net = Network(dtype=torch.float64)
        src = NeuronGroup(net, 2, [Axon()])
        dst = NeuronGroup(net, 2, [])
        with pytest.raises(RuntimeError, match="needs"):
            init(
                net,
                src,
                dst,
                [WeightInit(mode=0.5), DenseInput(), SpikeGather(), TripletSTDP(**ARGS)],
            )

    def test_unsupported_connectivity_raises(self):
        from neurosush.synapses.currents import SparseInput

        net = Network(dtype=torch.float64)
        src = NeuronGroup(net, 2, [Axon()])
        dst = NeuronGroup(net, 2, [Axon()])
        with pytest.raises(ValueError, match="connectivity"):
            init(
                net,
                src,
                dst,
                [
                    WeightInit(mode=0.5, sparse=True, density=0.5),
                    SparseInput(),
                    SpikeGather(),
                    TripletSTDP(**ARGS),
                ],
            )


class TestTraceUpdates:
    def test_decay_is_euler(self):
        out = trace_decay(torch.tensor([1.0, 2.0]), tau=4.0, dt=1.0)
        assert out.tolist() == [0.75, 1.5]

    def test_all_adds_and_nearest_resets(self):
        decayed = torch.tensor([0.75, 1.5, 0.25])
        spikes = torch.tensor([True, True, False])
        assert trace_increment(decayed, spikes, interaction="all").tolist() == [1.75, 2.5, 0.25]
        assert trace_increment(decayed, spikes, interaction="nearest").tolist() == [1.0, 1.0, 0.25]

    def test_behavior_traces_after_two_spikes(self):
        net = Network(dtype=torch.float64)
        src = NeuronGroup(net, 1, [ScriptedSpikes({1: [0], 2: [0]}), Axon()])
        dst = NeuronGroup(net, 1, [ScriptedSpikes({}), Axon()])
        for interaction, expected in (("all", 1 + 0.75), ("nearest", 1.0)):
            net = Network(dtype=torch.float64)
            src = NeuronGroup(net, 1, [ScriptedSpikes({1: [0], 2: [0]}), Axon()])
            dst = NeuronGroup(net, 1, [ScriptedSpikes({}), Axon()])
            rule = TripletSTDP(**{**ARGS, "tau_plus": 4.0}, interaction=interaction)
            SynapseGroup(net, src, dst, [WeightInit(mode=0.5), DenseInput(), SpikeGather(), rule])
            net.run(2)
            assert rule.r1.item() == pytest.approx(expected)


class TestState:
    def run_and_get(self):
        net, syn, rule = make()
        net.run(3)
        return net, syn, rule

    def test_state_dict_round_trip(self):
        _, _, rule = self.run_and_get()
        saved = rule.state_dict()
        assert set(saved) == {"r1", "r2", "o1", "o2"}
        assert saved["r1"].abs().sum() > 0
        _, _, other = make()
        other.load_state_dict(saved)
        for key, value in saved.items():
            assert torch.equal(getattr(other, key), value)

    def test_state_dict_is_a_copy(self):
        _, _, rule = self.run_and_get()
        saved = rule.state_dict()
        rule.reset_state(None)
        assert saved["r1"].abs().sum() > 0

    def test_load_rejects_bad_keys_and_shapes(self):
        _, _, rule = self.run_and_get()
        saved = rule.state_dict()
        with pytest.raises(KeyError):
            rule.load_state_dict({"r1": saved["r1"]})
        with pytest.raises(ValueError, match="shape"):
            rule.load_state_dict({**saved, "o1": torch.zeros(7)})

    def test_reset_state_zeroes_in_place(self):
        _, syn, rule = self.run_and_get()
        r1 = rule.r1
        assert r1.abs().sum() > 0
        rule.reset_state(syn)
        assert rule.r1 is r1
        for trace in (rule.r1, rule.r2, rule.o1, rule.o2):
            assert not trace.any()


@pytest.mark.gpu
def test_graph_stepper_matches_eager_bit_for_bit():
    def build():
        net = Network(device="cuda", seed=1)
        src = NeuronGroup(net, 8, [PoissonInput(0.3), Axon()], name="src")
        dst = NeuronGroup(net, 5, [PoissonInput(0.2), Axon()], name="dst")
        syn = SynapseGroup(
            net,
            src,
            dst,
            [
                WeightInit(mode=0.5),
                DenseInput(),
                SpikeGather(),
                TripletSTDP(**ARGS, interaction="nearest", bound="soft"),
            ],
        )
        return net, syn

    eager, eager_syn = build()
    eager.run(50)
    graphed, graphed_syn = build()
    GraphStepper(graphed).run(50)
    assert torch.equal(eager_syn.weights, graphed_syn.weights)
