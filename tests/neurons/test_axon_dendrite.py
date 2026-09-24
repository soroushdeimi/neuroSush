import math

import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Compartment, Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure, modulatory_drive
from neurosush.neurons.models import LIF


class ScriptedCurrent(Behavior):
    """Sets syn.I from a list, one entry per step (then zeros)."""

    order = Order.SYNAPTIC_INPUT

    def __init__(self, values):
        self.values = list(values)

    def initialize(self, syn):
        syn.I = torch.zeros(syn.dst.size)

    def forward(self, syn):
        step = syn.net.iteration - 1
        value = self.values[step] if step < len(self.values) else 0.0
        syn.I = torch.full((syn.dst.size,), float(value))


class ScriptedSpikes(Behavior):
    order = Order.FIRE

    def __init__(self, frames):
        self.frames = [torch.tensor(f) for f in frames]

    def initialize(self, group):
        group.spikes = group.vector(False, dtype=torch.bool)

    def forward(self, group):
        step = group.net.iteration - 1
        group.spikes = (
            self.frames[step] if step < len(self.frames) else group.vector(False, dtype=torch.bool)
        )


class TestAxon:
    def test_records_spike_history(self):
        net = Network()
        ng = NeuronGroup(
            net, 2, behaviors=[ScriptedSpikes([[True, False], [False, True]]), Axon(max_delay=3)]
        )
        net.run(2)
        assert ng.spike_history.read(0).tolist() == [False, True]
        assert ng.spike_history.read(1).tolist() == [True, False]
        assert ng.spike_history.depth == 3

    def test_invalid_max_delay(self):
        with pytest.raises(ValueError, match="max_delay"):
            Axon(max_delay=0)

    def test_efferent_delays_must_fit(self):
        net = Network()
        a = NeuronGroup(net, 2, behaviors=[ScriptedSpikes([]), Axon(max_delay=2)])
        b = NeuronGroup(net, 1)
        syn = SynapseGroup(net, a, b)
        syn.src_delay = torch.tensor([0, 2])
        with pytest.raises(ValueError, match=syn.name):
            net.initialize()

    def test_afferent_delays_must_fit(self):
        # dst_delay reads the destination's own history, so it has to fit that Axon too
        net = Network()
        a = NeuronGroup(net, 1)
        b = NeuronGroup(net, 2, behaviors=[ScriptedSpikes([]), Axon(max_delay=2)])
        syn = SynapseGroup(net, a, b)
        syn.dst_delay = torch.tensor([2, 0])
        with pytest.raises(
            ValueError, match=r"dst_delay must be less than max_delay=2 for sg0, got 2"
        ):
            net.initialize()


def dendrite_net(values, delay=0, compartment="proximal", depth=2):
    net = Network()
    src = NeuronGroup(net, 1)
    dst = NeuronGroup(net, 2, behaviors=[DendriteStructure(**{f"{compartment}_depth": depth})])
    syn = SynapseGroup(net, src, dst, behaviors=[ScriptedCurrent(values)], compartment=compartment)
    syn.dst_delay = torch.full((2,), delay)
    return net, dst


class TestDendriteStructure:
    def test_zero_delay_arrives_in_the_same_step(self):
        net, dst = dendrite_net([1.5])
        net.step()
        assert dst.I_proximal.tolist() == [1.5, 1.5]
        assert dst.I_distal.tolist() == [0.0, 0.0]
        assert dst.I_apical.tolist() == [0.0, 0.0]

    def test_delayed_current(self):
        net, dst = dendrite_net([2.0], delay=2, compartment="distal", depth=3)
        seen = []
        for _ in range(4):
            net.step()
            seen.append(dst.I_distal[0].item())
        assert seen == [0.0, 0.0, 2.0, 0.0]

    def test_currents_of_several_synapses_add_up(self):
        net = Network()
        src = NeuronGroup(net, 1)
        dst = NeuronGroup(net, 1, behaviors=[DendriteStructure()])
        for value in (1.0, 2.5):
            SynapseGroup(net, src, dst, behaviors=[ScriptedCurrent([value])], compartment="apical")
        net.step()
        assert dst.I_apical.tolist() == [3.5]

    def test_delay_beyond_depth_is_rejected(self):
        net, _ = dendrite_net([1.0], delay=2, depth=2)
        with pytest.raises(ValueError, match="dst_delay"):
            net.initialize()

    def test_invalid_depth(self):
        with pytest.raises(ValueError, match="apical_depth"):
            DendriteStructure(apical_depth=0)


def test_modulatory_drive():
    v = torch.tensor([-65.0, -55.0, -60.0])
    current = torch.tensor([100.0, 100.0, 0.0])
    # limit = v_rest + gain * (threshold - v_rest) = -65 + 0.5 * 20 = -55
    out = modulatory_drive(current, v, v_rest=-65.0, threshold=-45.0, gain=0.5)
    assert out.tolist() == pytest.approx([10.0 * math.tanh(100.0), 0.0, 0.0])


class TestDendriteIntegration:
    def make(self, **kwargs):
        net = Network()
        ng = NeuronGroup(
            net,
            1,
            behaviors=[
                DendriteStructure(),
                DendriteIntegration(**kwargs),
                LIF(tau=10.0, threshold=-45.0, v_reset=-70.0, v_rest=-65.0, resistance=2.0),
            ],
        )
        net.initialize()
        return ng

    def test_proximal_current_passes_through(self):
        ng = self.make()
        integ = ng.behaviors[1]
        ng.I_proximal, ng.I_distal, ng.I_apical = (torch.tensor([x]) for x in (3.0, 0.0, 0.0))
        integ.forward(ng)
        assert ng.I.tolist() == [3.0]

    def test_current_resets_every_step_without_tau(self):
        ng = self.make()
        ng.I = torch.tensor([5.0])
        ng.I_proximal, ng.I_distal, ng.I_apical = (torch.zeros(1) for _ in range(3))
        ng.behaviors[1].forward(ng)
        assert ng.I.tolist() == [0.0]

    def test_current_decays_with_tau(self):
        ng = self.make(tau_current=4.0)
        ng.I = torch.tensor([8.0])
        ng.I_proximal, ng.I_distal, ng.I_apical = (torch.zeros(1) for _ in range(3))
        ng.behaviors[1].forward(ng)
        # I *= 1 - dt / tau_current
        assert ng.I.tolist() == [6.0]

    def test_distal_priming(self):
        ng = self.make(distal_gain=0.5)
        ng.v = torch.tensor([-65.0])
        ng.I_proximal, ng.I_distal, ng.I_apical = (torch.tensor([x]) for x in (0.0, 50.0, 0.0))
        ng.behaviors[1].forward(ng)
        # I = tau / R * drive, so the next LIF step moves v by dt * drive toward the limit -55
        drive = 10.0 * math.tanh(50.0)
        assert ng.I.tolist() == pytest.approx([10.0 / 2.0 * drive])

    def test_gains_are_ignored_when_not_set(self):
        ng = self.make()
        ng.I_proximal, ng.I_distal, ng.I_apical = (torch.tensor([x]) for x in (0.0, 50.0, 50.0))
        ng.behaviors[1].forward(ng)
        assert ng.I.tolist() == [0.0]

    def test_needs_the_dendrite_structure(self):
        net = Network()
        NeuronGroup(net, 1, behaviors=[DendriteIntegration()])
        with pytest.raises(RuntimeError, match="DendriteStructure"):
            net.initialize()

    def test_invalid_tau(self):
        with pytest.raises(ValueError, match="tau_current"):
            DendriteIntegration(tau_current=0.0)


def test_compartment_names_match_the_enum():
    assert {c.value for c in Compartment} == {"proximal", "distal", "apical"}
