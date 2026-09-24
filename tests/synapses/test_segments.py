import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order
from neurosush.neurons.axon import Axon
from neurosush.structure.spec import registered
from neurosush.synapses.segments import ActiveSegments, plateau_step
from neurosush.synapses.traces import SpikeGather


class Volleys(Behavior):
    """Fires the listed neurons at the given steps: ``{step: [neuron, ...]}``."""

    order = Order.FIRE

    def __init__(self, schedule):
        self.schedule = schedule

    def initialize(self, group):
        group.spikes = group.state(False, dtype=torch.bool)

    def forward(self, group):
        spikes = group.state(False, dtype=torch.bool)
        spikes[self.schedule.get(group.net.iteration, [])] = True
        group.spikes = spikes


def build(segments, *, batch_size=None, src=4, dst=2):
    net = Network(batch_size=batch_size)
    pre = NeuronGroup(net, src, [Axon()])
    post = NeuronGroup(net, dst)
    syn = SynapseGroup(net, pre, post, [segments, SpikeGather()], compartment="distal")
    net.initialize()
    return net, syn


def two_segments():
    """Cell 0: segment 0 on inputs (0, 1), segment 1 on (2, 3); cell 1: one segment on (1, 2)."""
    presynaptic = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [-1, -1]]])
    return ActiveSegments(
        segments=2,
        synapses=2,
        activation_threshold=2,
        plateau=3.0,
        amplitude=0.5,
        presynaptic=presynaptic,
        permanence=torch.ones(2, 2, 2),
    )


class TestState:
    def test_allocates_empty_segments_by_default(self):
        _, syn = build(ActiveSegments(segments=3, synapses=4, activation_threshold=2, plateau=5.0))
        assert syn.presynaptic.shape == (2, 3, 4)
        assert (syn.presynaptic == -1).all()
        assert syn.plateau_steps.shape == (2, 3)
        assert syn.connectivity == "segments"

    def test_current_marks_cells_with_a_plateau(self):
        _, syn = build(two_segments())
        syn.pre_spike = torch.tensor([False, True, True, False])  # only cell 1's segment
        syn.input.forward(syn)
        assert syn.active_segments.tolist() == [[False, False], [True, False]]
        assert syn.I.tolist() == [0.0, 0.5]

    def test_batch_samples_are_independent(self):
        _, syn = build(two_segments(), batch_size=2)
        syn.pre_spike = torch.tensor([[True, True, False, False], [False, False, False, False]])
        syn.input.forward(syn)
        assert syn.plateau_steps.shape == (2, 2, 2)
        assert syn.I.tolist() == [[0.5, 0.0], [0.0, 0.0]]

    def test_unconnected_synapses_do_not_count(self):
        segments = two_segments()
        segments.initial = (segments.initial[0], torch.full((2, 2, 2), 0.3))
        _, syn = build(segments)
        syn.pre_spike = torch.ones(4, dtype=torch.bool)
        syn.input.forward(syn)
        assert not syn.active_segments.any()
        assert syn.segment_potential.tolist() == [[2, 2], [2, 0]]

    def test_plateau_step(self):
        remaining = torch.tensor([0, 1, 3])
        active = torch.tensor([True, False, False])
        assert plateau_step(remaining, active, 4).tolist() == [4, 0, 2]

    def test_registered_for_specs(self):
        assert "ActiveSegments" in registered()


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"segments": 0}, "segments and synapses"),
            ({"activation_threshold": 5}, "activation_threshold"),
            ({"plateau": 0.0}, "plateau"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        options = {"segments": 2, "synapses": 4, "activation_threshold": 2, "plateau": 5.0}
        with pytest.raises(ValueError, match=match):
            ActiveSegments(**{**options, **kwargs})

    def test_initial_segments_must_fit_the_groups(self):
        wrong = ActiveSegments(
            segments=2,
            synapses=2,
            activation_threshold=1,
            plateau=1.0,
            presynaptic=torch.zeros(3, 2, 2, dtype=torch.long),
        )
        with pytest.raises(ValueError, match=r"must have shape \(2, 2, 2\)"):
            build(wrong)
        outside = ActiveSegments(
            segments=1,
            synapses=1,
            activation_threshold=1,
            plateau=1.0,
            presynaptic=torch.tensor([[[4]], [[0]]]),
        )
        with pytest.raises(ValueError, match="index 4 sources"):
            build(outside)

    def test_needs_spike_gather(self):
        net = Network()
        pre, post = NeuronGroup(net, 2, [Axon()]), NeuronGroup(net, 2)
        SynapseGroup(
            net,
            pre,
            post,
            [ActiveSegments(segments=1, synapses=1, activation_threshold=1, plateau=1.0)],
        )
        with pytest.raises(RuntimeError, match="SpikeGather"):
            net.initialize()


class TestCoincidenceWindow:
    @pytest.mark.parametrize(("coincidence", "fires_at"), [(None, None), (2.0, None), (3.0, 6)])
    def test_inputs_count_while_inside_the_window(self, coincidence, fires_at):
        # inputs 0-1 fire at step 3 and 2-3 at step 5; the segment reads a spike one step
        # later, so it sees 0-1 from step 4 and 2-3 from step 6: two steps apart
        net = Network()
        pre = NeuronGroup(net, 4, [Volleys({3: [0, 1], 5: [2, 3]}), Axon()])
        post = NeuronGroup(net, 1)
        segments = ActiveSegments(
            segments=1,
            synapses=4,
            activation_threshold=4,
            plateau=2.0,
            coincidence=coincidence,
            presynaptic=torch.arange(4).reshape(1, 1, 4),
            permanence=torch.ones(1, 1, 4),
        )
        syn = SynapseGroup(net, pre, post, [segments, SpikeGather()], compartment="distal")
        first = None
        for _ in range(12):
            net.step()
            if first is None and bool(syn.active_segments.any()):
                first = net.iteration
        assert first == fires_at

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="coincidence"):
            ActiveSegments(
                segments=1, synapses=1, activation_threshold=1, plateau=1.0, coincidence=0.0
            )
