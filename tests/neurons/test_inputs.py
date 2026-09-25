import itertools

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.data import spike_frames
from neurosush.neurons.axon import Axon
from neurosush.neurons.inputs import SpikeInput
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.traces import SpikeGather


def run_input(frames, size=2, steps=1):
    net = Network()
    group = NeuronGroup(net, size, behaviors=[SpikeInput(frames)])
    net.run(steps)
    return group


class TestSpikeInput:
    def test_plain_frames(self):
        group = run_input([torch.tensor([True, False]), torch.tensor([False, True])], steps=2)
        assert group.spikes.tolist() == [False, True]
        assert group.label is None

    def test_labelled_frames_from_spike_frames(self):
        trains = [(torch.tensor([[1, 0], [1, 1]]), "cat")]
        group = run_input(spike_frames(trains), steps=2)
        assert group.spikes.tolist() == [True, True]
        assert group.label == "cat"

    def test_frames_are_flattened_and_converted(self):
        group = run_input([torch.tensor([[1.0, 0.0], [0.0, 1.0]])], size=4)
        assert group.spikes.dtype == torch.bool
        assert group.spikes.tolist() == [True, False, False, True]

    def test_initial_state_is_silent(self):
        net = Network()
        group = NeuronGroup(net, 3, behaviors=[SpikeInput([])])
        net.initialize()
        assert group.spikes.tolist() == [False, False, False]
        assert group.label is None

    def test_running_out_of_frames_is_an_error(self):
        with pytest.raises(RuntimeError, match="ran out of frames at iteration 2"):
            run_input([torch.tensor([True, True])], steps=2)

    def test_frame_size_must_match(self):
        with pytest.raises(ValueError, match="3"):
            run_input([torch.tensor([True, False, True])])

    def test_cycle_for_endless_input(self):
        group = run_input(itertools.cycle([torch.tensor([True, False])]), steps=5)
        assert group.spikes.tolist() == [True, False]

    def test_published_spikes_are_not_aliased_to_the_staged_frame(self):
        frames = [torch.tensor([True, False]), torch.tensor([False, True])]
        net = Network()
        group = NeuronGroup(net, 2, behaviors=[SpikeInput(frames)])
        net.step()
        first = group.spikes
        net.step()  # stages the next frame into the same tensor prepare wrote before
        assert first.tolist() == [True, False]
        assert group.spikes.tolist() == [False, True]

    def test_drives_a_synapse(self):
        net = Network()
        src = NeuronGroup(
            net, 2, behaviors=[SpikeInput(itertools.repeat(torch.tensor([True, True]))), Axon()]
        )
        dst = NeuronGroup(net, 1, behaviors=[Axon()])
        dst.spikes = dst.vector(False, dtype=torch.bool)
        syn = SynapseGroup(
            net, src, dst, behaviors=[WeightInit(mode=0.5), DenseInput(), SpikeGather()]
        )
        net.run(2)
        # step 1 gathers the spikes; step 2 turns them into current
        assert syn.I.tolist() == [1.0]
