import itertools

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.data import spike_frames
from neurosush.neurons.axon import Axon
from neurosush.neurons.inputs import (
    CorrelatedPoissonInput,
    PoissonDrive,
    PoissonInput,
    SpikeInput,
)
from neurosush.neurons.models import LIF
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.traces import SpikeGather

LIF_ARGS = {"tau": 10.0, "threshold": -55.0, "v_reset": -70.0, "v_rest": -65.0}


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


class TestPoissonInput:
    def test_negative_rate_is_rejected(self):
        with pytest.raises(ValueError, match=r"rates must be non-negative, got -0\.1"):
            PoissonInput(-0.1)

    def test_negative_rates_in_a_tensor_are_rejected(self):
        with pytest.raises(ValueError, match="rates must be non-negative"):
            PoissonInput(torch.tensor([0.1, -0.2]))

    def test_rates_of_wrong_shape_are_rejected(self):
        net = Network()
        NeuronGroup(net, 3, behaviors=[PoissonInput(torch.zeros(4))])
        with pytest.raises(ValueError, match=r"rates must have shape \(3,\).*got \(4,\)"):
            net.initialize()


class TestPoissonDrive:
    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((0, 1.0, 1.0), "count must be positive, got 0"),
            ((5, -0.1, 1.0), "rate must be non-negative, got -0.1"),
        ],
    )
    def test_invalid_arguments(self, args, match):
        with pytest.raises(ValueError, match=match):
            PoissonDrive(*args)

    def test_adds_whole_jumps_to_the_membrane(self):
        net = Network(dtype=torch.float64, seed=0)
        group = NeuronGroup(net, 50, [LIF(**LIF_ARGS), PoissonDrive(20, 0.1, 0.5)])
        net.initialize()
        net.step()
        steps = (group.v - (-65.0)) / 0.5  # the LIF leaves v at rest without input
        assert torch.allclose(steps, steps.round())
        assert steps.max() > 0

    def test_uses_the_predrawn_events_when_present(self):
        net = Network(dtype=torch.float64, seed=0)
        group = NeuronGroup(net, 5, [LIF(**LIF_ARGS), PoissonDrive(20, 0.1, 0.5)])
        net.initialize()
        drive = group.behaviors[1]
        drive.drawn = {"events": torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)}
        drive.forward(group)
        assert group.v.tolist() == pytest.approx([-65.0, -64.5, -64.0, -63.5, -63.0])

    def test_draw_consumes_the_same_generator_stream_as_eager(self):
        def run(use_draw):
            net = Network(dtype=torch.float64, seed=7)
            group = NeuronGroup(net, 6, [LIF(**LIF_ARGS), PoissonDrive(20, 0.1, 0.5)])
            net.initialize()
            drive = group.behaviors[1]
            if use_draw:
                drive.drawn = drive.draw(group)
            drive.forward(group)
            return group.v

        assert torch.equal(run(True), run(False))

    def test_seed_makes_it_reproducible_and_works_batched_and_independent(self):
        def run(**kwargs):
            net = Network(dtype=torch.float64, seed=3, **kwargs)
            group = NeuronGroup(net, 4, [LIF(**LIF_ARGS), PoissonDrive(20, 0.1, 0.5)])
            net.run(3)
            return group.v

        assert torch.equal(run(), run())
        assert run(batch_size=2, independent=True).shape == (2, 4)
        assert run(batch_size=2).shape == (2, 4)


class TestCorrelatedPoissonInput:
    @pytest.mark.parametrize(
        ("args", "match"),
        [
            ((0.0, 0.5), "rate must be positive, got 0.0"),
            ((0.1, 0.0), r"correlation must be in \(0, 1\], got 0.0"),
            ((0.1, 1.5), r"correlation must be in \(0, 1\], got 1.5"),
        ],
    )
    def test_invalid_arguments(self, args, match):
        with pytest.raises(ValueError, match=match):
            CorrelatedPoissonInput(*args)

    def test_mother_spike_probability_must_not_exceed_one(self):
        net = Network(dt=1.0)
        NeuronGroup(net, 3, [CorrelatedPoissonInput(0.5, 0.25)])
        with pytest.raises(
            ValueError, match=r"rate / correlation \* dt must be at most 1, got 2\.0"
        ):
            net.initialize()

    def test_spikes_are_bool_with_the_state_shape_and_reset_silences_them(self):
        net = Network(batch_size=3, seed=0)
        group = NeuronGroup(net, 5, [CorrelatedPoissonInput(0.4, 0.5)])
        net.run(10)
        assert group.spikes.dtype == torch.bool
        assert group.spikes.shape == (3, 5)
        net.reset_state()
        assert not group.spikes.any()

    def test_predrawn_numbers_decide_the_spikes(self):
        net = Network(seed=0)
        group = NeuronGroup(net, 4, [CorrelatedPoissonInput(0.2, 0.5)])
        net.initialize()
        behavior = group.behaviors[0]
        # p_mother = 0.4: the mother spikes below 0.4, a neuron copies below 0.5
        behavior.drawn = {
            "mother": torch.tensor([0.3]),
            "copy": torch.tensor([0.1, 0.6, 0.4, 0.9]),
        }
        behavior.forward(group)
        assert group.spikes.tolist() == [True, False, True, False]
        behavior.drawn = {"mother": torch.tensor([0.5]), "copy": torch.zeros(4)}
        behavior.forward(group)
        assert not group.spikes.any()

    def test_independent_members_have_their_own_mother_train(self):
        net = Network(batch_size=200, independent=True, seed=1)
        group = NeuronGroup(net, 3, [CorrelatedPoissonInput(0.2, 1.0)])
        net.run(1)
        # c = 1: all neurons of a member agree, members differ
        assert torch.equal(group.spikes[:, 0], group.spikes[:, 1])
        assert group.spikes[:, 0].any()
        assert not group.spikes[:, 0].all()
