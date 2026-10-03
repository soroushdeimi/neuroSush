import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons.inputs import SpikeInput
from neurosush.recording import Recorder, SpikeCounter


class Counter(Behavior):
    order = Order.NEURON_DYNAMICS

    def initialize(self, group):
        group.v = group.state()

    def forward(self, group):
        group.v = group.v + 1  # a new tensor every step


def test_records_end_of_step_values_at_the_interval():
    net = Network()
    recorder = Recorder("v", interval=2)
    NeuronGroup(net, 3, [recorder, Counter()])  # attached first, still runs last
    net.run(6)
    assert recorder.steps == [2, 4, 6]
    assert recorder.get("v").tolist() == [[2.0] * 3, [4.0] * 3, [6.0] * 3]


def test_copies_are_independent_of_the_host():
    net = Network()
    group = NeuronGroup(net, 2, [Counter(), recorder := Recorder("v")])
    net.run(1)
    group.v.add_(100)  # in-place change after recording
    assert recorder.get("v").tolist() == [[1.0, 1.0]]


def test_records_network_scalars():
    net = Network(behaviors=[recorder := Recorder("iteration")])
    net.run(3)
    assert recorder.get("iteration").tolist() == [1, 2, 3]


def test_batched_state_gets_a_time_dimension():
    net = Network(batch_size=4)
    NeuronGroup(net, 3, [Counter(), recorder := Recorder("v")])
    net.run(5)
    assert recorder.get("v").shape == (5, 4, 3)


def test_reset_drops_the_history():
    net = Network()
    NeuronGroup(net, 1, [Counter(), recorder := Recorder("v")])
    net.run(2)
    recorder.reset()
    net.run(1)
    assert recorder.steps == [3]
    assert recorder.get("v").tolist() == [[3.0]]


def test_missing_attribute_fails_at_initialization():
    net = Network()
    NeuronGroup(net, 1, [Recorder("v")], name="n")
    with pytest.raises(RuntimeError, match=r"Recorder on n: no attribute\(s\) \['v'\]"):
        net.initialize()


def test_unknown_name_and_arguments():
    recorder = Recorder("v")
    with pytest.raises(KeyError, match="'w' is not recorded"):
        recorder.get("w")
    with pytest.raises(ValueError, match="at least one"):
        Recorder()
    with pytest.raises(ValueError, match="interval"):
        Recorder("v", interval=0)


def test_empty_history():
    assert Recorder("v").get("v").tolist() == []
    assert torch.equal(Recorder("v").get("v"), torch.tensor([]))


def frames(*rows):
    return iter([torch.tensor(row, dtype=torch.bool) for row in rows])


class TestSpikeCounter:
    def test_counts_spikes_per_neuron(self):
        net = Network()
        group = NeuronGroup(
            net, 3, [SpikeInput(frames([1, 0, 1], [1, 1, 0], [1, 0, 0])), SpikeCounter()]
        )
        net.run(3)
        assert group.spike_count.tolist() == [3.0, 1.0, 1.0]
        assert group.spike_count.dtype == net.dtype

    def test_reset_state_starts_a_new_sample(self):
        net = Network()
        group = NeuronGroup(net, 2, [SpikeInput(frames([1, 1], [0, 1], [1, 0])), SpikeCounter()])
        net.run(2)
        address = group.spike_count.data_ptr()
        net.reset_state()
        assert group.spike_count.tolist() == [0.0, 0.0]
        assert group.spike_count.data_ptr() == address
        net.step()
        assert group.spike_count.tolist() == [1.0, 0.0]

    def test_batched_counts_are_per_sample(self):
        net = Network(batch_size=2)
        rows = [
            torch.tensor([[1, 0], [0, 0]], dtype=torch.bool),
            torch.tensor([[1, 1], [0, 1]], dtype=torch.bool),
        ]
        group = NeuronGroup(net, 2, [SpikeInput(iter(rows)), SpikeCounter()])
        net.run(2)
        assert group.spike_count.tolist() == [[2.0, 1.0], [0.0, 1.0]]

    def test_runs_after_the_spikes_of_the_step_and_below_the_recorder(self):
        assert SpikeCounter.order == Order.ACTIVITY_HOMEOSTASIS
        assert Order.FIRE < SpikeCounter.order < Order.RECORD
        assert SpikeCounter.graph_safe

    def test_a_group_without_spikes_is_rejected(self):
        net = Network()
        NeuronGroup(net, 2, [SpikeCounter()], name="bare")
        with pytest.raises(RuntimeError, match="bare"):
            net.initialize()
