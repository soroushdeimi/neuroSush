import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup
from neurosush.core.order import Order
from neurosush.recording import Recorder


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
