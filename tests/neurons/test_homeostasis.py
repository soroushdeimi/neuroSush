import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons.homeostasis import ActivityHomeostasis, VoltageHomeostasis
from neurosush.neurons.models import LIF

LIF_ARGS = {"tau": 10.0, "threshold": -50.0, "v_reset": -70.0, "v_rest": -65.0}


class ForceSpikes(Behavior):
    """Overwrites spikes with a fixed pattern right after firing."""

    order = Order.FIRE + 1

    def __init__(self, pattern):
        self.pattern = torch.tensor(pattern)

    def forward(self, group):
        group.spikes = self.pattern.clone()


def activity_group(pattern, **kwargs):
    net = Network(dtype=torch.float64)
    ng = NeuronGroup(
        net,
        len(pattern),
        behaviors=[LIF(**LIF_ARGS), ForceSpikes(pattern), ActivityHomeostasis(**kwargs)],
    )
    net.initialize()
    return net, ng


class TestActivityHomeostasis:
    def test_threshold_moves_toward_the_target_rate(self):
        # window 4, target 1 spike: a spike counts +1, a silent step -1/3
        net, ng = activity_group([True, False], target_spikes=1, window=4, rate=0.3)
        net.run(3)
        assert ng.threshold.tolist() == [-50.0, -50.0]
        net.step()
        assert ng.threshold.tolist() == pytest.approx([-50.0 + 4 * 0.3, -50.0 - 4 / 3 * 0.3])

    def test_exact_target_leaves_threshold_unchanged(self):
        net, ng = activity_group([False], target_spikes=1, window=4, rate=0.3)
        ng.behaviors[1].pattern = torch.tensor([False])
        net.run(3)
        ng.behaviors[1].pattern = torch.tensor([True])
        net.step()
        assert ng.threshold.tolist() == pytest.approx([-50.0])

    def test_rate_decays_after_each_window(self):
        net, ng = activity_group([True], target_spikes=1, window=2, rate=1.0, decay=0.5)
        net.run(4)
        # windows end at steps 2 and 4; each window: 2 spikes -> activity 2
        assert ng.threshold.tolist() == pytest.approx([-50.0 + 2 * 1.0 + 2 * 0.5])

    def test_rate_decays_geometrically_and_state_dict_rate_is_a_float(self):
        windows = 5
        net, ng = activity_group([True], target_spikes=1, window=2, rate=1.0, decay=0.9)
        net.run(2 * windows)
        behavior = ng.behaviors[-1]
        assert float(behavior.rate) == pytest.approx(1.0 * 0.9**windows)
        state = behavior.state_dict()
        assert isinstance(state["rate"], float)
        assert state["rate"] == pytest.approx(1.0 * 0.9**windows)

    def test_graph_key_is_whether_the_step_ends_a_window(self):
        net, ng = activity_group([True], target_spikes=1, window=3, rate=0.1)
        behavior = ng.behaviors[-1]
        net.iteration = 2
        assert behavior.graph_key(ng) is False
        net.iteration = 3
        assert behavior.graph_key(ng) is True

    def test_graph_ready(self):
        _, ng = activity_group([True], target_spikes=1, window=3, rate=0.1)
        assert ng.behaviors[-1].graph_ready(ng) is True

    def test_needs_a_neuron_model(self):
        net = Network()
        NeuronGroup(net, 1, behaviors=[ActivityHomeostasis(target_spikes=1, window=2, rate=0.1)])
        with pytest.raises(RuntimeError, match="threshold"):
            net.initialize()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"target_spikes": 0, "window": 4, "rate": 0.1}, "target_spikes"),
            ({"target_spikes": 4, "window": 4, "rate": 0.1}, "target_spikes"),
            ({"target_spikes": 1, "window": 4, "rate": 0.0}, "rate"),
            ({"target_spikes": 1, "window": 4, "rate": 0.1, "decay": 1.5}, "decay"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ActivityHomeostasis(**kwargs)


class TestVoltageHomeostasis:
    def make(self, **kwargs):
        net = Network(dtype=torch.float64)
        ng = NeuronGroup(net, 3, behaviors=[LIF(**LIF_ARGS), VoltageHomeostasis(**kwargs)])
        net.initialize()
        return ng

    def test_pushes_back_voltages_outside_the_band(self):
        ng = self.make(v_min=-60.0, v_max=-55.0, rate=0.1)
        ng.v = torch.tensor([-50.0, -58.0, -70.0], dtype=torch.float64)
        ng.behaviors[1].forward(ng)
        # exhaustion += rate * (excess above v_max or deficit below v_min); v -= exhaustion
        assert ng.exhaustion.tolist() == pytest.approx([0.5, 0.0, -1.0])
        assert ng.v.tolist() == pytest.approx([-50.5, -58.0, -69.0])

    def test_exhaustion_accumulates(self):
        ng = self.make(target=-60.0, rate=0.1)
        for _ in range(2):
            ng.v = torch.tensor([-50.0, -60.0, -60.0], dtype=torch.float64)
            ng.behaviors[1].forward(ng)
        assert ng.exhaustion.tolist() == pytest.approx([2.0, 0.0, 0.0])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({}, "target"),
            ({"v_min": -50.0}, "target"),
            ({"target": -60.0, "v_min": -61.0}, "target"),
            ({"v_min": -50.0, "v_max": -60.0}, "v_min"),
            ({"target": -60.0, "rate": 0.0}, "rate"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            VoltageHomeostasis(**kwargs)


def test_threshold_keeps_the_network_dtype():
    # regression: counting spikes in float64 turned float32 thresholds into float64
    net = Network()
    ng = NeuronGroup(
        net,
        2,
        behaviors=[LIF(**LIF_ARGS), ActivityHomeostasis(target_spikes=1, window=2, rate=0.1)],
    )
    net.run(2)
    assert ng.threshold.dtype == torch.float32
