import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons.adaptation import SpikeTriggeredCurrent
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.models import LIF, Fire

LIF_ARGS = {"tau": 10.0, "threshold": -55.0, "v_reset": -70.0, "v_rest": -65.0}


class Drive(Behavior):
    order = Order.DENDRITE_INTEGRATION

    def forward(self, group):
        group.I = group.state(14.0)

    def initialize(self, group):
        group.I = group.state(14.0)


def test_runs_between_dendrite_integration_and_the_neuron_model():
    assert Order.DENDRITE_INTEGRATION < SpikeTriggeredCurrent.order < Order.NEURON_DYNAMICS


class TestSpikeTriggeredCurrent:
    @pytest.mark.parametrize(
        ("amplitude", "tau", "match"),
        [
            (-1.0, 0.0, "tau must be positive, got 0.0"),
            (1.0, torch.tensor([1.0, -2.0]), "tau must"),
        ],
    )
    def test_invalid_arguments(self, amplitude, tau, match):
        with pytest.raises(ValueError, match=match):
            SpikeTriggeredCurrent(amplitude, tau)

    def test_tau_must_be_at_least_dt(self):
        net = Network(dt=2.0)
        NeuronGroup(net, 1, [SpikeTriggeredCurrent(-1.0, 1.0), LIF(**LIF_ARGS)])
        with pytest.raises(ValueError, match=r"tau \(1.0\) must be at least dt \(2.0\)"):
            net.initialize()

    def test_refuses_a_filtered_dendritic_current(self):
        net = Network()
        NeuronGroup(
            net,
            1,
            [
                DendriteStructure(),
                DendriteIntegration(tau_current=5.0),
                SpikeTriggeredCurrent(-1.0, 10.0),
                LIF(**LIF_ARGS),
            ],
        )
        with pytest.raises(ValueError, match="cannot follow DendriteIntegration"):
            net.initialize()

    def test_adds_to_the_current_and_resets(self):
        net = Network(dtype=torch.float64)
        group = NeuronGroup(
            net, 2, [DendriteStructure(), DendriteIntegration(), SpikeTriggeredCurrent(-2.0, 4.0)]
        )
        net.initialize()
        group.spikes = torch.tensor([True, False])
        net.step()
        assert group.I_adapt.tolist() == [-2.0, 0.0]
        assert group.I.tolist() == [-2.0, 0.0]
        group.spikes = torch.tensor([False, False])
        net.step()
        assert group.I_adapt.tolist() == [-1.5, 0.0]
        net.reset_state()
        assert group.I_adapt.tolist() == [0.0, 0.0]

    def test_independent_members_take_their_own_amplitude_and_tau(self):
        amplitude = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]], dtype=torch.float64)
        tau = torch.tensor([[2.0, 4.0], [8.0, 16.0]], dtype=torch.float64)
        net = Network(dtype=torch.float64, batch_size=2, independent=True)
        group = NeuronGroup(
            net,
            2,
            [DendriteStructure(), DendriteIntegration(), SpikeTriggeredCurrent(amplitude, tau)],
        )
        net.initialize()
        group.spikes = torch.ones(2, 2, dtype=torch.bool)
        net.step()
        group.spikes = torch.zeros(2, 2, dtype=torch.bool)
        net.step()
        assert torch.allclose(group.I_adapt, amplitude * (1 - 1.0 / tau))

    def test_slows_a_lif_neuron_down(self):
        def count(amplitude):
            net = Network(dtype=torch.float64)
            group = NeuronGroup(
                net,
                1,
                [
                    Drive(),
                    SpikeTriggeredCurrent(amplitude, 30.0),
                    LIF(**LIF_ARGS),
                    Fire(),
                ],
            )
            net.initialize()
            total = 0
            for _ in range(300):
                net.step()
                total += int(group.spikes.any())
            return total

        assert count(-2.0) < count(0.0)
