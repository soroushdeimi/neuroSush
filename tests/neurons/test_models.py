import pytest
import torch

from neurosush.core.network import Network, NeuronGroup
from neurosush.neurons.models import ELIF, LIF, AdaptiveELIF, Fire

LIF_ARGS = {"tau": 10.0, "threshold": -50.0, "v_reset": -70.0, "v_rest": -65.0}


def group(model, size=2, dt=1.0, fire_behavior=True, dtype=torch.float64):
    net = Network(dt=dt, dtype=dtype)
    behaviors = [model, Fire()] if fire_behavior else [model]
    ng = NeuronGroup(net, size, behaviors=behaviors)
    net.initialize()
    return net, ng


class TestLIF:
    def test_initial_state(self):
        _, ng = group(LIF(**LIF_ARGS))
        assert ng.v.tolist() == [-65.0, -65.0]
        assert ng.spikes.tolist() == [False, False]
        assert ng.spikes.dtype == torch.bool
        assert ng.I.tolist() == [0.0, 0.0]
        assert ng.threshold.tolist() == [-50.0, -50.0]
        assert ng.model is ng.behaviors[0]

    def test_v_init_scalar_and_tensor(self):
        _, ng = group(LIF(**LIF_ARGS, v_init=-60.0))
        assert ng.v.tolist() == [-60.0, -60.0]
        _, ng = group(LIF(**LIF_ARGS, v_init=torch.tensor([-61.0, -62.0])))
        assert ng.v.tolist() == [-61.0, -62.0]

    def test_one_step_with_constant_current(self):
        net, ng = group(LIF(**LIF_ARGS, resistance=2.0), dt=0.5)
        ng.I = torch.tensor([1.0, 0.0], dtype=torch.float64)
        # v += dt / tau * ((v_rest - v) + R * I) = -65 + 0.05 * 2
        ng.model.forward(ng)
        assert ng.v.tolist() == pytest.approx([-64.9, -65.0])
        assert net.iteration == 0

    def test_relaxes_to_rest(self):
        net, ng = group(LIF(**LIF_ARGS, v_init=-55.0))
        net.run(200)
        assert ng.v.tolist() == pytest.approx([-65.0, -65.0], abs=1e-6)

    def test_strong_current_makes_regular_spikes(self):
        net, ng = group(LIF(**LIF_ARGS), size=1)
        count = 0
        for _ in range(100):
            ng.I = torch.tensor([30.0], dtype=torch.float64)
            net.step()
            count += int(ng.spikes.sum())
        # v rises from -70 toward -35 with tau 10; crossing -50 takes about 6 steps
        assert 10 <= count <= 20

    def test_fire_resets_and_reports_spikes(self):
        _, ng = group(LIF(**LIF_ARGS))
        ng.v = torch.tensor([-49.0, -55.0], dtype=torch.float64)
        ng.model.fire(ng)
        assert ng.spikes.tolist() == [True, False]
        assert ng.v.tolist() == [-70.0, -55.0]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"tau": 0.0}, "tau"),
            ({"v_reset": -50.0}, "v_reset"),
            ({"resistance": -1.0}, "resistance"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            LIF(**{**LIF_ARGS, **kwargs})

    def test_v_init_of_wrong_size(self):
        with pytest.raises(ValueError, match="v_init"):
            group(LIF(**LIF_ARGS, v_init=torch.zeros(3)))


class TestELIF:
    def test_derivative_adds_exponential_term(self):
        _, ng = group(ELIF(**LIF_ARGS, delta=2.0, theta_rh=-55.0), size=1)
        ng.v = torch.tensor([-55.0], dtype=torch.float64)
        ng.model.forward(ng)
        # v += 1 / 10 * ((-65 + 55) + 2 * exp(0))
        assert ng.v.tolist() == pytest.approx([-55.0 + 0.1 * (-10.0 + 2.0)])

    def test_invalid_delta(self):
        with pytest.raises(ValueError, match="delta"):
            ELIF(**LIF_ARGS, delta=0.0, theta_rh=-55.0)


ADEX_ARGS = {**LIF_ARGS, "delta": 2.0, "theta_rh": -55.0, "alpha": 0.5, "beta": 3.0, "tau_w": 10.0}


class TestAdaptiveELIF:
    def test_initial_omega(self):
        _, ng = group(AdaptiveELIF(**ADEX_ARGS))
        assert ng.omega.tolist() == [0.0, 0.0]
        _, ng = group(AdaptiveELIF(**ADEX_ARGS, omega_init=1.5))
        assert ng.omega.tolist() == [1.5, 1.5]

    def test_omega_reduces_the_derivative(self):
        _, ng = group(AdaptiveELIF(**ADEX_ARGS, omega_init=1.0, resistance=2.0), size=1)
        ng.v = torch.tensor([-65.0], dtype=torch.float64)
        ng.model.forward(ng)
        expected = -65.0 + 0.1 * (0.0 + 2.0 * torch.exp(torch.tensor(-5.0)).item() - 2.0 * 1.0)
        assert ng.v.tolist() == pytest.approx([expected])

    def test_fire_updates_omega_with_the_pre_reset_voltage(self):
        _, ng = group(AdaptiveELIF(**ADEX_ARGS))
        ng.v = torch.tensor([-45.0, -60.0], dtype=torch.float64)
        ng.model.fire(ng)
        # omega += dt / tau_w * (alpha * (v - v_rest) - omega) + beta * spike, with v before reset
        assert ng.omega.tolist() == pytest.approx([0.1 * 0.5 * 20.0 + 3.0, 0.1 * 0.5 * 5.0])
        assert ng.v.tolist() == [-70.0, -60.0]

    def test_invalid_tau_w(self):
        with pytest.raises(ValueError, match="tau_w"):
            AdaptiveELIF(**{**ADEX_ARGS, "tau_w": 0.0})


class TestFire:
    def test_needs_a_neuron_model(self):
        net = Network()
        NeuronGroup(net, 1, behaviors=[Fire()])
        with pytest.raises(RuntimeError, match="model"):
            net.initialize()

    def test_runs_after_dynamics_in_a_step(self):
        net, ng = group(LIF(**LIF_ARGS), size=1)
        ng.v = torch.tensor([-50.5], dtype=torch.float64)
        ng.I = torch.tensor([10.0], dtype=torch.float64)
        net.step()
        # dynamics: -50.5 + 0.1 * (-14.5 + 10) = -50.95 -> no spike
        assert ng.spikes.tolist() == [False]
        ng.I = torch.tensor([30.0], dtype=torch.float64)
        net.step()
        assert ng.spikes.tolist() == [True]
        assert ng.v.tolist() == [-70.0]
