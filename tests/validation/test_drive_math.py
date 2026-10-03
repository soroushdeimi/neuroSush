"""Per-neuron time constants, spike-triggered currents and Poisson drives against closed forms."""

from itertools import pairwise

import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons import dynamics
from neurosush.neurons.adaptation import SpikeTriggeredCurrent
from neurosush.neurons.dendrite import ConductanceIntegration
from neurosush.neurons.inputs import CorrelatedPoissonInput, PoissonDrive
from neurosush.neurons.models import LIF, AdaptiveELIF, Fire

from .common import ConstantCurrent

ELIF_ARGS = {
    "tau": 10.0,
    "threshold": 1e9,
    "v_reset": -70.0,
    "v_rest": -65.0,
    "delta": 2.0,
    "theta_rh": -50.0,
}


class TestPerNeuronAdaptation:
    def test_omega_decays_with_each_neurons_own_tau_w(self):
        tau_w = torch.tensor([5.0, 20.0, 80.0], dtype=torch.float64)
        net = Network(dt=0.5, dtype=torch.float64)
        group = NeuronGroup(
            net, 3, [AdaptiveELIF(alpha=0.0, beta=0.0, tau_w=tau_w, **ELIF_ARGS), Fire()]
        )
        net.initialize()
        group.omega.fill_(2.0)
        group.v.fill_(-65.0)
        for n in range(1, 41):
            group.model.fire(group)
            expected = 2.0 * (1 - 0.5 / tau_w) ** n
            assert torch.allclose(group.omega, expected, atol=1e-12, rtol=0)

    def test_alpha_beta_tau_w_per_neuron_match_the_hand_computation(self):
        alpha = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
        beta = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        tau_w = torch.tensor([10.0, 20.0, 40.0], dtype=torch.float64)
        net = Network(dt=1.0, dtype=torch.float64)
        kwargs = {**ELIF_ARGS, "threshold": -55.0}
        group = NeuronGroup(
            net, 3, [AdaptiveELIF(alpha=alpha, beta=beta, tau_w=tau_w, **kwargs), Fire()]
        )
        net.initialize()
        group.v = torch.tensor([-50.0, -52.0, -40.0], dtype=torch.float64)  # all spike
        group.omega = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
        group.model.fire(group)
        v = [-50.0, -52.0, -40.0]
        w = [1.0, 0.0, -1.0]
        expected = [
            w[i] + (alpha[i].item() * (v[i] + 65.0) - w[i]) / tau_w[i].item() + beta[i].item()
            for i in range(3)
        ]
        assert group.omega.tolist() == pytest.approx(expected, abs=1e-12)

    def test_independent_members_take_their_own_tau_w(self):
        tau_w = torch.tensor([[10.0, 20.0], [40.0, 80.0]], dtype=torch.float64)
        net = Network(dt=1.0, dtype=torch.float64, batch_size=2, independent=True)
        group = NeuronGroup(
            net, 2, [AdaptiveELIF(alpha=0.0, beta=0.0, tau_w=tau_w, **ELIF_ARGS), Fire()]
        )
        net.initialize()
        group.omega.fill_(1.0)
        for n in range(1, 6):
            group.model.fire(group)
            assert torch.allclose(group.omega, (1 - 1.0 / tau_w) ** n, atol=1e-12, rtol=0)

    def test_scalar_parameters_stay_plain_floats(self):
        net = Network(dt=1.0, dtype=torch.float64)
        group = NeuronGroup(net, 2, [AdaptiveELIF(alpha=0.1, beta=1.0, tau_w=30.0, **ELIF_ARGS)])
        net.initialize()
        assert isinstance(group.model._tau_w, float)
        assert group.model._tau_w == 30.0


class TestPerNeuronConductance:
    def test_conductances_decay_with_each_neurons_own_time_constant(self):
        tau_exc = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float64)
        tau_inh = torch.tensor([4.0, 8.0, 16.0], dtype=torch.float64)
        net = Network(dt=1.0, dtype=torch.float64)
        lif = LIF(tau=10.0, threshold=1e9, v_reset=-70.0, v_rest=-65.0)
        group = NeuronGroup(net, 3, [ConductanceIntegration(tau_exc=tau_exc, tau_inh=tau_inh), lif])
        net.initialize()
        group.g_exc.fill_(1.0)
        group.g_inh.fill_(2.0)
        for n in range(1, 21):
            net.step()
            assert torch.allclose(group.g_exc, (1 - 1.0 / tau_exc) ** n, atol=1e-12, rtol=0)
            assert torch.allclose(group.g_inh, 2.0 * (1 - 1.0 / tau_inh) ** n, atol=1e-12, rtol=0)


class SpikeAt(Behavior):
    """Keeps ``group.v`` where it is (no neuron model): used to drive other behaviors."""

    order = Order.NEURON_DYNAMICS

    def initialize(self, group):
        group.v = group.state()
        group.spikes = group.state(False, dtype=torch.bool)

    def forward(self, group):
        group.v = group.state()


def euler_isis(*, drive, amplitude, tau_a, dt, steps):
    """The Euler recursion of a LIF with a spike-triggered current, step by step."""
    tau, v_rest, v_reset, threshold = 10.0, -65.0, -70.0, -55.0
    v, current, spiked, times = v_rest, 0.0, False, []
    for n in range(1, steps + 1):
        current = current * (1 - dt / tau_a) + amplitude * spiked
        v = v + dt / tau * ((v_rest - v) + drive + current)
        spiked = v >= threshold
        if spiked:
            v = v_reset
            times.append(n)
    return times


class TestSpikeTriggeredCurrent:
    @pytest.mark.parametrize("amplitude", [-3.0, 0.0, 1.5])
    def test_spike_times_follow_the_euler_recursion(self, amplitude):
        dt = 0.5
        net = Network(dt=dt, dtype=torch.float64)
        lif = LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0)
        group = NeuronGroup(
            net, 1, [ConstantCurrent(14.0), SpikeTriggeredCurrent(amplitude, 40.0), lif, Fire()]
        )
        net.initialize()
        times = []
        for n in range(1, 801):
            net.step()
            if bool(group.spikes.any()):
                times.append(n)
        assert times == euler_isis(drive=14.0, amplitude=amplitude, tau_a=40.0, dt=dt, steps=800)
        assert len(times) >= 5

    def test_negative_amplitude_makes_intervals_grow(self):
        net = Network(dt=0.5, dtype=torch.float64)
        lif = LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0)
        group = NeuronGroup(
            net, 1, [ConstantCurrent(14.0), SpikeTriggeredCurrent(-3.0, 40.0), lif, Fire()]
        )
        net.initialize()
        times = []
        for n in range(1, 801):
            net.step()
            if bool(group.spikes.any()):
                times.append(n)
        gaps = [b - a for a, b in pairwise(times)]
        assert gaps[0] < gaps[-1]

    def test_current_decays_exactly_with_its_own_tau_per_neuron(self):
        tau = torch.tensor([2.0, 10.0], dtype=torch.float64)
        amplitude = torch.tensor([-1.0, 4.0], dtype=torch.float64)
        net = Network(dt=1.0, dtype=torch.float64)
        group = NeuronGroup(net, 2, [SpikeAt(), SpikeTriggeredCurrent(amplitude, tau)])
        net.initialize()
        group.I = group.state()
        group.spikes = torch.tensor([True, True])
        net.step()
        assert torch.allclose(group.I_adapt, amplitude)
        group.spikes = torch.tensor([False, False])
        for n in range(1, 8):
            net.step()
            assert torch.allclose(group.I_adapt, amplitude * (1 - 1.0 / tau) ** n, atol=1e-12)


def poisson_drive_increments(count, rate, jump, *, dt, size=200_000, steps=1, seed=0):
    net = Network(dt=dt, dtype=torch.float64, seed=seed)
    group = NeuronGroup(net, size, [SpikeAt(), PoissonDrive(count, rate, jump)])
    net.initialize()
    out = []
    for _ in range(steps):
        net.step()
        out.append(group.v.clone())
    return torch.cat(out)


class TestPoissonDrive:
    @pytest.mark.parametrize(
        ("count", "rate", "jump", "dt"), [(50, 0.02, 0.3, 0.5), (200, 0.01, -0.1, 1.0)]
    )
    def test_increment_mean_and_variance(self, count, rate, jump, dt):
        increments = poisson_drive_increments(count, rate, jump, dt=dt)
        mean, var = dynamics.poisson_increment_moments(count=count, rate=rate, jump=jump, dt=dt)
        assert mean == pytest.approx(jump * count * rate * dt)
        assert var == pytest.approx(jump**2 * count * rate * dt)
        n = increments.numel()
        assert increments.mean().item() == pytest.approx(mean, abs=5 * (var / n) ** 0.5)
        assert increments.var().item() == pytest.approx(var, rel=0.03)

    def test_increments_are_whole_numbers_of_jumps(self):
        increments = poisson_drive_increments(30, 0.02, 0.25, dt=0.5, size=1000)
        counts = increments / 0.25
        assert torch.allclose(counts, counts.round(), atol=1e-9)
        assert counts.min() >= 0

    def test_zero_rate_adds_nothing(self):
        assert poisson_drive_increments(30, 0.0, 1.0, dt=1.0, size=100).abs().max() == 0


def mif_spikes(rate, correlation, *, n_neurons=4, batch=4000, steps=400, dt=1.0, seed=3):
    net = Network(dt=dt, dtype=torch.float64, batch_size=batch, seed=seed)
    group = NeuronGroup(net, n_neurons, [CorrelatedPoissonInput(rate, correlation)])
    net.initialize()
    out = []
    for _ in range(steps):
        net.step()
        out.append(group.spikes.clone())
    return torch.stack(out).to(torch.float64)  # (steps, batch, neurons)


class TestCorrelatedPoissonInput:
    @pytest.mark.parametrize("correlation", [0.2, 0.5])
    def test_rate_and_pairwise_count_correlation(self, correlation):
        rate, dt, bin_steps = 0.02, 1.0, 20
        spikes = mif_spikes(rate, correlation, dt=dt)
        assert spikes.mean().item() / dt == pytest.approx(rate, rel=0.03)
        steps, batch, n = spikes.shape
        counts = spikes.reshape(steps // bin_steps, bin_steps, batch, n).sum(1).reshape(-1, n)
        corr = torch.corrcoef(counts.T)
        p_mother = rate / correlation * dt
        expected = dynamics.correlated_pair_correlation(correlation=correlation, p_mother=p_mother)
        off_diagonal = corr[~torch.eye(n, dtype=torch.bool)]
        assert expected == pytest.approx(correlation, abs=0.03)
        assert off_diagonal.mean().item() == pytest.approx(expected, abs=0.012)

    def test_full_correlation_gives_identical_trains(self):
        spikes = mif_spikes(0.05, 1.0, steps=100, batch=10)
        assert torch.equal(spikes[..., 0], spikes[..., 1])
        assert spikes.sum() > 0

    def test_samples_of_a_batch_are_independent(self):
        spikes = mif_spikes(0.02, 0.5)
        counts = spikes.sum(0)[:, 0]  # neuron 0, total count per sample
        other = spikes.sum(0)[:, 1]
        cross = torch.corrcoef(torch.stack([counts[:-1], counts[1:]]))[0, 1]
        assert abs(cross.item()) < 0.05
        assert torch.corrcoef(torch.stack([counts, other]))[0, 1].item() > 0.3
