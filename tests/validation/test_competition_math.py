"""Refractoriness, adaptive thresholds, conductance synapses and Poisson input, in closed form.

These are the mechanisms of competitive STDP networks such as Diehl and Cook (2015):

- ``Refractory(period)`` holds a neuron at ``v_reset`` for ``ceil(period / dt)`` steps after
  a spike, so under a constant drive the interspike interval is the LIF crossing time plus
  that many steps.
- ``AdaptiveThreshold`` gives ``theta_N = increment * sum_k (1 - dt / tau)^(N - k)`` over
  the spike steps ``k``.
- ``ConductanceIntegration`` lands every step on the exact solution of
  ``tau dv/dt = (v_rest - v) + R g_e (e_e - v) + R g_i (e_i - v)`` with the conductances
  held over the step, so a constant conductance gives
  ``v_inf + (v_0 - v_inf) exp(-(1 + R g) t / tau)``.
- ``PoissonInput`` spike counts are ``Binomial(steps, rate * dt)``.
"""

import math
from itertools import pairwise

import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import ConductanceIntegration, conductance_step
from neurosush.neurons.homeostasis import AdaptiveThreshold
from neurosush.neurons.inputs import PoissonInput
from neurosush.neurons.models import LIF, Fire, Refractory
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.traces import SpikeGather

from .common import ConstantCurrent, ScriptedSpikes

TAU, V_REST, V_RESET, THRESHOLD = 10.0, -65.0, -70.0, -55.0


def crossing_steps(v0, v_inf, dt):
    """Euler steps of the LIF from ``v0`` until it reaches the threshold."""
    v, n = v0, 0
    while v < THRESHOLD:
        v = v + (v_inf - v) * dt / TAU
        n += 1
    return n


def spike_steps(net, group, steps):
    times = []
    for _ in range(steps):
        net.step()
        if bool(group.spikes.any()):
            times.append(net.iteration)
    return times


class TestRefractory:
    @pytest.mark.parametrize(("period", "dt"), [(3.0, 1.0), (5.0, 1.0), (2.5, 0.5), (1.2, 0.5)])
    def test_interspike_interval_adds_the_refractory_steps(self, period, dt):
        current = 20.0  # v_inf = -45, above threshold
        net = Network(dt=dt, dtype=torch.float64)
        group = NeuronGroup(
            net,
            1,
            [
                ConstantCurrent(current),
                LIF(tau=TAU, threshold=THRESHOLD, v_reset=V_RESET, v_rest=V_REST, v_init=V_RESET),
                Refractory(period),
                Fire(),
            ],
        )
        net.initialize()
        times = spike_steps(net, group, round(400 / dt))
        free = crossing_steps(V_RESET, V_REST + current, dt)
        assert times[0] == free  # nothing to recover from before the first spike
        assert {b - a for a, b in pairwise(times)} == {free + math.ceil(period / dt)}

    def test_held_at_reset_during_the_period(self):
        net = Network(dtype=torch.float64)
        group = NeuronGroup(
            net,
            1,
            [
                ConstantCurrent(30.0),
                LIF(tau=TAU, threshold=THRESHOLD, v_reset=V_RESET, v_rest=V_REST),
                Refractory(4.0),
                Fire(),
            ],
        )
        net.initialize()
        times = spike_steps(net, group, 20)
        net2 = Network(dtype=torch.float64)
        group2 = NeuronGroup(
            net2,
            1,
            [
                ConstantCurrent(30.0),
                LIF(tau=TAU, threshold=THRESHOLD, v_reset=V_RESET, v_rest=V_REST),
                Refractory(4.0),
                Fire(),
            ],
        )
        net2.initialize()
        net2.run(times[0])
        for _ in range(4):
            net2.step()
            assert group2.v.item() == V_RESET
        net2.step()
        assert group2.v.item() > V_RESET


class Forced(Behavior):
    """Overrides the spikes after firing: ``{step: [neuron, ...]}`` (per sample if batched)."""

    order = Order.FIRE + 1

    def __init__(self, schedule):
        self.schedule = schedule

    def forward(self, group):
        spikes = group.state(False, dtype=torch.bool)
        for index in self.schedule.get(group.net.iteration, []):
            spikes[index] = True
        group.spikes = spikes


class TestAdaptiveThreshold:
    def group(self, schedule, *, tau, dt=1.0, batch_size=None, increment=0.05, n=2):
        net = Network(dt=dt, dtype=torch.float64, batch_size=batch_size)
        group = NeuronGroup(
            net,
            n,
            [
                LIF(tau=TAU, threshold=THRESHOLD, v_reset=V_RESET, v_rest=V_REST),
                Forced(schedule),
                AdaptiveThreshold(increment=increment, tau=tau),
            ],
        )
        net.initialize()
        return net, group

    @pytest.mark.parametrize(("tau", "dt"), [(None, 1.0), (30.0, 1.0), (30.0, 0.5)])
    def test_theta_is_the_decayed_sum_of_increments(self, tau, dt):
        spikes = {3: [0], 4: [0], 9: [0, 1], 20: [1]}
        net, group = self.group(spikes, tau=tau, dt=dt)
        decay = 1.0 if tau is None else 1 - dt / tau
        for n in range(1, 41):
            net.step()
            for neuron in (0, 1):
                expected = 0.05 * sum(
                    decay ** (n - k) for k, fired in spikes.items() if k <= n and neuron in fired
                )
                assert group.theta[neuron].item() == pytest.approx(expected, rel=1e-12, abs=0)
                assert group.threshold[neuron].item() == pytest.approx(THRESHOLD + expected)

    def test_a_batch_raises_the_shared_threshold_by_its_mean(self):
        # four samples, neuron 0 fires in three of them at step 2: theta rises by 0.75 * inc
        net, group = self.group({2: [(0, 0), (1, 0), (3, 0)]}, tau=None, batch_size=4)
        net.run(2)
        assert group.theta.tolist() == pytest.approx([0.75 * 0.05, 0.0])

    def test_disabled_freezes_theta(self):
        net, group = self.group({1: [0], 5: [0]}, tau=None)
        net.run(2)
        group.behaviors[-1].enabled = False
        net.run(10)
        assert group.theta[0].item() == pytest.approx(0.05)


class TestConductanceStep:
    def test_matches_a_fine_integration_of_the_ode(self):
        v, g_e, g_i = torch.tensor(-70.0, dtype=torch.float64), 0.8, 0.3
        args = {"v_rest": -65.0, "e_exc": 0.0, "e_inh": -100.0, "resistance": 1.0, "tau": 10.0}
        exact = conductance_step(
            v,
            torch.tensor(g_e, dtype=torch.float64),
            torch.tensor(g_i, dtype=torch.float64),
            dt=1.0,
            **args,
        )

        def rate(x):
            return ((-65.0 - x) + g_e * (0.0 - x) + g_i * (-100.0 - x)) / 10.0

        x, h = -70.0, 1e-2
        for _ in range(round(1.0 / h)):  # RK4: global error O(h^4), about 1e-12 here
            k1 = rate(x)
            k2 = rate(x + h / 2 * k1)
            k3 = rate(x + h / 2 * k2)
            k4 = rate(x + h * k3)
            x += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        assert exact.item() == pytest.approx(x, abs=1e-9)

    def test_stays_between_the_reversal_potentials(self):
        v = torch.tensor([-70.0, -40.0], dtype=torch.float64)
        huge = torch.tensor(1e6, dtype=torch.float64)
        zero = torch.zeros((), dtype=torch.float64)
        args = {"v_rest": -65.0, "e_exc": 0.0, "e_inh": -100.0, "resistance": 1.0, "tau": 10.0}
        # the fixed point is v_rest / (1 + R g) = -6.5e-5, not exactly e_exc, for g = 1e6
        excited = conductance_step(v, huge, zero, dt=1.0, **args)
        assert torch.allclose(excited, torch.full((2,), -65.0 / (1 + 1e6), dtype=torch.float64))
        assert bool((excited <= 0.0).all())
        inhibited = conductance_step(v, zero, huge, dt=1.0, **args)
        assert torch.allclose(inhibited, torch.full((2,), -100.0, dtype=torch.float64))


def conductance_net(weight, *, inhibitory, tau_g, schedule, dt=1.0, e_inh=-100.0):
    """One source neuron firing on ``schedule`` into one LIF through a conductance synapse."""
    net = Network(dt=dt, dtype=torch.float64)
    source = NeuronGroup(net, 1, [ScriptedSpikes(schedule), Axon()], inhibitory=inhibitory)
    tau_exc, tau_inh = (dt, tau_g) if inhibitory else (tau_g, dt)
    target = NeuronGroup(
        net,
        1,
        [
            ConductanceIntegration(e_exc=0.0, e_inh=e_inh, tau_exc=tau_exc, tau_inh=tau_inh),
            LIF(tau=TAU, threshold=100.0, v_reset=V_RESET, v_rest=V_REST),
        ],
    )
    SynapseGroup(
        net,
        source,
        target,
        [
            WeightInit(weights=torch.tensor([[weight]], dtype=torch.float64)),
            DenseInput(),
            SpikeGather(),
        ],
    )
    net.initialize()
    return net, target


class TestConductanceIntegration:
    def test_constant_conductance_relaxes_exponentially(self):
        # a source firing every step with tau_g = dt holds g = w from the second step on
        w, steps = 0.6, 40
        net, target = conductance_net(
            w, inhibitory=False, tau_g=1.0, schedule={t: [0] for t in range(1, steps + 1)}
        )
        net.step()  # the first spike reaches the target one step later
        v0 = target.v.item()
        v_inf = (V_REST + w * 0.0) / (1 + w)
        for n in range(1, steps):
            net.step()
            expected = v_inf + (v0 - v_inf) * math.exp(-(1 + w) * n / TAU)
            assert target.v.item() == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize("tau_g", [1.0, 2.0, 5.0])
    def test_a_spike_conductance_integrates_to_weight_times_tau(self, tau_g):
        net, target = conductance_net(2.0, inhibitory=True, tau_g=tau_g, schedule={3: [0]})
        total = 0.0
        for _ in range(400):
            net.step()
            total += target.g_inh.item() * net.dt
        assert total == pytest.approx(2.0 * tau_g, rel=1e-9)

    def test_every_step_lands_on_the_exact_step(self):
        net, target = conductance_net(
            40.0, inhibitory=True, tau_g=2.0, schedule={2: [0], 3: [0], 4: [0], 9: [0]}
        )
        for _ in range(30):
            v = target.v.clone()
            net.step()
            expected = conductance_step(
                v,
                target.g_exc,
                target.g_inh,
                v_rest=V_REST,
                e_exc=0.0,
                e_inh=-100.0,
                resistance=1.0,
                tau=TAU,
                dt=1.0,
            )
            assert target.v.item() == pytest.approx(expected.item(), abs=1e-10)
            assert -100.0 <= target.v.item() <= V_REST  # strong inhibition never overshoots


class TestPoissonInput:
    def test_counts_are_binomial(self):
        rate, dt, steps, n = 0.035, 2.0, 300, 4000  # p = rate * dt = 0.07
        net = Network(dt=dt, seed=0)
        group = NeuronGroup(net, n, [PoissonInput(rate)])
        counts = torch.zeros(n, dtype=torch.float64)
        for _ in range(steps):
            net.step()
            counts += group.spikes
        p = rate * dt
        mean, var = steps * p, steps * p * (1 - p)
        assert abs(counts.mean().item() - mean) < 5 * math.sqrt(var / n)
        assert abs(counts.var().item() - var) < 5 * var * math.sqrt(2 / n)

    def test_rates_changed_in_place_take_effect(self):
        net = Network(seed=0, batch_size=2)
        group = NeuronGroup(net, 500, [PoissonInput()])
        net.run(5)
        assert not group.spikes.any()
        group.rates[1].fill_(1.0)  # probability one per step in the second sample
        net.step()
        assert not group.spikes[0].any()
        assert group.spikes[1].all()
