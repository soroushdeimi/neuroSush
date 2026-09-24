"""Neuron models against the closed-form solutions of their equations.

The simulator integrates ``tau dv/dt = f(v)`` with forward Euler. For the LIF the Euler
iteration has an exact solution, ``v_n = v_inf + (v_0 - v_inf) (1 - dt / tau)^n``, which the
simulation must reproduce to rounding error; the continuous solution
``v_inf + (v_0 - v_inf) exp(-t / tau)`` must be approached at first order in ``dt``.
"""

import math
from itertools import pairwise

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup
from neurosush.modulation import Dopamine, Payoff
from neurosush.neurons.models import ELIF, LIF, AdaptiveELIF, Fire

from .common import ConstantCurrent

TAU, V_REST, V_RESET, THRESHOLD = 10.0, -65.0, -70.0, -55.0


def lif_group(current, *, dt=1.0, threshold=THRESHOLD, v_init=None, model=None, n=1):
    net = Network(dt=dt, dtype=torch.float64)
    model = model or LIF(
        tau=TAU, threshold=threshold, v_reset=V_RESET, v_rest=V_REST, v_init=v_init
    )
    group = NeuronGroup(net, n, [ConstantCurrent(current), model, Fire()])
    net.initialize()
    return net, group


def spike_steps(net, group, steps):
    times = []
    for _ in range(steps):
        net.step()
        if bool(group.spikes.any()):
            times.append(net.iteration)
    return times


class TestLIF:
    def test_subthreshold_trajectory_is_the_exact_euler_solution(self):
        # v_inf = v_rest + R I = -60 stays below threshold
        net, group = lif_group(5.0, dt=0.5, v_init=-70.0)
        decay = 1 - 0.5 / TAU
        for n in range(1, 101):
            net.step()
            expected = -60.0 + (-70.0 + 60.0) * decay**n
            assert group.v.item() == pytest.approx(expected, abs=1e-12)

    def test_euler_converges_to_the_continuous_solution_at_first_order(self):
        def error(dt):
            net, group = lif_group(5.0, dt=dt, v_init=-70.0)
            net.run(round(20.0 / dt))
            exact = -60.0 + (-70.0 + 60.0) * math.exp(-20.0 / TAU)
            return abs(group.v.item() - exact)

        errors = [error(dt) for dt in (0.4, 0.2, 0.1, 0.05)]
        orders = [math.log2(a / b) for a, b in pairwise(errors)]
        assert all(order == pytest.approx(1.0, abs=0.05) for order in orders)

    def test_interspike_interval_is_the_first_threshold_crossing(self):
        # from v_reset the Euler iterate first reaches theta after n* steps:
        # n* = ceil(log((v_inf - theta) / (v_inf - v_reset)) / log(1 - dt / tau))
        current, dt = 20.0, 0.25  # v_inf = -45
        net, group = lif_group(current, dt=dt, v_init=V_RESET)
        times = spike_steps(net, group, 2000)
        v_inf = V_REST + current
        n_star = math.ceil(
            math.log((v_inf - THRESHOLD) / (v_inf - V_RESET)) / math.log(1 - dt / TAU)
        )
        assert {b - a for a, b in pairwise(times)} == {n_star}
        assert times[0] == n_star

    def test_firing_rate_approaches_the_analytic_rate(self):
        # T = tau ln((v_inf - v_reset) / (v_inf - theta)) for the continuous model
        current = 20.0
        v_inf = V_REST + current
        period = TAU * math.log((v_inf - V_RESET) / (v_inf - THRESHOLD))
        gaps = []
        for dt in (0.1, 0.01):
            net, group = lif_group(current, dt=dt, v_init=V_RESET)
            times = spike_steps(net, group, round(100 / dt))
            isi = (times[-1] - times[0]) / (len(times) - 1) * dt
            gaps.append(abs(isi - period))
        assert gaps[1] < gaps[0] / 5  # first order in dt (with ceil rounding)
        assert gaps[1] < 0.01 * period

    def test_no_spike_below_rheobase(self):
        # the membrane never reaches theta when v_inf = v_rest + R I < theta
        net, group = lif_group(THRESHOLD - V_REST - 1e-6, dt=0.1)
        assert spike_steps(net, group, 5000) == []


class TestExponentialLIF:
    # tau dv/dt = (v_rest - v) + R I + delta exp((v - theta_rh) / delta)
    # The stable and unstable fixed points merge at v = theta_rh, so the rheobase current is
    # R I_rh = theta_rh - v_rest - delta (a saddle-node bifurcation).
    DELTA, THETA_RH = 2.0, -58.0

    def run(self, current):
        model = ELIF(
            tau=TAU,
            threshold=-30.0,
            v_reset=V_RESET,
            v_rest=V_REST,
            delta=self.DELTA,
            theta_rh=self.THETA_RH,
        )
        net, group = lif_group(current, dt=0.05, model=model)
        return group, spike_steps(net, group, 8000)

    def test_rheobase_separates_rest_from_firing(self):
        rheobase = self.THETA_RH - V_REST - self.DELTA
        group, below = self.run(rheobase - 0.05)
        assert below == []

        # it settles on the stable fixed point, the root of f below theta_rh (forward Euler
        # has exactly the fixed points of the ODE); bisection finds it
        def f(v):
            return (
                (V_REST - v)
                + rheobase
                - 0.05
                + self.DELTA * math.exp((v - self.THETA_RH) / self.DELTA)
            )

        lo, hi = V_REST, self.THETA_RH  # f(lo) > 0 > f(hi)
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        assert group.v.item() == pytest.approx(lo, abs=1e-3)
        _, above = self.run(rheobase + 0.05)
        assert len(above) > 0


class TestAdaptiveExponentialLIF:
    def model(self, **kwargs):
        options = {
            "tau": TAU,
            "threshold": -40.0,
            "v_reset": V_RESET,
            "v_rest": V_REST,
            "delta": 1.0,
            "theta_rh": -50.0,
            "alpha": 0.5,
            "beta": 0.0,
            "tau_w": 50.0,
        }
        return AdaptiveELIF(**{**options, **kwargs})

    def test_subthreshold_steady_state_solves_the_coupled_equations(self):
        # at rest omega = alpha (v - v_rest), so v - v_rest = R I / (1 + R alpha) up to the
        # exponential term, which is included through a fixed-point iteration
        current, alpha = 6.0, 0.5
        net, group = lif_group(current, dt=0.1, model=self.model(alpha=alpha))
        net.run(6000)
        x = current / (1 + alpha)
        for _ in range(50):
            x = (current + math.exp((V_REST + x + 50.0) / 1.0)) / (1 + alpha)
        assert group.v.item() - V_REST == pytest.approx(x, abs=1e-6)
        assert group.omega.item() == pytest.approx(alpha * x, abs=1e-6)

    def test_spike_triggered_adaptation_lengthens_intervals_to_a_steady_rhythm(self):
        net, group = lif_group(30.0, dt=0.1, model=self.model(alpha=0.0, beta=2.0))
        times = spike_steps(net, group, 5000)
        isis = [b - a for a, b in pairwise(times)]
        assert isis[0] < isis[5] < isis[-1] + 1
        assert all(b >= a - 1 for a, b in pairwise(isis))
        assert isis[-1] == isis[-2] or abs(isis[-1] - isis[-2]) <= 1  # converged


class TestDopamine:
    def test_euler_solution_of_the_dopamine_equation(self):
        # d_{n+1} = d_n + dt (-d_n / tau + P) gives d_n = P tau + (d_0 - P tau)(1 - dt / tau)^n
        tau, payoff, dt, d0 = 8.0, 0.3, 0.5, 1.0
        net = Network(dt=dt, behaviors=[Payoff(lambda n: payoff), Dopamine(tau=tau, initial=d0)])
        for n in range(1, 60):
            net.step()
            expected = payoff * tau + (d0 - payoff * tau) * (1 - dt / tau) ** n
            assert net.dopamine == pytest.approx(expected, rel=1e-12)
