"""Validation of Hodgkin-Huxley equations against hand-computed values and convergence.

Validates gate rate functions, singularity limits at -40 mV and -55 mV, exact analytic
gating steps, temperature scaling, numerical resting potential, and spike time convergence.
"""

from __future__ import annotations

import math

import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons.hodgkin_huxley import (
    HodgkinHuxley,
    gate_rates,
    gate_steady_state,
    gate_time_constant,
    gating_step,
    resting_potential,
)

from .common import ConstantCurrent


class PulseCurrent(Behavior):
    """Injects current pulses of specified amplitude and duration at given times."""

    order = Order.DENDRITE_INTEGRATION

    def __init__(
        self,
        *,
        pulse_times: list[float],
        amplitude: float = 10.0,
        duration: float = 1.0,
    ) -> None:
        self.pulse_times = pulse_times
        self.amplitude = amplitude
        self.duration = duration

    def initialize(self, group: NeuronGroup) -> None:
        group.I = group.state(0.0)

    def forward(self, group: NeuronGroup) -> None:
        t = group.net.iteration * group.net.dt
        active = any(p_time <= t < p_time + self.duration for p_time in self.pulse_times)
        current = self.amplitude if active else 0.0
        group.I = group.state(current)


class TestGateRatesMath:
    def test_hand_computed_values_at_minus_65(self) -> None:
        v = torch.tensor([-65.0], dtype=torch.float64)
        rates = gate_rates(v)
        steady = gate_steady_state(v)
        tau = gate_time_constant(v)

        # Hand calculation for alpha_m(-65):
        # alpha_m = 0.1 * (-65 + 40) / (1 - exp(-(-65 + 40) / 10))
        #         = 0.1 * (-25) / (1 - exp(2.5)) = -2.5 / (1 - exp(2.5)) ≈ 0.223564
        hand_am = -2.5 / (1.0 - math.exp(2.5))
        # beta_m = 4 * exp(-(-65 + 65) / 18) = 4 * exp(0) = 4.0
        hand_bm = 4.0
        hand_m_inf = hand_am / (hand_am + hand_bm)

        # Hand calculation for alpha_h(-65):
        # alpha_h = 0.07 * exp(-(-65 + 65) / 20) = 0.07 * exp(0) = 0.07
        hand_ah = 0.07
        # beta_h = 1 / (1 + exp(-(-65 + 35) / 10)) = 1 / (1 + exp(3)) ≈ 0.04742587
        hand_bh = 1.0 / (1.0 + math.exp(3.0))
        hand_h_inf = hand_ah / (hand_ah + hand_bh)

        # Hand calculation for alpha_n(-65):
        # alpha_n = 0.01 * (-65 + 55) / (1 - exp(-(-65 + 55) / 10))
        #         = 0.01 * (-10) / (1 - exp(1)) = -0.1 / (1 - exp(1)) ≈ 0.05819767
        hand_an = -0.1 / (1.0 - math.exp(1.0))
        # beta_n = 0.125 * exp(-(-65 + 65) / 80) = 0.125 * exp(0) = 0.125
        hand_bn = 0.125
        hand_n_inf = hand_an / (hand_an + hand_bn)

        assert rates.m.alpha.item() == pytest.approx(hand_am, abs=1e-6)
        assert rates.m.alpha.item() == pytest.approx(0.2236, abs=1e-4)
        assert rates.m.beta.item() == pytest.approx(hand_bm, abs=1e-6)
        assert rates.m.beta.item() == pytest.approx(4.0, abs=1e-6)
        assert steady.m.item() == pytest.approx(hand_m_inf, abs=1e-6)
        assert steady.m.item() == pytest.approx(0.0529, abs=1e-4)

        assert rates.h.alpha.item() == pytest.approx(hand_ah, abs=1e-6)
        assert rates.h.beta.item() == pytest.approx(hand_bh, abs=1e-6)
        assert steady.h.item() == pytest.approx(hand_h_inf, abs=1e-6)
        assert steady.h.item() == pytest.approx(0.5961, abs=1e-4)

        assert rates.n.alpha.item() == pytest.approx(hand_an, abs=1e-6)
        assert rates.n.beta.item() == pytest.approx(hand_bn, abs=1e-6)
        assert steady.n.item() == pytest.approx(hand_n_inf, abs=1e-6)
        assert steady.n.item() == pytest.approx(0.3177, abs=1e-4)

        # Time constants tau = 1 / (alpha + beta)
        assert tau.m.item() == pytest.approx(1.0 / (hand_am + hand_bm), abs=1e-6)
        assert tau.h.item() == pytest.approx(1.0 / (hand_ah + hand_bh), abs=1e-6)
        assert tau.n.item() == pytest.approx(1.0 / (hand_an + hand_bn), abs=1e-6)

    def test_singular_points_finite_and_continuous(self) -> None:
        # V = -40 mV for alpha_m has limit 1.0
        v_m_sing = torch.tensor([-40.0], dtype=torch.float64)
        rates_m = gate_rates(v_m_sing)
        assert math.isfinite(rates_m.m.alpha.item())
        assert rates_m.m.alpha.item() == pytest.approx(1.0, abs=1e-9)

        # Continuity across V = -40 mV
        v_m_plus = torch.tensor([-40.0 + 1e-6], dtype=torch.float64)
        v_m_minus = torch.tensor([-40.0 - 1e-6], dtype=torch.float64)
        diff_plus_m = abs(gate_rates(v_m_plus).m.alpha.item() - 1.0)
        diff_minus_m = abs(gate_rates(v_m_minus).m.alpha.item() - 1.0)
        assert diff_plus_m < 1e-6
        assert diff_minus_m < 1e-6

        # V = -55 mV for alpha_n has limit 0.1
        v_n_sing = torch.tensor([-55.0], dtype=torch.float64)
        rates_n = gate_rates(v_n_sing)
        assert math.isfinite(rates_n.n.alpha.item())
        assert rates_n.n.alpha.item() == pytest.approx(0.1, abs=1e-9)

        # Continuity across V = -55 mV
        v_n_plus = torch.tensor([-55.0 + 1e-6], dtype=torch.float64)
        v_n_minus = torch.tensor([-55.0 - 1e-6], dtype=torch.float64)
        diff_plus_n = abs(gate_rates(v_n_plus).n.alpha.item() - 0.1)
        diff_minus_n = abs(gate_rates(v_n_minus).n.alpha.item() - 0.1)
        assert diff_plus_n < 1e-6
        assert diff_minus_n < 1e-6

    def test_temperature_scaling_phi(self) -> None:
        v = torch.tensor([-65.0], dtype=torch.float64)
        rates_base = gate_rates(v, phi=1.0)

        # Default temperature 6.3 °C gives phi = 1.0
        model_default = HodgkinHuxley(temperature=6.3, q10=3.0)
        assert model_default.phi == 1.0

        # Temperature 16.3 °C with q10 = 3.0 gives phi = 3 ** ((16.3 - 6.3) / 10) = 3.0
        model_warm = HodgkinHuxley(temperature=16.3, q10=3.0)
        assert model_warm.phi == pytest.approx(3.0, abs=1e-12)

        rates_warm = gate_rates(v, phi=model_warm.phi)
        assert rates_warm.m.alpha.item() == pytest.approx(
            3.0 * rates_base.m.alpha.item(), abs=1e-12
        )
        assert rates_warm.m.beta.item() == pytest.approx(3.0 * rates_base.m.beta.item(), abs=1e-12)
        assert rates_warm.h.alpha.item() == pytest.approx(
            3.0 * rates_base.h.alpha.item(), abs=1e-12
        )
        assert rates_warm.h.beta.item() == pytest.approx(3.0 * rates_base.h.beta.item(), abs=1e-12)
        assert rates_warm.n.alpha.item() == pytest.approx(
            3.0 * rates_base.n.alpha.item(), abs=1e-12
        )
        assert rates_warm.n.beta.item() == pytest.approx(3.0 * rates_base.n.beta.item(), abs=1e-12)


class TestGatingStepMath:
    def test_analytic_solution_and_step_composition(self) -> None:
        # At frozen V = -65 mV, compare 1000 steps of dt = 0.01 with 1 step of dt = 10.0
        v = torch.tensor([-65.0], dtype=torch.float64)
        rates = gate_rates(v)
        alpha, beta = rates.m.alpha, rates.m.beta

        # Start from non-equilibrium gate state
        x_init = torch.tensor([0.5], dtype=torch.float64)

        # Single step of dt = 10.0
        x_single = gating_step(x_init, alpha, beta, dt=10.0)

        # 1000 composed steps of dt = 0.01
        x_composed = x_init.clone()
        for _ in range(1000):
            x_composed = gating_step(x_composed, alpha, beta, dt=0.01)

        # Analytic closed-form value: x_inf + (x_init - x_inf) * exp(-(alpha + beta) * 10)
        sum_rate = alpha + beta
        x_inf = alpha / sum_rate
        analytic = x_inf + (x_init - x_inf) * torch.exp(-sum_rate * 10.0)

        assert abs(x_single.item() - analytic.item()) < 1e-12
        assert abs(x_composed.item() - analytic.item()) < 1e-12
        assert abs(x_single.item() - x_composed.item()) < 1e-12


class TestRestingPotentialMath:
    def test_resting_potential_value(self) -> None:
        v_rest = resting_potential()
        # Classic Hodgkin-Huxley resting potential is within 0.1 mV of -65 mV
        assert v_rest == pytest.approx(-65.0, abs=0.1)
        assert v_rest == pytest.approx(-64.99638, abs=1e-4)

    def test_quiescence_at_rest_over_200ms(self) -> None:
        # At rest with no external current, voltage stays within 0.5 mV and emits no spikes
        net = Network(dt=0.025, dtype=torch.float64)
        model = HodgkinHuxley()
        group = NeuronGroup(net, 1, [ConstantCurrent(0.0), model])
        net.initialize()

        v_initial = group.v.item()
        total_steps = round(200.0 / 0.025)
        v_max = v_initial
        v_min = v_initial
        spike_count = 0

        for _ in range(total_steps):
            net.step()
            v_curr = group.v.item()
            v_max = max(v_max, v_curr)
            v_min = min(v_min, v_curr)
            spike_count += int(group.spikes.sum().item())

        assert spike_count == 0
        assert abs(v_max - v_initial) < 0.5
        assert abs(v_min - v_initial) < 0.5


class TestConvergence:
    def test_spike_time_convergence(self) -> None:
        pulse_schedule = [5.0, 25.0, 45.0, 65.0, 85.0]

        def record_spike_times(dt: float) -> list[float]:
            net = Network(dt=dt, dtype=torch.float64)
            model = HodgkinHuxley(substeps=1)
            group = NeuronGroup(
                net,
                1,
                [PulseCurrent(pulse_times=pulse_schedule, amplitude=15.0, duration=1.0), model],
            )
            net.initialize()
            spikes: list[float] = []
            steps = round(105.0 / dt)
            for _ in range(steps):
                net.step()
                if bool(group.spikes.any().item()):
                    spikes.append(net.iteration * dt)
            return spikes

        # Reference run at fine resolution dt = 0.002 ms
        ref_spikes = record_spike_times(0.002)[:5]
        assert len(ref_spikes) == 5

        # Test at dt = 0.05 ms: error for first 5 spikes is strictly less than 0.2 ms
        spikes_05 = record_spike_times(0.05)[:5]
        assert len(spikes_05) == 5
        errors_05 = [abs(s - r) for s, r in zip(spikes_05, ref_spikes, strict=True)]
        assert all(err < 0.2 for err in errors_05)

        # Check monotonic error reduction as dt is halved in {0.05, 0.025, 0.0125}
        spikes_025 = record_spike_times(0.025)[:5]
        spikes_0125 = record_spike_times(0.0125)[:5]

        errors_025 = [abs(s - r) for s, r in zip(spikes_025, ref_spikes, strict=True)]
        errors_0125 = [abs(s - r) for s, r in zip(spikes_0125, ref_spikes, strict=True)]

        max_err_05 = max(errors_05)
        max_err_025 = max(errors_025)
        max_err_0125 = max(errors_0125)

        assert max_err_05 > max_err_025 >= max_err_0125
