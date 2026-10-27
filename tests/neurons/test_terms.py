import math

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup
from neurosush.neurons.dynamics import (
    adex_terms,
    elif_terms,
    euler_step,
    exponential_boost,
    izhikevich_step,
    izhikevich_terms,
    lif_derivative,
    lif_terms,
)
from neurosush.neurons.models import ELIF, LIF, AdaptiveELIF, Izhikevich


def t(*values: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


class TestLifTerms:
    def test_hand_computed_values_case_1(self) -> None:
        # leak: (-65.0 - (-70.0)) * 0.5 / 20.0 = 5.0 * 0.025 = 0.125
        # input: 3.0 * 2.0 * 0.5 / 20.0 = 6.0 * 0.025 = 0.15
        out = lif_terms(
            t(-70.0),
            2.0,
            v_rest=-65.0,
            resistance=3.0,
            tau=20.0,
            dt=0.5,
        )
        assert out["leak"].tolist() == pytest.approx([0.125])
        assert out["input"].tolist() == pytest.approx([0.15])

    def test_hand_computed_values_case_2(self) -> None:
        # leak: (-65.0 - (-50.0)) * 1.0 / 10.0 = -15.0 * 0.1 = -1.5
        # input: 1.0 * 0.0 * 1.0 / 10.0 = 0.0
        out = lif_terms(
            t(-50.0),
            0.0,
            v_rest=-65.0,
            resistance=1.0,
            tau=10.0,
            dt=1.0,
        )
        assert out["leak"].tolist() == pytest.approx([-1.5])
        assert out["input"].tolist() == pytest.approx([0.0])

    def test_hand_computed_values_case_3(self) -> None:
        # leak: (-65.0 - (-65.0)) * 0.5 / 25.0 = 0.0
        # input: 2.0 * 10.0 * 0.5 / 25.0 = 20.0 * 0.02 = 0.4
        out = lif_terms(
            t(-65.0),
            10.0,
            v_rest=-65.0,
            resistance=2.0,
            tau=25.0,
            dt=0.5,
        )
        assert out["leak"].tolist() == pytest.approx([0.0])
        assert out["input"].tolist() == pytest.approx([0.4])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_terms_sum_equals_euler_step(self, dtype: torch.dtype) -> None:
        v = t(-70.0, -60.0, -50.0, dtype=dtype)
        current = t(1.0, 2.0, 3.0, dtype=dtype)
        terms = lif_terms(v, current, v_rest=-65.0, resistance=2.0, tau=15.0, dt=0.5)
        expected = euler_step(
            v,
            lif_derivative(v, current, v_rest=-65.0, resistance=2.0),
            tau=15.0,
            dt=0.5,
        )
        v_next = v + sum(terms.values())
        atol = 1e-6 if dtype == torch.float32 else 1e-12
        assert torch.allclose(v_next, expected, atol=atol)

    def test_matches_network_behavior(self) -> None:
        net = Network(dt=0.5, dtype=torch.float64)
        group = NeuronGroup(
            net,
            1,
            [
                LIF(
                    tau=20.0,
                    threshold=0.0,
                    v_reset=-70.0,
                    v_rest=-65.0,
                    resistance=2.0,
                    v_init=-60.0,
                )
            ],
        )
        net.initialize()
        group.I.fill_(3.0)
        v_before = group.v.clone()
        terms = lif_terms(
            v_before,
            group.I,
            v_rest=group.v_rest,
            resistance=group.resistance,
            tau=group.tau,
            dt=net.dt,
        )
        net.step()
        assert group.v.item() == pytest.approx((v_before + sum(terms.values())).item(), abs=1e-12)

    def test_broadcasting_over_vector_of_voltages(self) -> None:
        v = t(-80.0, -70.0, -60.0, -50.0)
        terms = lif_terms(v, 2.0, v_rest=-65.0, resistance=1.0, tau=20.0, dt=1.0)
        assert terms["leak"].shape == (4,)
        assert terms["input"].shape == (4,)

    def test_inputs_not_mutated(self) -> None:
        v = t(-70.0, -50.0)
        current = t(2.0, 4.0)
        v_clone = v.clone()
        current_clone = current.clone()
        lif_terms(v, current, v_rest=-65.0, resistance=1.5, tau=20.0, dt=0.5)
        assert torch.equal(v, v_clone)
        assert torch.equal(current, current_clone)

    def test_per_neuron_tensor_parameters(self) -> None:
        v = t(-70.0, -50.0)
        resistance = t(1.0, 3.0)
        terms = lif_terms(v, 2.0, v_rest=-65.0, resistance=resistance, tau=20.0, dt=0.5)
        assert terms["input"].tolist() == pytest.approx([0.05, 0.15])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(RuntimeError):
            lif_terms(
                t(-70.0, -60.0), t(1.0, 2.0, 3.0), v_rest=-65.0, resistance=1.0, tau=20.0, dt=0.5
            )


class TestElifTerms:
    def test_hand_computed_values_case_1(self) -> None:
        # leak: (-65.0 - (-50.0)) * 1.0 / 20.0 = -15.0 * 0.05 = -0.75
        # input: 1.0 * 2.0 * 1.0 / 20.0 = 2.0 * 0.05 = 0.1
        # spike_initiation: 2.0 * exp(0.0) * 1.0 / 20.0 = 2.0 * 0.05 = 0.1
        out = elif_terms(
            t(-50.0),
            2.0,
            v_rest=-65.0,
            resistance=1.0,
            delta=2.0,
            theta_rh=-50.0,
            tau=20.0,
            dt=1.0,
        )
        assert out["leak"].tolist() == pytest.approx([-0.75])
        assert out["input"].tolist() == pytest.approx([0.1])
        assert out["spike_initiation"].tolist() == pytest.approx([0.1])

    def test_hand_computed_values_case_2(self) -> None:
        # leak: (-60.0 - (-60.0)) * 0.5 / 10.0 = 0.0
        # input: 2.0 * 0.0 * 0.5 / 10.0 = 0.0
        # spike_initiation: 5.0 * exp((-60 - (-55)) / 5) * 0.5 / 10 = 0.25 * exp(-1)
        out = elif_terms(
            t(-60.0),
            0.0,
            v_rest=-60.0,
            resistance=2.0,
            delta=5.0,
            theta_rh=-55.0,
            tau=10.0,
            dt=0.5,
        )
        assert out["leak"].tolist() == pytest.approx([0.0])
        assert out["input"].tolist() == pytest.approx([0.0])
        assert out["spike_initiation"].tolist() == pytest.approx([0.25 * math.exp(-1.0)])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_terms_sum_equals_euler_step(self, dtype: torch.dtype) -> None:
        v = t(-70.0, -55.0, -48.0, dtype=dtype)
        current = t(1.0, 2.0, 0.5, dtype=dtype)
        terms = elif_terms(
            v,
            current,
            v_rest=-65.0,
            resistance=1.5,
            delta=2.0,
            theta_rh=-50.0,
            tau=15.0,
            dt=0.5,
        )
        deriv = lif_derivative(v, current, v_rest=-65.0, resistance=1.5) + exponential_boost(
            v, delta=2.0, theta_rh=-50.0
        )
        expected = euler_step(v, deriv, tau=15.0, dt=0.5)
        v_next = v + sum(terms.values())
        atol = 1e-6 if dtype == torch.float32 else 1e-12
        assert torch.allclose(v_next, expected, atol=atol)

    def test_matches_network_behavior(self) -> None:
        net = Network(dt=0.5, dtype=torch.float64)
        group = NeuronGroup(
            net,
            1,
            [
                ELIF(
                    tau=15.0,
                    threshold=0.0,
                    v_reset=-70.0,
                    v_rest=-65.0,
                    resistance=1.5,
                    delta=2.0,
                    theta_rh=-50.0,
                    v_init=-55.0,
                )
            ],
        )
        net.initialize()
        group.I.fill_(2.0)
        v_before = group.v.clone()
        terms = elif_terms(
            v_before,
            group.I,
            v_rest=group.v_rest,
            resistance=group.resistance,
            delta=group.model.delta,
            theta_rh=group.model.theta_rh,
            tau=group.tau,
            dt=net.dt,
        )
        net.step()
        assert group.v.item() == pytest.approx((v_before + sum(terms.values())).item(), abs=1e-12)

    def test_broadcasting_over_vector_of_voltages(self) -> None:
        v = t(-80.0, -70.0, -60.0, -50.0, -40.0)
        terms = elif_terms(
            v,
            1.5,
            v_rest=-65.0,
            resistance=1.0,
            delta=2.0,
            theta_rh=-50.0,
            tau=20.0,
            dt=0.5,
        )
        assert terms["leak"].shape == (5,)
        assert terms["input"].shape == (5,)
        assert terms["spike_initiation"].shape == (5,)

    def test_inputs_not_mutated(self) -> None:
        v = t(-60.0, -45.0)
        current = t(1.0, 3.0)
        v_clone = v.clone()
        current_clone = current.clone()
        elif_terms(
            v,
            current,
            v_rest=-65.0,
            resistance=1.0,
            delta=2.0,
            theta_rh=-50.0,
            tau=20.0,
            dt=0.5,
        )
        assert torch.equal(v, v_clone)
        assert torch.equal(current, current_clone)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(RuntimeError):
            elif_terms(
                t(-70.0, -60.0),
                t(1.0, 2.0, 3.0),
                v_rest=-65.0,
                resistance=1.0,
                delta=2.0,
                theta_rh=-50.0,
                tau=20.0,
                dt=0.5,
            )


class TestAdexTerms:
    def test_hand_computed_values_case_1(self) -> None:
        # leak: (-65.0 - (-50.0)) * 1.0 / 20.0 = -15.0 * 0.05 = -0.75
        # input: 2.0 * 2.0 * 1.0 / 20.0 = 4.0 * 0.05 = 0.2
        # spike_initiation: 2.0 * exp(0.0) * 1.0 / 20.0 = 2.0 * 0.05 = 0.1
        # adaptation: -2.0 * 1.5 * 1.0 / 20.0 = -3.0 * 0.05 = -0.15
        out = adex_terms(
            t(-50.0),
            2.0,
            1.5,
            v_rest=-65.0,
            resistance=2.0,
            delta=2.0,
            theta_rh=-50.0,
            tau=20.0,
            dt=1.0,
        )
        assert out["leak"].tolist() == pytest.approx([-0.75])
        assert out["input"].tolist() == pytest.approx([0.2])
        assert out["spike_initiation"].tolist() == pytest.approx([0.1])
        assert out["adaptation"].tolist() == pytest.approx([-0.15])

    def test_hand_computed_values_case_2(self) -> None:
        # leak: (-65.0 - (-70.0)) * 2.0 / 10.0 = 5.0 * 0.2 = 1.0
        # input: 1.0 * 0.0 * 2.0 / 10.0 = 0.0
        # adaptation: -1.0 * 0.5 * 2.0 / 10.0 = -0.1
        out = adex_terms(
            t(-70.0),
            0.0,
            0.5,
            v_rest=-65.0,
            resistance=1.0,
            delta=1.0,
            theta_rh=-50.0,
            tau=10.0,
            dt=2.0,
        )
        assert out["leak"].tolist() == pytest.approx([1.0])
        assert out["input"].tolist() == pytest.approx([0.0])
        assert out["adaptation"].tolist() == pytest.approx([-0.1])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_terms_sum_equals_euler_step(self, dtype: torch.dtype) -> None:
        v = t(-70.0, -55.0, -48.0, dtype=dtype)
        current = t(1.0, 2.0, 0.5, dtype=dtype)
        omega = t(0.5, 1.0, 2.0, dtype=dtype)
        terms = adex_terms(
            v,
            current,
            omega,
            v_rest=-65.0,
            resistance=1.5,
            delta=2.0,
            theta_rh=-50.0,
            tau=15.0,
            dt=0.5,
        )
        deriv = (
            lif_derivative(v, current, v_rest=-65.0, resistance=1.5)
            + exponential_boost(v, delta=2.0, theta_rh=-50.0)
            - 1.5 * omega
        )
        expected = euler_step(v, deriv, tau=15.0, dt=0.5)
        v_next = v + sum(terms.values())
        atol = 1e-6 if dtype == torch.float32 else 1e-12
        assert torch.allclose(v_next, expected, atol=atol)

    def test_matches_network_behavior(self) -> None:
        net = Network(dt=0.5, dtype=torch.float64)
        group = NeuronGroup(
            net,
            1,
            [
                AdaptiveELIF(
                    tau=15.0,
                    threshold=0.0,
                    v_reset=-70.0,
                    v_rest=-65.0,
                    resistance=1.5,
                    delta=2.0,
                    theta_rh=-50.0,
                    alpha=0.5,
                    beta=2.0,
                    tau_w=30.0,
                    omega_init=1.0,
                    v_init=-55.0,
                )
            ],
        )
        net.initialize()
        group.I.fill_(2.0)
        v_before = group.v.clone()
        omega_before = group.omega.clone()
        terms = adex_terms(
            v_before,
            group.I,
            omega_before,
            v_rest=group.v_rest,
            resistance=group.resistance,
            delta=group.model.delta,
            theta_rh=group.model.theta_rh,
            tau=group.tau,
            dt=net.dt,
        )
        net.step()
        assert group.v.item() == pytest.approx((v_before + sum(terms.values())).item(), abs=1e-12)

    def test_broadcasting_over_vector_of_voltages(self) -> None:
        v = t(-80.0, -70.0, -60.0, -50.0)
        terms = adex_terms(
            v,
            2.0,
            0.5,
            v_rest=-65.0,
            resistance=1.0,
            delta=2.0,
            theta_rh=-50.0,
            tau=20.0,
            dt=0.5,
        )
        assert terms["leak"].shape == (4,)
        assert terms["input"].shape == (4,)
        assert terms["spike_initiation"].shape == (4,)
        assert terms["adaptation"].shape == (4,)

    def test_inputs_not_mutated(self) -> None:
        v = t(-60.0, -45.0)
        current = t(1.0, 3.0)
        omega = t(0.2, 0.4)
        v_clone = v.clone()
        current_clone = current.clone()
        omega_clone = omega.clone()
        adex_terms(
            v,
            current,
            omega,
            v_rest=-65.0,
            resistance=1.0,
            delta=2.0,
            theta_rh=-50.0,
            tau=20.0,
            dt=0.5,
        )
        assert torch.equal(v, v_clone)
        assert torch.equal(current, current_clone)
        assert torch.equal(omega, omega_clone)

    def test_per_neuron_tensor_parameters(self) -> None:
        v = t(-70.0, -50.0)
        omega = t(0.5, 1.5)
        terms = adex_terms(
            v,
            2.0,
            omega,
            v_rest=-65.0,
            resistance=2.0,
            delta=2.0,
            theta_rh=-50.0,
            tau=20.0,
            dt=0.5,
        )
        # adaptation: -2.0 * omega * 0.5 / 20.0 = -omega * 0.05 -> [-0.025, -0.075]
        assert terms["adaptation"].tolist() == pytest.approx([-0.025, -0.075])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(RuntimeError):
            adex_terms(
                t(-70.0, -60.0),
                t(1.0, 2.0),
                t(0.1, 0.2, 0.3),
                v_rest=-65.0,
                resistance=1.0,
                delta=2.0,
                theta_rh=-50.0,
                tau=20.0,
                dt=0.5,
            )


class TestIzhikevichTerms:
    def test_hand_computed_values_substeps_1(self) -> None:
        # v = -65.0, u = -13.0, current = 10.0, dt = 1.0, substeps = 1, h = 1.0
        # quadratic: 1.0 * 0.04 * (-65.0)^2 = 0.04 * 4225.0 = 169.0
        # linear: 1.0 * (5.0 * (-65.0) + 140.0) = -325.0 + 140.0 = -185.0
        # recovery: -1.0 * (-13.0) = 13.0
        # input: 1.0 * 10.0 = 10.0
        out = izhikevich_terms(
            t(-65.0),
            t(-13.0),
            10.0,
            dt=1.0,
            substeps=1,
        )
        assert out["quadratic"].tolist() == pytest.approx([169.0])
        assert out["linear"].tolist() == pytest.approx([-185.0])
        assert out["recovery"].tolist() == pytest.approx([13.0])
        assert out["input"].tolist() == pytest.approx([10.0])

    def test_hand_computed_values_substeps_2(self) -> None:
        # v = -65.0, u = -13.0, current = 10.0, dt = 1.0, substeps = 2, h = 0.5
        # substep 1: v=-65.0 -> quad=84.5, lin=-92.5, rec=6.5, inp=5.0 -> v_mid=-61.5
        # substep 2: v=-61.5 -> quad=75.645, lin=-83.75, rec=6.5, inp=5.0
        # total quad: 84.5 + 75.645 = 160.145
        # total lin: -92.5 + -83.75 = -176.25
        # total rec: 6.5 + 6.5 = 13.0
        # total inp: 5.0 + 5.0 = 10.0
        out = izhikevich_terms(
            t(-65.0),
            t(-13.0),
            10.0,
            dt=1.0,
            substeps=2,
        )
        assert out["quadratic"].tolist() == pytest.approx([160.145])
        assert out["linear"].tolist() == pytest.approx([-176.25])
        assert out["recovery"].tolist() == pytest.approx([13.0])
        assert out["input"].tolist() == pytest.approx([10.0])

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_terms_sum_equals_step(self, dtype: torch.dtype) -> None:
        v = t(-70.0, -65.0, -50.0, dtype=dtype)
        u = t(-14.0, -13.0, -10.0, dtype=dtype)
        current = t(0.0, 5.0, 10.0, dtype=dtype)
        terms = izhikevich_terms(v, u, current, dt=1.0, substeps=2)
        expected_v, _ = izhikevich_step(v, u, current, a=0.02, b=0.2, dt=1.0, substeps=2)
        v_next = v + sum(terms.values())
        atol = 1e-4 if dtype == torch.float32 else 1e-10
        assert torch.allclose(v_next, expected_v, atol=atol)

    def test_matches_network_behavior(self) -> None:
        net = Network(dt=1.0, dtype=torch.float64)
        group = NeuronGroup(
            net,
            1,
            [Izhikevich(a=0.02, b=0.2, c=-65.0, d=8.0, v_init=-65.0, u_init=-13.0)],
        )
        net.initialize()
        group.I.fill_(10.0)
        v_before = group.v.clone()
        u_before = group.u.clone()
        terms = izhikevich_terms(v_before, u_before, group.I, dt=net.dt, substeps=2)
        net.step()
        assert group.v.item() == pytest.approx((v_before + sum(terms.values())).item(), abs=1e-10)

    def test_broadcasting_over_vector_of_voltages(self) -> None:
        v = t(-80.0, -70.0, -60.0, -50.0)
        terms = izhikevich_terms(v, -13.0, 5.0, dt=1.0, substeps=2)
        assert terms["quadratic"].shape == (4,)
        assert terms["linear"].shape == (4,)
        assert terms["recovery"].shape == (4,)
        assert terms["input"].shape == (4,)

    def test_inputs_not_mutated(self) -> None:
        v = t(-65.0, -50.0)
        u = t(-13.0, -10.0)
        current = t(5.0, 10.0)
        v_clone = v.clone()
        u_clone = u.clone()
        current_clone = current.clone()
        izhikevich_terms(v, u, current, dt=1.0, substeps=2)
        assert torch.equal(v, v_clone)
        assert torch.equal(u, u_clone)
        assert torch.equal(current, current_clone)

    def test_per_neuron_tensor_parameters(self) -> None:
        v = t(-65.0, -55.0)
        u = t(-13.0, -11.0)
        current = t(5.0, 10.0)
        terms = izhikevich_terms(v, u, current, dt=1.0, substeps=1)
        # substeps=1: recovery is -h * u = -1.0 * u
        assert terms["recovery"].tolist() == pytest.approx([13.0, 11.0])
        # input is h * current = 1.0 * current
        assert terms["input"].tolist() == pytest.approx([5.0, 10.0])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(RuntimeError):
            izhikevich_terms(t(-65.0, -50.0), t(-13.0, -10.0, -8.0), 5.0, dt=1.0)
