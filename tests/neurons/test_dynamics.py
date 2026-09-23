import math

import pytest
import torch

from neurosush.neurons.dynamics import (
    adaptation_step,
    euler_step,
    exponential_boost,
    fire,
    lif_derivative,
)


def t(*values):
    return torch.tensor(values, dtype=torch.float64)


def test_lif_derivative():
    # tau * dv/dt = (v_rest - v) + R * I
    out = lif_derivative(t(-70.0, -50.0), t(2.0, 0.0), v_rest=-65.0, resistance=3.0)
    assert out.tolist() == [5.0 + 6.0, -15.0]


def test_exponential_boost():
    out = exponential_boost(t(-50.0, -60.0), delta=2.0, theta_rh=-55.0)
    assert out.tolist() == pytest.approx([2.0 * math.exp(2.5), 2.0 * math.exp(-2.5)])


def test_euler_step():
    assert euler_step(t(-65.0), t(10.0), tau=20.0, dt=0.5).tolist() == [-65.0 + 10.0 * 0.5 / 20.0]


def test_fire_resets_crossing_neurons_only():
    v = t(-49.0, -50.0, -51.0)
    spikes, v_after = fire(v, threshold=t(-50.0, -50.0, -50.0), v_reset=-70.0)
    assert spikes.tolist() == [True, True, False]
    assert spikes.dtype == torch.bool
    assert v_after.tolist() == [-70.0, -70.0, -51.0]


def test_fire_does_not_modify_input():
    v = t(0.0, 2.0)
    fire(v, threshold=1.0, v_reset=-1.0)
    assert v.tolist() == [0.0, 2.0]


def test_adaptation_step_subthreshold():
    # omega + dt / tau_w * (alpha * (v - v_rest) - omega)
    out = adaptation_step(
        t(1.0),
        t(-60.0),
        torch.tensor([False]),
        v_rest=-65.0,
        alpha=0.5,
        beta=3.0,
        tau_w=10.0,
        dt=2.0,
    )
    assert out.tolist() == pytest.approx([1.0 + 0.2 * (2.5 - 1.0)])


def test_adaptation_step_spike_adds_beta():
    out = adaptation_step(
        t(0.0, 0.0),
        t(-65.0, -65.0),
        torch.tensor([True, False]),
        v_rest=-65.0,
        alpha=0.5,
        beta=3.0,
        tau_w=10.0,
        dt=0.1,
    )
    assert out.tolist() == [3.0, 0.0]
