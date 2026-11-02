"""Tests for surrogate spike gradients and LIF dynamics."""

from __future__ import annotations

import math

import pytest
import torch

from neurosush.neurons.surrogate import lif_step, spike


def test_forward_threshold():
    threshold = 1.0
    v_test = torch.tensor([threshold - 1e-6, threshold, threshold + 1e-6])
    out = spike(v_test, threshold=threshold, width=2.0, kind="arctan")
    torch.testing.assert_close(out, torch.tensor([0.0, 0.0, 1.0]))


def test_derivatives_both_kinds():
    for kind in ("arctan", "triangle"):
        v = torch.tensor([-1.0, 0.0, 0.5, 1.0, 1.5], requires_grad=True)
        out = spike(v, threshold=1.0, width=2.0, kind=kind)
        out.sum().backward()
        g = v.grad
        assert g is not None
        assert torch.isfinite(g).all()
        assert (g >= 0).all()


def test_derivative_values_and_properties():
    width = 2.0
    v_arctan = torch.tensor([1.0, 2.0], requires_grad=True)
    out_a = spike(v_arctan, threshold=1.0, width=width, kind="arctan")
    out_a.sum().backward()
    g_a = v_arctan.grad
    assert g_a is not None
    assert float(g_a[0]) == pytest.approx(width / 2.0)
    expected_x1 = width / (1.0 + (math.pi * width * (1.0) / 2.0) ** 2) / 2.0
    assert float(g_a[1]) == pytest.approx(expected_x1)

    v_tri = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    out_t = spike(v_tri, threshold=1.0, width=width, kind="triangle")
    out_t.sum().backward()
    g_t = v_tri.grad
    assert g_t is not None
    assert float(g_t[0]) == pytest.approx(1.0)
    assert float(g_t[1]) == pytest.approx(0.5)
    assert float(g_t[2]) == pytest.approx(0.0)

    v_sym = torch.tensor([0.5, 1.5], requires_grad=True)
    out_sym = spike(v_sym, threshold=1.0, width=width, kind="triangle")
    out_sym.sum().backward()
    assert v_sym.grad is not None
    torch.testing.assert_close(v_sym.grad[0], v_sym.grad[1])


def test_chained_gradient():
    current = torch.tensor([1.5], requires_grad=True)
    v = torch.tensor([0.0])
    total_spikes = 0.0
    for _ in range(20):
        v, s = lif_step(
            v,
            current,
            tau=10.0,
            dt=1.0,
            v_rest=0.0,
            threshold=1.0,
            reset=0.0,
            width=2.0,
            kind="arctan",
        )
        total_spikes = total_spikes + s.sum()
    total_spikes.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad)
    assert float(current.grad.item()) != pytest.approx(0.0)


def test_lif_step_hand_computed():
    v = torch.tensor([0.5])
    current = torch.tensor([2.0])
    v_next, s = lif_step(
        v,
        current,
        tau=10.0,
        dt=1.0,
        v_rest=0.0,
        threshold=1.0,
        reset=0.0,
        resistance=1.0,
    )
    assert float(s) == 0.0
    assert float(v_next) == pytest.approx(0.65)

    v_spk = torch.tensor([0.9])
    curr_spk = torch.tensor([5.0])
    v_n2, s2 = lif_step(
        v_spk,
        curr_spk,
        tau=10.0,
        dt=1.0,
        v_rest=0.0,
        threshold=1.0,
        reset=0.0,
        resistance=1.0,
    )
    assert float(s2) == 1.0
    assert float(v_n2) == pytest.approx(0.0)


def test_batch_and_float64():
    v_b = torch.zeros(2, 3, dtype=torch.float64)
    curr_b = torch.ones(3, dtype=torch.float64)
    v_nb, s_b = lif_step(
        v_b,
        curr_b,
        tau=10.0,
        dt=1.0,
        v_rest=0.0,
        threshold=1.0,
        reset=0.0,
        kind="triangle",
    )
    assert v_nb.shape == (2, 3)
    assert s_b.shape == (2, 3)
    assert v_nb.dtype == torch.float64


def test_validation_errors():
    with pytest.raises(ValueError, match="width"):
        spike(torch.tensor([0.0]), width=0.0)

    with pytest.raises(ValueError, match="kind"):
        spike(torch.tensor([0.0]), kind="invalid")
