"""Tests for theta oscillation current function."""

from __future__ import annotations

import pytest
import torch

from neurosush.neurons.oscillation import theta_current


def test_theta_current():
    alpha = 10.0
    freq = 8.0
    period = 1.0 / freq

    assert theta_current(0.0, alpha=alpha, freq=freq) == pytest.approx(-alpha)
    assert theta_current(1.0 / (4.0 * freq), alpha=alpha, freq=freq) == pytest.approx(-2.0 * alpha)
    assert theta_current(3.0 / (4.0 * freq), alpha=alpha, freq=freq) == pytest.approx(0.0)

    t_vals = torch.linspace(0, 2 * period, 100)
    vals = theta_current(t_vals, alpha=alpha, freq=freq)
    assert (vals <= 0.0).all()
    assert (vals >= -2.0 * alpha - 1e-6).all()
    vals_period = theta_current(t_vals + period, alpha=alpha, freq=freq)
    torch.testing.assert_close(vals, vals_period)


def test_theta_current_invalid_freq():
    with pytest.raises(ValueError, match="freq"):
        theta_current(0.0, freq=0.0)
    with pytest.raises(ValueError, match="freq"):
        theta_current(0.0, freq=-1.0)
