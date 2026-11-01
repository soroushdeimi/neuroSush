"""Tests for theta phase regression and phase precession classification."""

from __future__ import annotations

import math

import pytest
import torch

from neurosush.spatial import circular_linear_regression, phase_slope_class


class TestPhase:
    @pytest.fixture(autouse=True)
    def _single_thread(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        yield
        torch.set_num_threads(threads)

    def test_wrapped_theta_regression_and_direction(self):
        position = torch.linspace(0, 39, 120)
        for direction in (-1, 1):
            for slope in (0.0, -0.4, 0.75):
                phase = (slope * direction * position + 1.2) % (2 * math.pi)
                result = circular_linear_regression(position, phase, direction=direction)
                assert result.slope == pytest.approx(slope, abs=0.0011)
                assert result.intercept == pytest.approx(1.2, abs=0.002)
                assert result.concentration > 0.999
                assert not result.at_bound

    def test_metric_validation(self):
        with pytest.raises(ValueError, match="variation"):
            circular_linear_regression(torch.ones(4), torch.ones(4))

    def test_phase_slope_class(self):
        assert phase_slope_class(0.1) == "locked"
        assert phase_slope_class(0.2) == "locked"
        assert phase_slope_class(-0.2) == "locked"
        assert phase_slope_class(0.4) == "precession"
        assert phase_slope_class(-0.5) == "precession"
        assert phase_slope_class(0.6) == "independent"
        assert phase_slope_class(1.5) == "independent"
        assert phase_slope_class(-0.8) == "independent"

        with pytest.raises(ValueError, match="slope must be finite"):
            phase_slope_class(float("nan"))
        with pytest.raises(ValueError, match="slope must be finite"):
            phase_slope_class(torch.tensor(float("nan")))
