"""Tests for neurosush.schedule."""

from __future__ import annotations

import pytest

from neurosush.schedule import linear_ramp, step_decay


def test_linear_ramp() -> None:
    # at 0
    assert linear_ramp(0, 10, 0.0, 1.0) == 0.0
    # at middle
    assert linear_ramp(5, 10, 0.0, 1.0) == 0.5
    # at end
    assert linear_ramp(10, 10, 0.0, 1.0) == 1.0
    # beyond end
    assert linear_ramp(15, 10, 0.0, 1.0) == 1.0
    # negative step
    assert linear_ramp(-2, 10, 0.0, 1.0) == 0.0
    # invalid steps
    with pytest.raises(ValueError, match="steps must be > 0"):
        linear_ramp(5, 0, 0.0, 1.0)
    with pytest.raises(ValueError, match="steps must be > 0"):
        linear_ramp(5, -5, 0.0, 1.0)


def test_step_decay() -> None:
    # at 0
    assert step_decay(0, base=1.0, rate=0.5, every=10) == 1.0
    # within first interval
    assert step_decay(5, base=1.0, rate=0.5, every=10) == 1.0
    # at interval boundary
    assert step_decay(10, base=1.0, rate=0.5, every=10) == 0.5
    # at second interval
    assert step_decay(25, base=1.0, rate=0.5, every=10) == 0.25
    # negative step
    with pytest.raises(ValueError, match="step must be non-negative"):
        step_decay(-1, base=1.0, rate=0.5, every=10)
    # invalid rate
    with pytest.raises(ValueError, match="rate must be in"):
        step_decay(0, base=1.0, rate=0.0, every=10)
    with pytest.raises(ValueError, match="rate must be in"):
        step_decay(0, base=1.0, rate=1.5, every=10)
    # invalid every
    with pytest.raises(ValueError, match="every must be >= 1"):
        step_decay(0, base=1.0, rate=0.5, every=0)
