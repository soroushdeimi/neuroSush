"""Schedule helpers for ramps and decay rates over training steps."""

from __future__ import annotations


def linear_ramp(step: int | float, steps: int | float, start: float, end: float) -> float:
    """Linearly interpolate from ``start`` to ``end`` over ``steps`` steps.

    Args:
        step: Current training step (int or float).
        steps: Total number of steps for the ramp (> 0).
        start: Starting value at step 0.
        end: Ending value at step ``steps``.

    Returns:
        Interpolated value at the given step.
    """
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    progress = min(max(float(step) / float(steps), 0.0), 1.0)
    return float(start + (end - start) * progress)


def step_decay(step: int | float, base: float, rate: float, every: int) -> float:
    """Multiply base by rate once every ``every`` steps.

    Args:
        step: Current training step (non-negative int or float).
        base: Base value at step 0.
        rate: Multiplicative decay factor in (0, 1].
        every: Step interval for applying decay (>= 1).

    Returns:
        Decayed value at the given step.
    """
    if not (0.0 < rate <= 1.0):
        raise ValueError(f"rate must be in (0, 1], got {rate}")
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    s = int(step)
    return float(base * (rate ** (s // every)))
