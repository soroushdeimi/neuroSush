"""Pure equations of the leaky integrate-and-fire family.

Every function returns new tensors and leaves its inputs unchanged; the derivative
functions return ``tau * dv/dt``.
"""

from __future__ import annotations

from typing import overload

import torch

Scalar = float | torch.Tensor


@overload
def lif_derivative(
    v: torch.Tensor, current: Scalar, *, v_rest: float, resistance: float
) -> torch.Tensor: ...


@overload
def lif_derivative(
    v: float, current: torch.Tensor, *, v_rest: float, resistance: float
) -> torch.Tensor: ...


@overload
def lif_derivative(v: float, current: float, *, v_rest: float, resistance: float) -> float: ...


def lif_derivative(v: Scalar, current: Scalar, *, v_rest: float, resistance: float) -> Scalar:
    """Leak plus input drive, ``tau * dv/dt = (v_rest - v) + R * I``."""
    return (v_rest - v) + resistance * current


def exponential_boost(v: torch.Tensor, *, delta: float, theta_rh: float) -> torch.Tensor:
    """Spike-initiation term of the exponential LIF, ``delta * exp((v - theta_rh) / delta)``."""
    return delta * torch.exp((v - theta_rh) / delta)


@overload
def euler_step(v: torch.Tensor, tau_dv_dt: Scalar, *, tau: float, dt: float) -> torch.Tensor: ...


@overload
def euler_step(v: float, tau_dv_dt: torch.Tensor, *, tau: float, dt: float) -> torch.Tensor: ...


@overload
def euler_step(v: float, tau_dv_dt: float, *, tau: float, dt: float) -> float: ...


def euler_step(v: Scalar, tau_dv_dt: Scalar, *, tau: float, dt: float) -> Scalar:
    """One forward Euler step, ``v + tau_dv_dt * dt / tau``."""
    return v + tau_dv_dt * (dt / tau)


def fire(
    v: torch.Tensor, *, threshold: Scalar, v_reset: Scalar
) -> tuple[torch.Tensor, torch.Tensor]:
    """Threshold crossing: the spikes and the voltage with spiking neurons reset."""
    spikes = v >= threshold
    if isinstance(v_reset, torch.Tensor):
        v_after = torch.where(spikes, v_reset, v)
    else:
        # masked_fill takes the scalar directly: no host-to-device copy every step
        v_after = v.masked_fill(spikes, v_reset)
    return spikes, v_after


def adaptation_step(
    omega: torch.Tensor,
    v: Scalar,
    spikes: torch.Tensor,
    *,
    v_rest: float,
    alpha: Scalar,
    beta: Scalar,
    tau_w: Scalar,
    dt: float,
) -> torch.Tensor:
    """Adaptation current update of the adaptive exponential LIF.

    ``tau_w * d(omega)/dt = alpha * (v - v_rest) - omega``, plus a jump of ``beta`` per spike.
    """
    return omega + (alpha * (v - v_rest) - omega) * (dt / tau_w) + beta * spikes.to(omega.dtype)


def izhikevich_step(
    v: torch.Tensor,
    u: torch.Tensor,
    current: Scalar,
    *,
    a: Scalar,
    b: Scalar,
    dt: float,
    substeps: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of the Izhikevich (2003) neuron, without the spike reset.

    ``v' = 0.04 v^2 + 5 v + 140 - u + I`` is advanced by ``substeps`` forward Euler steps of
    ``dt / substeps`` with ``I`` and ``u`` held (two half steps for ``dt = 1`` ms, as in the
    paper), then ``u' = a (b v - u)`` by one Euler step of ``dt`` using the updated ``v``.
    """
    h = dt / substeps
    for _ in range(substeps):
        v = v + h * (0.04 * v * v + 5.0 * v + 140.0 - u + current)
    u = u + dt * a * (b * v - u)
    return v, u


def izhikevich_reset(
    v: torch.Tensor, u: torch.Tensor, *, v_peak: Scalar, c: Scalar, d: Scalar
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Spike where ``v >= v_peak``; those neurons get ``v = c`` and ``u += d``.

    Returns:
        ``(spikes, v, u)``.
    """
    spikes, v_after = fire(v, threshold=v_peak, v_reset=c)
    return spikes, v_after, u + d * spikes.to(u.dtype)


def decaying_current_step(
    current: torch.Tensor, spikes: torch.Tensor, *, amplitude: Scalar, tau: Scalar, dt: float
) -> torch.Tensor:
    """``current * (1 - dt / tau) + amplitude * spikes``: a current that jumps at each spike."""
    return current * (1 - dt / tau) + amplitude * spikes.to(current.dtype)


def poisson_increment_moments(
    *, count: float, rate: float, jump: float, dt: float
) -> tuple[float, float]:
    """Mean and variance of ``jump * Poisson(count * rate * dt)``."""
    lam = count * rate * dt
    return jump * lam, jump * jump * lam


def correlated_pair_correlation(*, correlation: float, p_mother: float) -> float:
    """Count correlation of two copies of a Bernoulli mother train, ``c (1 - p) / (1 - c p)``.

    Independent of the bin length; ``c`` as ``p = rate dt / c`` goes to 0 (Poisson mother).
    """
    return correlation * (1 - p_mother) / (1 - correlation * p_mother)
