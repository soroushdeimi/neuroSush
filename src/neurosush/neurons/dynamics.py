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
    alpha: float,
    beta: float,
    tau_w: float,
    dt: float,
) -> torch.Tensor:
    """Adaptation current update of the adaptive exponential LIF.

    ``tau_w * d(omega)/dt = alpha * (v - v_rest) - omega``, plus a jump of ``beta`` per spike.
    """
    return omega + (alpha * (v - v_rest) - omega) * (dt / tau_w) + beta * spikes.to(omega.dtype)
