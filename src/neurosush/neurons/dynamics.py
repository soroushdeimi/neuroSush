"""Pure equations of the leaky integrate-and-fire family.

Every function returns new tensors and leaves its inputs unchanged; the derivative
functions return ``tau * dv/dt``.
"""

from __future__ import annotations

import torch

Scalar = float | torch.Tensor


def lif_derivative(v: Scalar, current: Scalar, *, v_rest: float, resistance: float) -> Scalar:
    """Leak plus input drive, ``tau * dv/dt = (v_rest - v) + R * I``."""
    return (v_rest - v) + resistance * current


def exponential_boost(v: Scalar, *, delta: float, theta_rh: float) -> Scalar:
    """Spike-initiation term of the exponential LIF, ``delta * exp((v - theta_rh) / delta)``."""
    return delta * torch.exp((v - theta_rh) / delta)


def euler_step(v: Scalar, tau_dv_dt: Scalar, *, tau: float, dt: float) -> Scalar:
    """One forward Euler step, ``v + tau_dv_dt * dt / tau``."""
    return v + tau_dv_dt * (dt / tau)


def fire(v: Scalar, *, threshold: Scalar, v_reset: Scalar) -> tuple[torch.Tensor, torch.Tensor]:
    """Threshold crossing: the spikes and the voltage with spiking neurons reset."""
    spikes = v >= threshold
    v_after = torch.where(spikes, torch.as_tensor(v_reset, dtype=v.dtype, device=v.device), v)
    return spikes, v_after


def adaptation_step(
    omega: Scalar,
    v: Scalar,
    spikes: torch.Tensor,
    *,
    v_rest: float,
    alpha: float,
    beta: float,
    tau_w: float,
    dt: float,
) -> Scalar:
    """Adaptation current update of the adaptive exponential LIF.

    ``tau_w * d(omega)/dt = alpha * (v - v_rest) - omega``, plus a jump of ``beta`` per spike.
    """
    return omega + (alpha * (v - v_rest) - omega) * (dt / tau_w) + beta * spikes.to(omega.dtype)
