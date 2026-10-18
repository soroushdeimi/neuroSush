"""Pure equations of the leaky integrate-and-fire family and neuron dynamics.

Every function returns new tensors and leaves its inputs unchanged; the derivative
functions return ``tau * dv/dt``, and the term decomposition functions return dictionaries
of voltage changes per term for one Euler step.
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
    v: torch.Tensor,
    *,
    threshold: Scalar,
    v_reset: Scalar,
    blocked: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Threshold crossing: the spikes and the voltage with spiking neurons reset.

    Neurons in ``blocked`` (a boolean mask) cannot spike: their voltage is left as it is.
    """
    spikes = v >= threshold
    if blocked is not None:
        spikes = spikes & ~blocked
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
    v: torch.Tensor,
    u: torch.Tensor,
    *,
    v_peak: Scalar,
    c: Scalar,
    d: Scalar,
    blocked: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Spike where ``v >= v_peak``; those neurons get ``v = c`` and ``u += d``.

    Returns:
        ``(spikes, v, u)``.
    """
    spikes, v_after = fire(v, threshold=v_peak, v_reset=c, blocked=blocked)
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


def lif_terms(
    v: torch.Tensor,
    current: Scalar,
    *,
    v_rest: Scalar,
    resistance: Scalar,
    tau: Scalar,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Decompose one LIF Euler step into leak and input terms.

    Args:
        v: Membrane voltage before the step.
        current: Input current drive.
        v_rest: Resting membrane potential.
        resistance: Membrane resistance.
        tau: Membrane time constant.
        dt: Integration time step.

    Returns:
        Dictionary with ``leak`` and ``input`` voltage increments.
    """
    step = dt / torch.as_tensor(tau, dtype=v.dtype, device=v.device)
    vr = torch.as_tensor(v_rest, dtype=v.dtype, device=v.device)
    r = torch.as_tensor(resistance, dtype=v.dtype, device=v.device)
    c = torch.as_tensor(current, dtype=v.dtype, device=v.device)
    terms = {
        "leak": (vr - v) * step,
        "input": (r * c) * step,
    }
    broadcasted = torch.broadcast_tensors(*terms.values())  # type: ignore[no-untyped-call]
    return {k: broadcasted[i] for i, k in enumerate(terms)}


def elif_terms(
    v: torch.Tensor,
    current: Scalar,
    *,
    v_rest: Scalar,
    resistance: Scalar,
    delta: float,
    theta_rh: float,
    tau: Scalar,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Decompose one ELIF Euler step into leak, input, and spike initiation terms.

    Args:
        v: Membrane voltage before the step.
        current: Input current drive.
        v_rest: Resting membrane potential.
        resistance: Membrane resistance.
        delta: Exponential boost scale.
        theta_rh: Rheobase threshold of the exponential term.
        tau: Membrane time constant.
        dt: Integration time step.

    Returns:
        Dictionary with ``leak``, ``input``, and ``spike_initiation`` voltage increments.
    """
    terms = lif_terms(v, current, v_rest=v_rest, resistance=resistance, tau=tau, dt=dt)
    step = dt / torch.as_tensor(tau, dtype=v.dtype, device=v.device)
    terms["spike_initiation"] = exponential_boost(v, delta=delta, theta_rh=theta_rh) * step
    broadcasted = torch.broadcast_tensors(*terms.values())  # type: ignore[no-untyped-call]
    return {k: broadcasted[i] for i, k in enumerate(terms)}


def adex_terms(
    v: torch.Tensor,
    current: Scalar,
    omega: Scalar,
    *,
    v_rest: Scalar,
    resistance: Scalar,
    delta: float,
    theta_rh: float,
    tau: Scalar,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Decompose one AdEx Euler step into leak, input, spike initiation, and adaptation terms.

    Args:
        v: Membrane voltage before the step.
        current: Input current drive.
        omega: Adaptation variable before the step.
        v_rest: Resting membrane potential.
        resistance: Membrane resistance.
        delta: Exponential boost scale.
        theta_rh: Rheobase threshold of the exponential term.
        tau: Membrane time constant.
        dt: Integration time step.

    Returns:
        Dictionary with ``leak``, ``input``, ``spike_initiation``, and ``adaptation`` increments.
    """
    terms = elif_terms(
        v,
        current,
        v_rest=v_rest,
        resistance=resistance,
        delta=delta,
        theta_rh=theta_rh,
        tau=tau,
        dt=dt,
    )
    step = dt / torch.as_tensor(tau, dtype=v.dtype, device=v.device)
    r = torch.as_tensor(resistance, dtype=v.dtype, device=v.device)
    w = torch.as_tensor(omega, dtype=v.dtype, device=v.device)
    terms["adaptation"] = -(r * w) * step
    broadcasted = torch.broadcast_tensors(*terms.values())  # type: ignore[no-untyped-call]
    return {k: broadcasted[i] for i, k in enumerate(terms)}


def izhikevich_terms(
    v: torch.Tensor,
    u: torch.Tensor,
    current: Scalar,
    *,
    dt: float,
    substeps: int = 2,
) -> dict[str, torch.Tensor]:
    """Decompose one Izhikevich step into quadratic, linear, recovery, and input terms.

    Args:
        v: Membrane voltage before the step.
        u: Membrane recovery variable before the step.
        current: Input current drive.
        dt: Integration time step.
        substeps: Number of Euler sub-steps for membrane voltage integration.

    Returns:
        Dictionary with ``quadratic``, ``linear``, ``recovery``, and ``input`` increments.
    """
    u_tensor = torch.as_tensor(u, dtype=v.dtype, device=v.device)
    current_t = torch.as_tensor(current, dtype=v.dtype, device=v.device)
    v_b, u_b, current_b = torch.broadcast_tensors(v, u_tensor, current_t)  # type: ignore[no-untyped-call]

    h = dt / substeps
    v_curr = v_b
    quad = torch.zeros_like(v_b)
    lin = torch.zeros_like(v_b)
    rec = torch.zeros_like(v_b)
    inp = torch.zeros_like(v_b)

    for _ in range(substeps):
        step_quad = h * (0.04 * v_curr * v_curr)
        step_lin = h * (5.0 * v_curr + 140.0)
        step_rec = -h * u_b
        step_inp = h * current_b
        quad = quad + step_quad
        lin = lin + step_lin
        rec = rec + step_rec
        inp = inp + step_inp
        v_curr = v_curr + h * (0.04 * v_curr * v_curr + 5.0 * v_curr + 140.0 - u_b + current_b)

    return {
        "quadratic": quad,
        "linear": lin,
        "recovery": rec,
        "input": inp,
    }
