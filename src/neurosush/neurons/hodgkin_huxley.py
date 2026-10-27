"""Hodgkin-Huxley (1952) squid-axon neuron model with visible ion-channel state.

Units:
    Time: ms
    Voltage: mV
    Current density: µA/cm²
    Conductance: mS/cm²
    Capacitance: µF/cm²
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons.models import Fire
from neurosush.neurons.params import positive

_EPS = 1e-6


class GatePair(NamedTuple):
    """Pair of forward and backward transition rate tensors for one gate."""

    alpha: torch.Tensor
    beta: torch.Tensor


class GateRates(NamedTuple):
    """Transition rates for m, h, and n ion-channel gates."""

    m: GatePair
    h: GatePair
    n: GatePair


class GateValues(NamedTuple):
    """Values of the three ion-channel gates m, h, and n."""

    m: torch.Tensor
    h: torch.Tensor
    n: torch.Tensor


def gate_rates(v: torch.Tensor, *, phi: float = 1.0) -> GateRates:
    """Transition rates alpha and beta for the m, h, and n gates scaled by temperature factor phi.

    Args:
        v: Membrane voltage tensor in mV.
        phi: Temperature scaling factor (dimensionless).

    Returns:
        GateRates containing (alpha, beta) pairs for m, h, and n.
    """
    # Removable singularity at V = -40 mV via Taylor expansion of u / (1 - exp(-u)).
    u_m = (v + 40.0) / 10.0
    safe_u_m = torch.where(u_m.abs() < _EPS, torch.ones_like(u_m), u_m)
    taylor_m = 1.0 + 0.5 * u_m + (u_m**2) / 12.0
    alpha_m = torch.where(u_m.abs() < _EPS, taylor_m, -safe_u_m / torch.expm1(-safe_u_m))
    beta_m = 4.0 * torch.exp(-(v + 65.0) / 18.0)

    alpha_h = 0.07 * torch.exp(-(v + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + torch.exp(-(v + 35.0) / 10.0))

    # Removable singularity at V = -55 mV via Taylor expansion of 0.1 * u / (1 - exp(-u)).
    u_n = (v + 55.0) / 10.0
    safe_u_n = torch.where(u_n.abs() < _EPS, torch.ones_like(u_n), u_n)
    taylor_n = 0.1 * (1.0 + 0.5 * u_n + (u_n**2) / 12.0)
    alpha_n = torch.where(u_n.abs() < _EPS, taylor_n, 0.1 * (-safe_u_n / torch.expm1(-safe_u_n)))
    beta_n = 0.125 * torch.exp(-(v + 65.0) / 80.0)

    if phi != 1.0:
        alpha_m = phi * alpha_m
        beta_m = phi * beta_m
        alpha_h = phi * alpha_h
        beta_h = phi * beta_h
        alpha_n = phi * alpha_n
        beta_n = phi * beta_n

    return GateRates(
        m=GatePair(alpha=alpha_m, beta=beta_m),
        h=GatePair(alpha=alpha_h, beta=beta_h),
        n=GatePair(alpha=alpha_n, beta=beta_n),
    )


def gate_steady_state(v: torch.Tensor, *, phi: float = 1.0) -> GateValues:
    """Steady-state open probabilities x_inf = alpha / (alpha + beta) for gates m, h, and n.

    Args:
        v: Membrane voltage tensor in mV.
        phi: Temperature scaling factor.

    Returns:
        GateValues with steady-state open probabilities for m, h, and n.
    """
    rates = gate_rates(v, phi=phi)
    m_inf = rates.m.alpha / (rates.m.alpha + rates.m.beta)
    h_inf = rates.h.alpha / (rates.h.alpha + rates.h.beta)
    n_inf = rates.n.alpha / (rates.n.alpha + rates.n.beta)
    return GateValues(m=m_inf, h=h_inf, n=n_inf)


def gate_time_constant(v: torch.Tensor, *, phi: float = 1.0) -> GateValues:
    """Time constants tau = 1 / (alpha + beta) in ms for gates m, h, and n.

    Args:
        v: Membrane voltage tensor in mV.
        phi: Temperature scaling factor.

    Returns:
        GateValues with time constants in ms for m, h, and n.
    """
    rates = gate_rates(v, phi=phi)
    tau_m = 1.0 / (rates.m.alpha + rates.m.beta)
    tau_h = 1.0 / (rates.h.alpha + rates.h.beta)
    tau_n = 1.0 / (rates.n.alpha + rates.n.beta)
    return GateValues(m=tau_m, h=tau_h, n=tau_n)


def gating_step(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    dt: float,
) -> torch.Tensor:
    """Exact analytic integration of dx/dt = alpha * (1 - x) - beta * x over dt with frozen V.

    Args:
        x: Gate state tensor at the start of the step.
        alpha: Forward transition rate tensor.
        beta: Backward transition rate tensor.
        dt: Time step duration in ms.

    Returns:
        Updated gate state tensor after dt.
    """
    sum_rate = alpha + beta
    x_inf = alpha / sum_rate
    return x_inf + (x - x_inf) * torch.exp(-sum_rate * dt)


def hh_currents(
    v: torch.Tensor,
    m: torch.Tensor,
    h: torch.Tensor,
    n: torch.Tensor,
    *,
    g_na: float,
    g_k: float,
    g_leak: float,
    e_na: float,
    e_k: float,
    e_leak: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate sodium, potassium, and leak currents with outward positive convention.

    Args:
        v: Membrane voltage in mV.
        m: Sodium activation gate open probability.
        h: Sodium inactivation gate open probability.
        n: Potassium activation gate open probability.
        g_na: Maximum sodium conductance in mS/cm².
        g_k: Maximum potassium conductance in mS/cm².
        g_leak: Leak conductance in mS/cm².
        e_na: Sodium reversal potential in mV.
        e_k: Potassium reversal potential in mV.
        e_leak: Leak reversal potential in mV.

    Returns:
        Tuple of (i_na, i_k, i_leak) current densities in µA/cm².
    """
    i_na = g_na * (m**3) * h * (v - e_na)
    i_k = g_k * (n**4) * (v - e_k)
    i_leak = g_leak * (v - e_leak)
    return i_na, i_k, i_leak


def membrane_step(
    v: torch.Tensor,
    i_ext: torch.Tensor,
    i_na: torch.Tensor,
    i_k: torch.Tensor,
    i_leak: torch.Tensor,
    *,
    c: float,
    dt: float,
) -> torch.Tensor:
    """Forward Euler step for the membrane potential: v + (dt / c) * (i_ext - sum(i_ion)).

    Args:
        v: Membrane voltage in mV.
        i_ext: Injected external current density in µA/cm².
        i_na: Outward sodium current density in µA/cm².
        i_k: Outward potassium current density in µA/cm².
        i_leak: Outward leak current density in µA/cm².
        c: Membrane capacitance in µF/cm².
        dt: Time step duration in ms.

    Returns:
        Updated membrane voltage in mV after dt.
    """
    return v + (dt / c) * (i_ext - i_na - i_k - i_leak)


def resting_potential(
    *,
    g_na: float = 120.0,
    g_k: float = 36.0,
    g_leak: float = 0.3,
    e_na: float = 50.0,
    e_k: float = -77.0,
    e_leak: float = -54.387,
    temperature: float = 6.3,
    q10: float = 3.0,
) -> float:
    """Calculate the resting potential by bisection on net ionic current at steady state.

    Args:
        g_na: Maximum sodium conductance in mS/cm².
        g_k: Maximum potassium conductance in mS/cm².
        g_leak: Leak conductance in mS/cm².
        e_na: Sodium reversal potential in mV.
        e_k: Potassium reversal potential in mV.
        e_leak: Leak reversal potential in mV.
        temperature: Temperature in °C.
        q10: Temperature coefficient Q10.

    Returns:
        Resting potential in mV.
    """

    def _total_ionic_current(v_val: float) -> float:
        # phi = 1.0 because the steady state does not depend on the temperature factor.
        v_t = torch.tensor(v_val, dtype=torch.float64)
        steady = gate_steady_state(v_t, phi=1.0)
        i_na, i_k, i_leak = hh_currents(
            v_t,
            steady.m,
            steady.h,
            steady.n,
            g_na=g_na,
            g_k=g_k,
            g_leak=g_leak,
            e_na=e_na,
            e_k=e_k,
            e_leak=e_leak,
        )
        return float(i_na + i_k + i_leak)

    low, high = -100.0, 0.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if _total_ionic_current(mid) < 0.0:
            low = mid
        else:
            high = mid
    return mid


class HodgkinHuxley(Behavior):
    """Squid-axon Hodgkin-Huxley (1952) neuron model with visible ion-channel state.

    Gates use the exact exponential update with V frozen over each substep and V uses
    forward Euler, so dt/substeps <= 0.05 ms is required. Spikes are detected as upward
    threshold crossings of v_spike and do not reset the membrane potential; do not
    combine with Fire. State on the group: ``v``, ``m``, ``h``, ``n``, ``spikes``,
    ``i_na``, ``i_k``, ``i_leak``, ``v_prev``, and resting potential ``v_rest`` (in mV).

    Args:
        c: Membrane capacitance in µF/cm².
        g_na: Maximum sodium conductance in mS/cm².
        g_k: Maximum potassium conductance in mS/cm².
        g_leak: Leak conductance in mS/cm².
        e_na: Sodium reversal potential in mV.
        e_k: Potassium reversal potential in mV.
        e_leak: Leak reversal potential in mV.
        temperature: Temperature in °C.
        q10: Temperature coefficient Q10.
        v_spike: Spike detection threshold in mV for upward crossings.
        substeps: Number of Euler substeps per network simulation step.
        v_init: Initial membrane potential in mV; None sets the numerical resting potential.
    """

    order = Order.NEURON_DYNAMICS
    independent_ok = True
    graph_safe = True

    def __init__(
        self,
        *,
        c: float = 1.0,
        g_na: float = 120.0,
        g_k: float = 36.0,
        g_leak: float = 0.3,
        e_na: float = 50.0,
        e_k: float = -77.0,
        e_leak: float = -54.387,
        temperature: float = 6.3,
        q10: float = 3.0,
        v_spike: float = 0.0,
        substeps: int = 1,
        v_init: float | None = None,
    ) -> None:
        positive(c=c, g_na=g_na, g_k=g_k, g_leak=g_leak, q10=q10)
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError(f"substeps must be at least 1, got {substeps}")

        self.c = float(c)
        self.g_na = float(g_na)
        self.g_k = float(g_k)
        self.g_leak = float(g_leak)
        self.e_na = float(e_na)
        self.e_k = float(e_k)
        self.e_leak = float(e_leak)
        self.temperature = float(temperature)
        self.q10 = float(q10)
        self.v_spike = float(v_spike)
        self.substeps = substeps
        self.v_init = float(v_init) if v_init is not None else None
        self.phi = float(q10 ** ((temperature - 6.3) / 10.0))

    def initialize(self, group: NeuronGroup) -> None:
        """Check numerical stability, verify collaborators, and allocate group state."""
        dt = group.net.dt
        if (dt / self.substeps) > 0.05 + 1e-12:
            raise ValueError(
                f"Hodgkin-Huxley on {group.name} needs dt/substeps <= 0.05 ms; "
                f"got dt={dt}, substeps={self.substeps}"
            )

        for behavior in group.behaviors:
            if isinstance(behavior, Fire):
                raise ValueError(
                    f"Hodgkin-Huxley on {group.name} cannot be combined with Fire; "
                    "Hodgkin-Huxley generates its own action potentials without reset"
                )

        v_rest = resting_potential(
            g_na=self.g_na,
            g_k=self.g_k,
            g_leak=self.g_leak,
            e_na=self.e_na,
            e_k=self.e_k,
            e_leak=self.e_leak,
            temperature=self.temperature,
            q10=self.q10,
        )
        self._v_rest = v_rest
        self._v_init = self.v_init if self.v_init is not None else v_rest

        group.v = group.state(self._v_init)
        group.v_rest = self._v_rest
        group.threshold = group.vector(self.v_spike)
        group.spikes = group.state(False, dtype=torch.bool)
        if not hasattr(group, "I") or group.I is None:
            group.I = group.state()

        steady_gates = gate_steady_state(group.v, phi=self.phi)
        group.m = steady_gates.m.clone()
        group.h = steady_gates.h.clone()
        group.n = steady_gates.n.clone()

        group.i_na = group.state(0.0)
        group.i_k = group.state(0.0)
        group.i_leak = group.state(0.0)
        group.v_prev = group.state(self._v_init)
        group.model = self

    def reset_state(self, group: NeuronGroup) -> None:
        """Return voltage and gates to initial steady-state values and clear currents and spikes."""
        group.v.fill_(self._v_init)
        steady_gates = gate_steady_state(group.v, phi=self.phi)
        group.m.copy_(steady_gates.m)
        group.h.copy_(steady_gates.h)
        group.n.copy_(steady_gates.n)
        group.i_na.zero_()
        group.i_k.zero_()
        group.i_leak.zero_()
        group.spikes.zero_()
        group.I.zero_()
        group.v_prev.copy_(group.v)

    def forward(self, group: NeuronGroup) -> None:
        """Integrate gates and membrane potential over one simulation step."""
        dt_sub = group.net.dt / self.substeps
        v = group.v
        m = group.m
        h = group.h
        n = group.n
        i_ext = group.I
        c = self.c
        phi = self.phi
        v_spike = self.v_spike
        g_na = self.g_na
        g_k = self.g_k
        g_leak = self.g_leak
        e_na = self.e_na
        e_k = self.e_k
        e_leak = self.e_leak

        spikes = torch.zeros_like(group.spikes)
        group.v_prev.copy_(v)

        for _ in range(self.substeps):
            v_before = v
            rates = gate_rates(v_before, phi=phi)
            m = gating_step(m, rates.m.alpha, rates.m.beta, dt=dt_sub)
            h = gating_step(h, rates.h.alpha, rates.h.beta, dt=dt_sub)
            n = gating_step(n, rates.n.alpha, rates.n.beta, dt=dt_sub)
            i_na, i_k, i_leak = hh_currents(
                v_before,
                m,
                h,
                n,
                g_na=g_na,
                g_k=g_k,
                g_leak=g_leak,
                e_na=e_na,
                e_k=e_k,
                e_leak=e_leak,
            )
            v_after = membrane_step(v_before, i_ext, i_na, i_k, i_leak, c=c, dt=dt_sub)
            spikes = spikes | ((v_before < v_spike) & (v_after >= v_spike))
            v = v_after

        group.v = v
        group.m = m
        group.h = h
        group.n = n
        group.i_na = i_na
        group.i_k = i_k
        group.i_leak = i_leak
        group.spikes = spikes

    def fire(self, group: NeuronGroup) -> None:
        """Emit spikes and reset. Not used for Hodgkin-Huxley as spikes occur in forward."""
        raise RuntimeError(
            f"Hodgkin-Huxley on {group.name} cannot be combined with Fire; "
            "Hodgkin-Huxley generates its own action potentials without reset"
        )
