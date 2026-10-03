"""Leaky integrate-and-fire neuron models and the firing step.

A model integrates the membrane in ``Order.NEURON_DYNAMICS`` and exposes ``fire``, which the
:class:`Fire` behavior calls later in the step so that noise and competition can act on the
voltage in between. State written on the group: ``v``, ``spikes``, ``I`` (if absent),
``threshold`` (per neuron), ``v_rest``, ``v_reset``, ``tau``, ``resistance`` and ``model``.
"""

from __future__ import annotations

from typing import TypedDict

import torch
from typing_extensions import NotRequired, Unpack

from neurosush.core.behavior import Behavior
from neurosush.core.network import NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons import dynamics
from neurosush.neurons.params import keep, per_neuron, positive, state_like

_positive = positive


class _LIFOptions(TypedDict):
    tau: float
    threshold: float | torch.Tensor
    v_reset: float
    v_rest: float
    resistance: NotRequired[float]
    v_init: NotRequired[float | torch.Tensor | None]


class _ELIFOptions(_LIFOptions):
    delta: float
    theta_rh: float


class LIF(Behavior):
    """Leaky Integrate-and-Fire neuron model.

    Equation: tau * dv/dt = (v_rest - v) + R * I

    Args:
        tau: Membrane time constant.
        threshold: Voltage threshold for spiking.
        v_reset: Voltage after a spike.
        v_rest: Resting membrane voltage.
        resistance: Membrane resistance.
        v_init: Initial membrane voltage.
    """

    order = Order.NEURON_DYNAMICS
    independent_ok = True
    graph_safe = True

    def __init__(
        self,
        *,
        tau: float,
        threshold: float | torch.Tensor,
        v_reset: float,
        v_rest: float,
        resistance: float = 1.0,
        v_init: float | torch.Tensor | None = None,
    ) -> None:
        _positive(tau=tau, resistance=resistance)
        if isinstance(threshold, torch.Tensor):
            if torch.any(torch.as_tensor(v_reset, device=threshold.device) >= threshold):
                raise ValueError(f"v_reset must be less than threshold, got {v_reset}")
        elif v_reset >= threshold:
            raise ValueError(f"v_reset must be less than threshold, got {v_reset}")

        self.tau = float(tau)
        self.threshold = threshold
        self.v_reset = float(v_reset)
        self.v_rest = float(v_rest)
        self.resistance = float(resistance)
        self.v_init = v_init

    def initialize(self, group: NeuronGroup) -> None:
        """Set the parameters and the initial state on ``group``."""
        group.tau = self.tau
        group.resistance = self.resistance
        group.v_rest = self.v_rest
        group.v_reset = self.v_reset
        if isinstance(self.threshold, torch.Tensor) and group.net.independent:
            threshold = self.threshold.to(dtype=group.net.dtype, device=group.net.device)
            if threshold.shape not in ((group.size,), group.state_shape):
                raise ValueError(
                    f"threshold must have shape ({group.size},) or {group.state_shape}, "
                    f"got {tuple(threshold.shape)}"
                )
            group.threshold = threshold.expand(group.state_shape).clone()
        elif isinstance(self.threshold, torch.Tensor):
            group.threshold = self.threshold.to(group.net.dtype).to(group.net.device)
        else:
            group.threshold = group.vector(self.threshold)
        group.spikes = group.state(False, dtype=torch.bool)

        if not hasattr(group, "I") or group.I is None:
            group.I = group.state()

        if self.v_init is None:
            group.v = group.state(self.v_rest)
        elif isinstance(self.v_init, torch.Tensor):
            if self.v_init.shape not in ((group.size,), group.state_shape):
                raise ValueError(
                    f"v_init must have shape ({group.size},) or {group.state_shape}, "
                    f"got {tuple(self.v_init.shape)}"
                )
            v_init = self.v_init.to(dtype=group.net.dtype, device=group.net.device)
            group.v = v_init.expand(group.state_shape).clone()
        else:
            group.v = group.state(self.v_init)

        group.model = self

    def reset_state(self, group: NeuronGroup) -> None:
        """Return ``v`` to ``v_init`` (``v_rest`` without one), clear the spikes and ``I``."""
        if self.v_init is None:
            group.v.fill_(self.v_rest)
        elif isinstance(self.v_init, torch.Tensor):
            v_init = self.v_init.to(dtype=group.net.dtype, device=group.net.device)
            group.v.copy_(v_init.expand(group.state_shape))
        else:
            group.v.fill_(self.v_init)
        group.spikes.zero_()
        group.I.zero_()

    def derivative(self, group: NeuronGroup) -> torch.Tensor:
        """``tau * dv/dt`` of the current state."""
        return dynamics.lif_derivative(
            group.v, group.I, v_rest=group.v_rest, resistance=group.resistance
        )

    def forward(self, group: NeuronGroup) -> None:
        """Integrate the membrane for one step (forward Euler)."""
        group.v = dynamics.euler_step(
            group.v, self.derivative(group), tau=group.tau, dt=group.net.dt
        )

    def fire(self, group: NeuronGroup) -> None:
        """Emit spikes where ``v >= threshold`` and reset those neurons."""
        group.spikes, group.v = dynamics.fire(
            group.v, threshold=group.threshold, v_reset=group.v_reset
        )


class ELIF(LIF):
    """Exponential Leaky Integrate-and-Fire neuron model.

    Equation: tau * dv/dt = (v_rest - v) + R * I + delta * exp((v - theta_rh) / delta)

    Args:
        tau: Membrane time constant.
        threshold: Voltage threshold for spiking.
        v_reset: Voltage after a spike.
        v_rest: Resting membrane voltage.
        resistance: Membrane resistance.
        delta: Exponential boost scale.
        theta_rh: Rheobase threshold of the exponential term.
        v_init: Initial membrane voltage.
    """

    def __init__(
        self,
        *,
        delta: float,
        theta_rh: float,
        **kwargs: Unpack[_LIFOptions],
    ) -> None:
        super().__init__(**kwargs)
        _positive(delta=delta)
        self.delta = float(delta)
        self.theta_rh = float(theta_rh)

    def derivative(self, group: NeuronGroup) -> torch.Tensor:
        """``tau * dv/dt`` including the exponential term."""
        return super().derivative(group) + dynamics.exponential_boost(
            group.v, delta=self.delta, theta_rh=self.theta_rh
        )


class AdaptiveELIF(ELIF):
    """Adaptive Exponential Leaky Integrate-and-Fire neuron model.

    Equation: tau * dv/dt = (v_rest - v) + R * I + delta * exp((v - theta_rh) / delta) - R * omega

    Args:
        tau: Membrane time constant.
        threshold: Voltage threshold for spiking.
        v_reset: Voltage after a spike.
        v_rest: Resting membrane voltage.
        resistance: Membrane resistance.
        delta: Exponential boost scale.
        theta_rh: Rheobase threshold of the exponential term.
        alpha: Adaptation sensitivity: a number, or a tensor of shape ``(size,)``
            (``(batch_size, size)`` in an independent network) for one value per neuron.
        beta: Adaptation spike-triggered increment, per neuron as ``alpha``.
        tau_w: Adaptation time constant (positive), per neuron as ``alpha``.
        omega_init: Initial adaptation variable.
        v_init: Initial membrane voltage.
    """

    def __init__(
        self,
        *,
        alpha: float | torch.Tensor,
        beta: float | torch.Tensor,
        tau_w: float | torch.Tensor,
        omega_init: float = 0.0,
        **kwargs: Unpack[_ELIFOptions],
    ) -> None:
        super().__init__(**kwargs)
        _positive(tau_w=tau_w)
        self.alpha = keep(alpha)
        self.beta = keep(beta)
        self.tau_w = keep(tau_w)
        self.omega_init = float(omega_init)

    def initialize(self, group: NeuronGroup) -> None:
        """Set the ELIF state and the adaptation current ``omega``."""
        super().initialize(group)
        self._alpha = per_neuron(group, self.alpha, "alpha")
        self._beta = per_neuron(group, self.beta, "beta")
        self._tau_w = per_neuron(group, self.tau_w, "tau_w")
        group.omega = group.state(self.omega_init)

    def reset_state(self, group: NeuronGroup) -> None:
        """Reset the LIF state and return ``omega`` to ``omega_init``."""
        super().reset_state(group)
        group.omega.fill_(self.omega_init)

    def derivative(self, group: NeuronGroup) -> torch.Tensor:
        """``tau * dv/dt`` including the adaptation current."""
        return super().derivative(group) - group.resistance * group.omega

    def fire(self, group: NeuronGroup) -> None:
        """Fire, then update ``omega`` using the voltage from before the reset."""
        v_before = group.v.clone()
        super().fire(group)
        group.omega = dynamics.adaptation_step(
            group.omega,
            v_before,
            group.spikes,
            v_rest=group.v_rest,
            alpha=self._alpha,
            beta=self._beta,
            tau_w=self._tau_w,
            dt=group.net.dt,
        )


class Izhikevich(Behavior):
    """Izhikevich (2003) neuron: a two-variable model with spike-frequency dynamics.

    Equations (``v`` in mV, time in ms)::

        v' = 0.04 v^2 + 5 v + 140 - u + I
        u' = a (b v - u)
        if v >= v_peak: v <- c, u <- u + d

    ``v`` takes ``substeps`` forward Euler steps of ``dt / substeps`` (two half steps of
    0.5 ms for ``dt = 1`` ms, as in the paper), then ``u`` one Euler step of ``dt`` from the
    updated ``v``. The spike test and reset run in :meth:`fire`, so the model works with
    :class:`Fire` (and noise or competition in between) and takes its input from
    ``group.I``, for example from ``DendriteIntegration``. Runs at ``Order.NEURON_DYNAMICS``.
    State on the group: ``v``, ``u``, ``spikes``, ``I`` (if absent); parameters
    ``threshold`` (``v_peak``), ``v_reset`` (``c``), ``tau`` and ``resistance`` (1) and
    ``v_rest`` (-65, only read by dendritic priming).

    Typical sets: regular spiking ``a=0.02, b=0.2, c=-65, d=8``; fast spiking ``a=0.1, d=2``;
    chattering ``c=-50, d=2``; intrinsically bursting ``c=-55, d=4``; low-threshold
    spiking ``b=0.25``.

    Args:
        a: Recovery time scale (positive): a number, or a tensor of shape ``(size,)``
            (``(batch_size, size)`` in an independent network) for one value per neuron.
        b: Recovery sensitivity to ``v``, per neuron as ``a``.
        c: Voltage after a spike (below ``v_peak``), per neuron as ``a``.
        d: Recovery increment per spike, per neuron as ``a``.
        v_peak: Spike threshold.
        v_init: Initial voltage: a number or a tensor of shape ``(size,)`` or the state shape.
        u_init: Initial recovery variable; ``b * v_init`` when omitted.
        substeps: Euler sub-steps of ``v`` per step.
    """

    order = Order.NEURON_DYNAMICS
    independent_ok = True
    graph_safe = True

    def __init__(
        self,
        *,
        a: float | torch.Tensor,
        b: float | torch.Tensor,
        c: float | torch.Tensor,
        d: float | torch.Tensor,
        v_peak: float = 30.0,
        v_init: float | torch.Tensor = -65.0,
        u_init: float | torch.Tensor | None = None,
        substeps: int = 2,
    ) -> None:
        positive(a=a)
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError(f"substeps must be a positive integer, got {substeps}")
        if torch.any(torch.as_tensor(c) >= v_peak):
            raise ValueError(f"c must be less than v_peak ({v_peak}), got {c}")
        self.a, self.b, self.c, self.d = keep(a), keep(b), keep(c), keep(d)
        self.v_peak = float(v_peak)
        self.v_init = v_init
        self.u_init = u_init
        self.substeps = substeps

    def initialize(self, group: NeuronGroup) -> None:
        """Set the parameters and the initial ``v`` and ``u`` on ``group``."""
        self._a = per_neuron(group, self.a, "a")
        self._b = per_neuron(group, self.b, "b")
        self._c = per_neuron(group, self.c, "c")
        self._d = per_neuron(group, self.d, "d")
        group.tau = 1.0
        group.resistance = 1.0
        group.v_rest = -65.0
        group.v_reset = self._c
        group.threshold = group.vector(self.v_peak)
        group.spikes = group.state(False, dtype=torch.bool)
        if not hasattr(group, "I") or group.I is None:
            group.I = group.state()
        group.v = state_like(group, self.v_init, "v_init")
        group.u = self._initial_u(group)
        group.model = self

    def _initial_u(self, group: NeuronGroup) -> torch.Tensor:
        if self.u_init is None:
            return self._b * group.v if isinstance(self._b, torch.Tensor) else group.v * self._b
        return state_like(group, self.u_init, "u_init")

    def reset_state(self, group: NeuronGroup) -> None:
        """Return ``v`` and ``u`` to their initial values and clear the spikes and ``I``."""
        group.v.copy_(state_like(group, self.v_init, "v_init"))
        group.u.copy_(self._initial_u(group))
        group.spikes.zero_()
        group.I.zero_()

    def forward(self, group: NeuronGroup) -> None:
        """Advance ``v`` and ``u`` by one step."""
        group.v, group.u = dynamics.izhikevich_step(
            group.v,
            group.u,
            group.I,
            a=self._a,
            b=self._b,
            dt=group.net.dt,
            substeps=self.substeps,
        )

    def fire(self, group: NeuronGroup) -> None:
        """Emit spikes where ``v >= v_peak``; reset ``v`` to ``c`` and add ``d`` to ``u``."""
        group.spikes, group.v, group.u = dynamics.izhikevich_reset(
            group.v, group.u, v_peak=group.threshold, c=self._c, d=self._d
        )


class Fire(Behavior):
    """Calls the group's neuron model to emit spikes (after noise and competition)."""

    order = Order.FIRE
    independent_ok = True
    graph_safe = True

    def initialize(self, group: NeuronGroup) -> None:
        """Check that the group has a neuron model."""
        if not hasattr(group, "model") or group.model is None or not hasattr(group.model, "fire"):
            raise RuntimeError(f"Fire on {group.name} needs a neuron model such as LIF")

    def forward(self, group: NeuronGroup) -> None:
        """Emit spikes and reset."""
        group.model.fire(group)


class Refractory(Behavior):
    """Absolute refractory period: after a spike the membrane stays at ``v_reset``.

    A neuron that fires is held at ``v_reset`` for the next ``ceil(period / dt)`` steps, so
    it cannot fire again sooner. Runs after the neuron model, noise and competition and
    before :class:`Fire` (``Order.REFRACTORY``). State on the group: ``refractory``, the
    time left.

    Args:
        period: Refractory period, in the unit of ``dt``.
    """

    order = Order.REFRACTORY
    independent_ok = True
    graph_safe = True

    def __init__(self, period: float) -> None:
        _positive(period=period)
        self.period = float(period)

    def initialize(self, group: NeuronGroup) -> None:
        """Check for a neuron model and allocate the countdown."""
        if not hasattr(group, "v_reset"):
            raise RuntimeError(f"Refractory on {group.name} needs a neuron model such as LIF")
        group.refractory = group.state()

    def reset_state(self, group: NeuronGroup) -> None:
        """End every countdown."""
        group.refractory.zero_()

    def forward(self, group: NeuronGroup) -> None:
        """Start the countdown on the last step's spikes and clamp neurons still in it."""
        group.refractory = torch.where(
            group.spikes, self.period, (group.refractory - group.net.dt).clamp(min=0)
        )
        group.v = group.v.masked_fill(group.refractory > 1e-9, group.v_reset)
