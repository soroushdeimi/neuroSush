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


def _positive(**values: float | torch.Tensor) -> None:
    """Raise ValueError if any value is <= 0."""
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            if torch.any(value <= 0):
                raise ValueError(f"{name} must be positive, got {value}")
        elif value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


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
        if isinstance(self.threshold, torch.Tensor):
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
        alpha: Adaptation sensitivity.
        beta: Adaptation spike-triggered increment.
        tau_w: Adaptation time constant.
        omega_init: Initial adaptation variable.
        v_init: Initial membrane voltage.
    """

    def __init__(
        self,
        *,
        alpha: float,
        beta: float,
        tau_w: float,
        omega_init: float = 0.0,
        **kwargs: Unpack[_ELIFOptions],
    ) -> None:
        super().__init__(**kwargs)
        _positive(tau_w=tau_w)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau_w = float(tau_w)
        self.omega_init = float(omega_init)

    def initialize(self, group: NeuronGroup) -> None:
        """Set the ELIF state and the adaptation current ``omega``."""
        super().initialize(group)
        group.omega = group.state(self.omega_init)

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
            alpha=self.alpha,
            beta=self.beta,
            tau_w=self.tau_w,
            dt=group.net.dt,
        )


class Fire(Behavior):
    """Calls the group's neuron model to emit spikes (after noise and competition)."""

    order = Order.FIRE

    def initialize(self, group: NeuronGroup) -> None:
        """Check that the group has a neuron model."""
        if not hasattr(group, "model") or group.model is None or not hasattr(group.model, "fire"):
            raise RuntimeError(f"Fire on {group.name} needs a neuron model such as LIF")

    def forward(self, group: NeuronGroup) -> None:
        """Emit spikes and reset."""
        group.model.fire(group)
