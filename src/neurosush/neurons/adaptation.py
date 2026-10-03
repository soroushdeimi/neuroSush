"""Spike-triggered currents: after-spike currents and spike-frequency adaptation."""

from __future__ import annotations

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons import dynamics
from neurosush.neurons.dendrite import DendriteIntegration
from neurosush.neurons.params import at_least, keep, per_neuron, positive


class SpikeTriggeredCurrent(Behavior):
    """A current that jumps by ``amplitude`` at each spike of the group and decays with ``tau``.

    ``J <- J (1 - dt / tau) + amplitude * spike`` where ``spike`` is the group's spike of the
    previous step, and ``group.I`` gets ``+ J`` before the neuron model integrates. A negative
    amplitude is a hyperpolarizing after-spike current: spike-frequency adaptation for a plain
    :class:`~neurosush.neurons.models.LIF`, or the afterpotential of a spike response model.

    It runs at ``Order.ADAPTATION`` (250): after ``DendriteIntegration`` and
    ``ConductanceIntegration`` (240) have rebuilt ``group.I`` for the step and before the
    neuron model (260) integrates it, so the jump acts one step after the spike, as every
    spike effect does. ``group.I`` must be rebuilt every step; combining with
    ``DendriteIntegration(tau_current=...)`` (which filters the old ``group.I``) is refused,
    since the filter would feed the current back on itself. State on the group: ``I_adapt``.

    Args:
        amplitude: Jump per spike, in the unit of ``group.I``: a number, or a tensor of shape
            ``(size,)`` (``(batch_size, size)`` in an independent network).
        tau: Decay time constant (positive, at least ``dt``), per neuron as ``amplitude``.
    """

    order = Order.ADAPTATION
    independent_ok = True
    graph_safe = True

    def __init__(self, amplitude: float | torch.Tensor, tau: float | torch.Tensor) -> None:
        positive(tau=tau)
        self.amplitude = keep(amplitude)
        self.tau = keep(tau)

    def initialize(self, group: NeuronGroup) -> None:
        """Check the time constant and allocate ``I_adapt``."""
        at_least(group.net.dt, tau=self.tau)
        for behavior in group.behaviors:
            if isinstance(behavior, DendriteIntegration) and behavior.tau_current is not None:
                raise ValueError(
                    f"SpikeTriggeredCurrent on {group.name} cannot follow "
                    "DendriteIntegration(tau_current=...)"
                )
        self._amplitude = per_neuron(group, self.amplitude, "amplitude")
        self._tau = per_neuron(group, self.tau, "tau")
        group.I_adapt = group.state()
        if not hasattr(group, "I") or group.I is None:
            group.I = group.state()
        if not hasattr(group, "spikes"):
            group.spikes = group.state(False, dtype=torch.bool)

    def reset_state(self, group: NeuronGroup) -> None:
        """Zero the current."""
        group.I_adapt.zero_()

    def forward(self, group: NeuronGroup) -> None:
        """Decay the current, add the jumps of last step's spikes and add it to ``group.I``."""
        group.I_adapt = dynamics.decaying_current_step(
            group.I_adapt,
            group.spikes,
            amplitude=self._amplitude,
            tau=self._tau,
            dt=group.net.dt,
        )
        group.I = group.I + group.I_adapt
