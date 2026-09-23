"""Dendritic compartments and integration behaviors."""

from __future__ import annotations

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.buffers import ArrivalBuffer
from neurosush.core.network import Compartment, NeuronGroup
from neurosush.core.order import Order


class DendriteStructure(Behavior):
    """Allocates dendritic compartments and buffers for incoming synaptic current.

    Args:
        proximal_depth: Number of steps for proximal compartment.
        distal_depth: Number of steps for distal compartment.
        apical_depth: Number of steps for apical compartment.
    """

    order = Order.DENDRITE_STRUCTURE

    def __init__(
        self,
        *,
        proximal_depth: int = 1,
        distal_depth: int = 1,
        apical_depth: int = 1,
    ) -> None:
        self.depths = {
            Compartment.PROXIMAL: proximal_depth,
            Compartment.DISTAL: distal_depth,
            Compartment.APICAL: apical_depth,
        }
        for compartment, depth in self.depths.items():
            if depth < 1:
                raise ValueError(f"{compartment.value}_depth must be at least 1, got {depth}")

    def initialize(self, group: NeuronGroup) -> None:
        """Allocate dendritic buffers and current attributes on the group."""
        for compartment, synapses in group.afferent.items():
            for syn in synapses:
                if int(syn.dst_delay.max()) >= self.depths[compartment]:
                    raise ValueError(
                        f"dst_delay must be less than {self.depths[compartment]} for {syn.name}"
                    )

        group.dendrite = {
            c: ArrivalBuffer(
                self.depths[c], group.size, dtype=group.net.dtype, device=group.net.device
            )
            for c in Compartment
        }
        for c in Compartment:
            setattr(group, f"I_{c.value}", group.vector())

    def forward(self, group: NeuronGroup) -> None:
        """Advance buffers and accumulate synaptic currents."""
        for c in Compartment:
            buffer = group.dendrite[c]
            buffer.advance()
            for syn in group.afferent[c]:
                buffer.add(syn.I, syn.dst_delay)
            setattr(group, f"I_{c.value}", buffer.current())


def modulatory_drive(
    current: torch.Tensor,
    v: torch.Tensor,
    *,
    v_rest: float,
    threshold: float | torch.Tensor,
    gain: float,
) -> torch.Tensor:
    """Primes the neuron toward the limit and never pushes past it.

    Args:
        current: Input current.
        v: Membrane voltage.
        v_rest: Resting membrane voltage.
        threshold: Voltage threshold.
        gain: Gain of the modulatory drive.

    Returns:
        The modulatory drive term.
    """
    limit = v_rest + gain * (threshold - v_rest)
    return torch.tanh(current) * torch.clamp(limit - v, min=0)


class DendriteIntegration(Behavior):
    """Integrates dendritic currents into the main membrane current.

    Args:
        tau_current: Decay time constant for the integrated current.
        distal_gain: Gain for distal dendritic input.
        apical_gain: Gain for apical dendritic input.
    """

    order = Order.DENDRITE_INTEGRATION

    def __init__(
        self,
        *,
        tau_current: float | None = None,
        distal_gain: float | None = None,
        apical_gain: float | None = None,
    ) -> None:
        if tau_current is not None and tau_current <= 0:
            raise ValueError(f"tau_current must be positive, got {tau_current}")
        self.tau_current = tau_current
        self.distal_gain = distal_gain
        self.apical_gain = apical_gain

    def initialize(self, group: NeuronGroup) -> None:
        """Allocate integrated current on the group."""
        if not hasattr(group, "dendrite"):
            raise RuntimeError(f"DendriteIntegration on {group.name} needs a DendriteStructure")
        group.I = group.vector()

    def forward(self, group: NeuronGroup) -> None:
        """Integrate dendritic currents using decay and priming.

        The priming term moves the membrane toward the limit at rate
        tanh(I_compartment) per unit time through the LIF step.
        """
        # 1. Decay current
        if self.tau_current is None:
            current = torch.zeros_like(group.I)
        else:
            current = group.I * (1 - group.net.dt / self.tau_current)

        # 2. Add proximal
        current = current + group.I_proximal

        # 3. Add distal and apical with priming
        for gain, compartment_current in [
            (self.distal_gain, group.I_distal),
            (self.apical_gain, group.I_apical),
        ]:
            if gain is not None:
                current = current + (group.tau / group.resistance) * modulatory_drive(
                    compartment_current,
                    group.v,
                    v_rest=group.v_rest,
                    threshold=group.threshold,
                    gain=gain,
                )

        # 4. Update group.I
        group.I = current
