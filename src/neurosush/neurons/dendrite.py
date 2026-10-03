"""Dendritic compartments and integration behaviors."""

from __future__ import annotations

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.buffers import ArrivalBuffer
from neurosush.core.network import Compartment, NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons.params import at_least, keep, per_neuron, positive

_COMPARTMENTS = tuple(Compartment)  # older dynamo cannot iterate an enum class


class DendriteStructure(Behavior):
    """Allocates dendritic compartments and buffers for incoming synaptic current.

    Args:
        proximal_depth: Steps of delay storage for proximal input; delays must be below it.
        distal_depth: Same for distal input.
        apical_depth: Same for apical input.
    """

    order = Order.DENDRITE_STRUCTURE
    independent_ok = True

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
                self.depths[c],
                group.size,
                dtype=group.net.dtype,
                device=group.net.device,
                batch=group.net.batch_size,
            )
            for c in Compartment
        }
        for c in Compartment:
            setattr(group, f"I_{c.value}", group.state())
        self._silent = group.state()

    def reset_state(self, group: NeuronGroup) -> None:
        """Clear the delay buffers and the compartment currents.

        A compartment current may be the (read-only) current of its synapse, which is cleared
        too, so zeroing it in place is safe.
        """
        for buffer in group.dendrite.values():
            buffer.reset()
        for c in Compartment:
            getattr(group, f"I_{c.value}").zero_()

    def forward(self, group: NeuronGroup) -> None:
        """Advance buffers and accumulate synaptic currents.

        The compartment currents are read-only: a compartment without synapses shares one
        zero tensor, and one without delays passes its synapse's current on as is.
        """
        for c in _COMPARTMENTS:
            synapses = group.afferent[c]
            if not synapses:
                current = self._silent
            elif self.depths[c] == 1:  # no delays (checked at initialization): no buffer
                current = synapses[0].I
                for syn in synapses[1:]:
                    current = current + syn.I
            else:
                buffer = group.dendrite[c]
                buffer.advance()
                for syn in synapses:
                    buffer.add(syn.I, syn.dst_delay)
                current = buffer.current()
            setattr(group, f"I_{c.value}", current)

    def graph_ready(self, group: NeuronGroup) -> bool:
        """Ready when every compartment with synapses has depth 1 (no delay buffer to advance)."""
        return all(
            self.depths[compartment] == 1
            for compartment, synapses in group.afferent.items()
            if synapses
        )


def modulatory_drive(
    current: torch.Tensor,
    v: torch.Tensor,
    *,
    v_rest: float,
    threshold: float | torch.Tensor,
    gain: float,
) -> torch.Tensor:
    """Priming voltage rate ``tanh(current) * max(limit - v, 0)``.

    ``limit = v_rest + gain * (threshold - v_rest)``: the drive primes the neuron toward the
    limit and never pushes past it.

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
    independent_ok = True
    graph_safe = True

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
        group.I = group.state()

    def reset_state(self, group: NeuronGroup) -> None:
        """Zero the integrated current."""
        group.I.zero_()

    def forward(self, group: NeuronGroup) -> None:
        """Integrate dendritic currents using decay and priming.

        The priming term moves the membrane toward the limit at rate
        tanh(I_compartment) per unit time through the LIF step.
        """
        if self.tau_current is None:
            current = torch.zeros_like(group.I)
        else:
            current = group.I * (1 - group.net.dt / self.tau_current)
        current = current + group.I_proximal
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
        group.I = current


def conductance_step(
    v: torch.Tensor,
    g_exc: torch.Tensor,
    g_inh: torch.Tensor,
    *,
    v_rest: float,
    e_exc: float,
    e_inh: float,
    resistance: float,
    tau: float,
    dt: float,
) -> torch.Tensor:
    """Exact voltage after ``dt`` with the conductances held constant over the step.

    Solves ``tau dv/dt = (v_rest - v) + R g_exc (e_exc - v) + R g_inh (e_inh - v)``, which is
    linear in ``v``: ``v_inf + (v - v_inf) exp(-(1 + R g) dt / tau)`` with
    ``g = g_exc + g_inh`` and ``v_inf = (v_rest + R (g_exc e_exc + g_inh e_inh)) / (1 + R g)``.
    Unlike a forward Euler step it stays between the reversal potentials for any
    conductance.
    """
    g = resistance * (g_exc + g_inh)
    v_inf = (v_rest + resistance * (g_exc * e_exc + g_inh * e_inh)) / (1 + g)
    return v_inf + (v - v_inf) * torch.exp(-(1 + g) * (dt / tau))


class ConductanceIntegration(Behavior):
    """Conductance-based synapses: input pulls the membrane towards a reversal potential.

    Proximal synaptic input from excitatory groups adds to the conductance ``g_exc``, from
    inhibitory groups (negative currents) to ``g_inh``; each conductance decays with its
    own time constant, ``g = g (1 - dt / tau_g) + input``, so a spike's conductance
    integrates to ``weight * tau_g`` as in the continuous model. The synaptic current is
    ``g_exc (e_exc - v) + g_inh (e_inh - v)``: it shrinks as the voltage approaches the
    reversal potential, so excitation saturates.

    It replaces :class:`DendriteIntegration` (no :class:`DendriteStructure` is needed) and
    sets ``group.I`` to the current with which the neuron model's Euler step lands exactly
    on :func:`conductance_step`: integration is exact for :class:`~neurosush.neurons.models.LIF`
    and stays stable under strong conductances. Synapses need no dendritic delay and must
    target the proximal compartment.

    Args:
        e_exc: Excitatory reversal potential.
        e_inh: Inhibitory reversal potential.
        tau_exc: Decay time constant of ``g_exc``, at least ``dt``: a number, or a tensor of
            shape ``(size,)`` (``(batch_size, size)`` in an independent network) for one
            value per neuron.
        tau_inh: Decay time constant of ``g_inh``, at least ``dt``, per neuron as ``tau_exc``.
    """

    order = Order.DENDRITE_INTEGRATION
    independent_ok = True
    graph_safe = True

    def __init__(
        self,
        *,
        e_exc: float = 0.0,
        e_inh: float = -100.0,
        tau_exc: float | torch.Tensor = 1.0,
        tau_inh: float | torch.Tensor = 1.0,
    ) -> None:
        if e_inh >= e_exc:
            raise ValueError(f"e_inh ({e_inh}) must be below e_exc ({e_exc})")
        positive(tau_exc=tau_exc, tau_inh=tau_inh)
        self.e_exc, self.e_inh = e_exc, e_inh
        self.tau_exc, self.tau_inh = keep(tau_exc), keep(tau_inh)

    def initialize(self, group: NeuronGroup) -> None:
        """Check the synapses and time constants; allocate the conductances."""
        dt = group.net.dt
        at_least(dt, tau_exc=self.tau_exc, tau_inh=self.tau_inh)
        self._tau_exc = per_neuron(group, self.tau_exc, "tau_exc")
        self._tau_inh = per_neuron(group, self.tau_inh, "tau_inh")
        for compartment, synapses in group.afferent.items():
            for syn in synapses:
                if compartment is not Compartment.PROXIMAL:
                    raise ValueError(
                        f"ConductanceIntegration on {group.name} takes proximal synapses "
                        f"only, got {syn.name} on {compartment.value}"
                    )
                if int(syn.dst_delay.max()) > 0:
                    raise ValueError(
                        f"ConductanceIntegration on {group.name} does not support dst_delay "
                        f"({syn.name})"
                    )
        if any(isinstance(b, DendriteIntegration) for b in group.behaviors):
            raise ValueError(
                f"{group.name} has both ConductanceIntegration and DendriteIntegration"
            )
        group.g_exc = group.state()
        group.g_inh = group.state()
        group.I = group.state()

    def reset_state(self, group: NeuronGroup) -> None:
        """Zero both conductances and the current."""
        group.g_exc.zero_()
        group.g_inh.zero_()
        group.I.zero_()

    def forward(self, group: NeuronGroup) -> None:
        """Update the conductances and set the equivalent current ``group.I``."""
        dt = group.net.dt
        g_exc = group.g_exc * (1 - dt / self._tau_exc)
        g_inh = group.g_inh * (1 - dt / self._tau_inh)
        for syn in group.afferent[Compartment.PROXIMAL]:
            if syn.src.inhibitory:
                g_inh = g_inh - syn.I
            else:
                g_exc = g_exc + syn.I
        group.g_exc, group.g_inh = g_exc, g_inh
        v, tau, resistance = group.v, group.tau, group.resistance
        v_next = conductance_step(
            v,
            g_exc,
            g_inh,
            v_rest=group.v_rest,
            e_exc=self.e_exc,
            e_inh=self.e_inh,
            resistance=resistance,
            tau=tau,
            dt=dt,
        )
        # the current whose Euler step, v + dt/tau ((v_rest - v) + R I), gives v_next
        group.I = ((v_next - v) * (tau / dt) - (group.v_rest - v)) / resistance
