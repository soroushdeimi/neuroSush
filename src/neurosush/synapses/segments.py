"""Active dendritic segments: coincidence detection with NMDA-like plateaus.

A destination cell owns up to ``segments`` distal segments with up to ``synapses`` synapses
each (Hawkins and Ahmad 2016). A synapse counts when its permanence is at least
``connected``; a segment whose counted synapses see at least ``activation_threshold``
active presynaptic cells fires a dendritic (NMDA) spike, which holds a plateau for
``plateau`` time units (Antic et al. 2010; Bouhadjar et al. 2022). A synapse is active for
``coincidence`` time units after its presynaptic spike, so a segment sums input that
arrives within that window. While any segment of a
cell is in its plateau, the cell receives ``amplitude`` on the distal compartment, where
:class:`~neurosush.neurons.dendrite.DendriteIntegration` turns it into a subthreshold
depolarization: the cell is predicted, not fired.

Segments are stored as fixed-width tensors (``syn.presynaptic`` with -1 for an empty slot,
and ``syn.permanence``) of shape ``(dst.size, segments, synapses)``.
"""

from __future__ import annotations

import math

import torch

from neurosush.core.network import SynapseGroup
from neurosush.synapses.currents import _SynapticInput


def segment_counts(
    spikes: torch.Tensor, presynaptic: torch.Tensor, permanence: torch.Tensor, connected: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Active connected and active potential synapses of every segment.

    Args:
        spikes: Presynaptic spikes ``(..., n_src)``.
        presynaptic: Source index of every synapse ``(n_dst, S, M)``; -1 marks an empty slot.
        permanence: Permanence of every synapse ``(n_dst, S, M)``.
        connected: Permanence from which a synapse counts toward activation.

    Returns:
        Two tensors of shape ``(..., n_dst, S)``.
    """
    used = presynaptic >= 0
    active = spikes[..., presynaptic.clamp(min=0)] & used
    return (active & (permanence >= connected)).sum(-1), active.sum(-1)


def _steps(duration: float, dt: float) -> int:
    """Whole steps that cover ``duration`` (at least one)."""
    return max(1, math.ceil(duration / dt - 1e-9))


def plateau_step(remaining: torch.Tensor, active: torch.Tensor, duration: int) -> torch.Tensor:
    """Steps left of every countdown: ``duration`` where ``active``, else one fewer.

    Used for plateaus (restarted by a dendritic spike) and for the coincidence window of
    presynaptic spikes.
    """
    return torch.where(active, duration, (remaining - 1).clamp(min=0))


class ActiveSegments(_SynapticInput):
    """Distal segments that turn coincident input into plateau currents.

    Args:
        segments: Segments per destination cell.
        synapses: Synapse slots per segment.
        activation_threshold: Active connected synapses that fire a dendritic spike.
        plateau: Plateau duration, in the unit of ``dt``; lasts ``ceil(plateau / dt)`` steps.
        coincidence: How long a presynaptic spike keeps its synapses active, in the unit of
            ``dt`` (``ceil(coincidence / dt)`` steps); ``None`` means the spike's step only.
        connected: Permanence from which a synapse counts.
        amplitude: Current a cell receives while one of its segments is in a plateau.
        presynaptic: Initial source indices ``(dst.size, segments, synapses)``; empty if None.
        permanence: Initial permanences of the same shape; zeros if None.
    """

    connectivity = "segments"
    needs_weights = False

    def __init__(
        self,
        *,
        segments: int,
        synapses: int,
        activation_threshold: int,
        plateau: float,
        coincidence: float | None = None,
        connected: float = 0.5,
        amplitude: float = 1.0,
        presynaptic: torch.Tensor | None = None,
        permanence: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if segments < 1 or synapses < 1:
            raise ValueError(f"segments and synapses must be positive, got {segments}, {synapses}")
        if not 1 <= activation_threshold <= synapses:
            raise ValueError(
                f"activation_threshold must be in [1, {synapses}], got {activation_threshold}"
            )
        if plateau <= 0:
            raise ValueError(f"plateau must be positive, got {plateau}")
        if coincidence is not None and coincidence <= 0:
            raise ValueError(f"coincidence must be positive, got {coincidence}")
        self.segments, self.synapses = segments, synapses
        self.activation_threshold, self.plateau = activation_threshold, plateau
        self.coincidence = coincidence
        self.connected, self.amplitude = connected, amplitude
        self.initial = (presynaptic, permanence)

    def validate(self, syn: SynapseGroup) -> None:
        """Check the shape and range of initial segments."""
        shape = (syn.dst.size, self.segments, self.synapses)
        for name, value in zip(("presynaptic", "permanence"), self.initial, strict=True):
            if value is not None and tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} of {syn.name} must have shape {shape}, got {tuple(value.shape)}"
                )
        presynaptic = self.initial[0]
        if presynaptic is not None and bool(
            ((presynaptic < -1) | (presynaptic >= syn.src.size)).any()
        ):
            raise ValueError(
                f"presynaptic of {syn.name} must be -1 or index {syn.src.size} sources"
            )

    def initialize(self, syn: SynapseGroup) -> None:
        """Allocate segments and plateau timers."""
        super().initialize(syn)
        net = syn.net
        shape = (syn.dst.size, self.segments, self.synapses)
        presynaptic, permanence = self.initial
        syn.presynaptic = (
            torch.full(shape, -1, dtype=torch.long, device=net.device)
            if presynaptic is None
            else presynaptic.to(device=net.device, dtype=torch.long).clone()
        )
        syn.permanence = (
            torch.zeros(shape, dtype=net.dtype, device=net.device)
            if permanence is None
            else permanence.to(device=net.device, dtype=net.dtype).clone()
        )
        batch = () if net.batch_size is None else (net.batch_size,)
        syn.plateau_steps = torch.zeros(
            (*batch, syn.dst.size, self.segments), dtype=torch.long, device=net.device
        )
        syn.active_segments = syn.plateau_steps > 0
        syn.segment_potential = torch.zeros_like(syn.plateau_steps)
        syn.pre_recent = syn.src.state(0, dtype=torch.long)
        self.duration = _steps(self.plateau, net.dt)
        self.window = 1 if self.coincidence is None else _steps(self.coincidence, net.dt)

    def forward(self, syn: SynapseGroup) -> None:
        """Detect dendritic spikes, advance the plateaus and write the current."""
        syn.pre_recent = plateau_step(syn.pre_recent, syn.pre_spike, self.window)
        counts, syn.segment_potential = segment_counts(
            syn.pre_recent > 0, syn.presynaptic, syn.permanence, self.connected
        )
        syn.active_segments = counts >= self.activation_threshold
        syn.plateau_steps = plateau_step(syn.plateau_steps, syn.active_segments, self.duration)
        super().forward(syn)

    def current(self, syn: SynapseGroup) -> torch.Tensor:
        """``amplitude`` for cells with a segment in its plateau, else 0."""
        return self.amplitude * (syn.plateau_steps > 0).any(-1).to(syn.net.dtype)
