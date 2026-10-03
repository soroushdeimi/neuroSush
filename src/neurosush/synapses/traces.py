"""Delayed spike lookup for synapses and the spike traces used by plasticity."""

from __future__ import annotations

from typing import Literal

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import SynapseGroup
from neurosush.core.order import Order


def trace_step(
    trace: torch.Tensor,
    spikes: torch.Tensor,
    *,
    tau: float,
    dt: float,
    scale: float = 1.0,
    nearest: bool = False,
) -> torch.Tensor:
    """Decay the trace by ``dt / tau``, then add ``scale`` for each spike.

    With ``nearest`` a spike sets the trace to ``scale`` instead of adding to it, so the
    trace only remembers the latest spike.
    """
    decayed = trace * (1 - dt / tau)
    if nearest:
        return torch.where(spikes.bool(), torch.full_like(decayed, scale), decayed)
    return decayed.add_(spikes, alpha=scale)


class SpikeGather(Behavior):
    """Reads ``syn.pre_spike`` (and ``syn.post_spike``) through the synapse delays.

    The source group needs an :class:`~neurosush.neurons.axon.Axon`; its spikes are read
    with ``src_delay``. When the destination has an Axon too, its spikes are read with
    ``dst_delay`` into ``syn.post_spike``, which traces and plasticity need.
    """

    order = Order.SPIKE_GATHER
    independent_ok = True

    def initialize(self, syn: SynapseGroup) -> None:
        """Check for the source axon and read the initial spikes."""
        if not hasattr(syn.src, "spike_history"):
            raise RuntimeError(f"SpikeGather on {syn.name} needs an Axon on {syn.src.name}")
        self.post = hasattr(syn.dst, "spike_history")
        self.forward(syn)

    def reset_state(self, syn: SynapseGroup) -> None:
        """Silence the gathered spikes."""
        syn.pre_spike.zero_()
        if self.post:
            syn.post_spike.zero_()

    def forward(self, syn: SynapseGroup) -> None:
        """Read this step's delayed spikes."""
        syn.pre_spike = syn.src.spike_history.read(syn.src_delay)
        if self.post:
            syn.post_spike = syn.dst.spike_history.read(syn.dst_delay)

    def graph_ready(self, syn: SynapseGroup) -> bool:
        """Ready when the histories read from have depth 1: no per-neuron delay lookup."""
        if syn.src.spike_history.depth != 1:
            return False
        return not self.post or syn.dst.spike_history.depth == 1


class Traces(Behavior):
    """Exponential traces of the gathered pre- and postsynaptic spikes.

    Args:
        tau_pre: Presynaptic trace time constant.
        tau_post: Postsynaptic trace time constant; defaults to ``tau_pre``.
        scale: Increment per spike (the value the trace is set to, for ``"nearest"``).
        interaction: ``"all"`` adds ``scale`` at every spike, so every earlier spike counts
            (all-to-all STDP); ``"nearest"`` sets the trace to ``scale``, so only the latest
            spike counts (nearest-spike STDP, Masquelier et al. 2008, Diehl and Cook 2015).
    """

    order = Order.TRACE
    independent_ok = True
    graph_safe = True

    def __init__(
        self,
        *,
        tau_pre: float,
        tau_post: float | None = None,
        scale: float = 1.0,
        interaction: Literal["all", "nearest"] = "all",
    ) -> None:
        tau_post = tau_pre if tau_post is None else tau_post
        for name, tau in (("tau_pre", tau_pre), ("tau_post", tau_post)):
            if tau <= 0:
                raise ValueError(f"{name} must be positive, got {tau}")
        if interaction not in ("all", "nearest"):
            raise ValueError(f"interaction must be 'all' or 'nearest', got {interaction!r}")
        self.tau_pre, self.tau_post, self.scale = tau_pre, tau_post, scale
        self.interaction = interaction

    def initialize(self, syn: SynapseGroup) -> None:
        """Allocate both traces."""
        if not hasattr(syn, "post_spike"):
            raise RuntimeError(
                f"Traces on {syn.name} needs SpikeGather and an Axon on {syn.dst.name}"
            )
        syn.pre_trace = syn.src.state()
        syn.post_trace = syn.dst.state()

    def reset_state(self, syn: SynapseGroup) -> None:
        """Zero both traces."""
        syn.pre_trace.zero_()
        syn.post_trace.zero_()

    def forward(self, syn: SynapseGroup) -> None:
        """Update both traces."""
        dt = syn.net.dt
        nearest = self.interaction == "nearest"
        syn.pre_trace = trace_step(
            syn.pre_trace, syn.pre_spike, tau=self.tau_pre, dt=dt, scale=self.scale, nearest=nearest
        )
        syn.post_trace = trace_step(
            syn.post_trace,
            syn.post_spike,
            tau=self.tau_post,
            dt=dt,
            scale=self.scale,
            nearest=nearest,
        )
