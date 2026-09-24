"""Delayed spike lookup for synapses and the spike traces used by plasticity."""

from __future__ import annotations

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import SynapseGroup
from neurosush.core.order import Order


def trace_step(
    trace: torch.Tensor, spikes: torch.Tensor, *, tau: float, dt: float, scale: float = 1.0
) -> torch.Tensor:
    """Decay the trace by ``dt / tau``, then add ``scale`` for each spike."""
    return trace * (1 - dt / tau) + scale * spikes.to(trace.dtype)


class SpikeGather(Behavior):
    """Reads ``syn.pre_spike`` (and ``syn.post_spike``) through the synapse delays.

    The source group needs an :class:`~neurosush.neurons.axon.Axon`; its spikes are read
    with ``src_delay``. When the destination has an Axon too, its spikes are read with
    ``dst_delay`` into ``syn.post_spike``, which traces and plasticity need.
    """

    order = Order.SPIKE_GATHER

    def initialize(self, syn: SynapseGroup) -> None:
        """Check for the source axon and read the initial spikes."""
        if not hasattr(syn.src, "spike_history"):
            raise RuntimeError(f"SpikeGather on {syn.name} needs an Axon on {syn.src.name}")
        self.post = hasattr(syn.dst, "spike_history")
        self.forward(syn)

    def forward(self, syn: SynapseGroup) -> None:
        """Read this step's delayed spikes."""
        syn.pre_spike = syn.src.spike_history.read(syn.src_delay)
        if self.post:
            syn.post_spike = syn.dst.spike_history.read(syn.dst_delay)


class Traces(Behavior):
    """Exponential traces of the gathered pre- and postsynaptic spikes.

    Args:
        tau_pre: Presynaptic trace time constant.
        tau_post: Postsynaptic trace time constant; defaults to ``tau_pre``.
        scale: Increment per spike.
    """

    order = Order.TRACE

    def __init__(self, *, tau_pre: float, tau_post: float | None = None, scale: float = 1.0):
        tau_post = tau_pre if tau_post is None else tau_post
        for name, tau in (("tau_pre", tau_pre), ("tau_post", tau_post)):
            if tau <= 0:
                raise ValueError(f"{name} must be positive, got {tau}")
        self.tau_pre, self.tau_post, self.scale = tau_pre, tau_post, scale

    def initialize(self, syn: SynapseGroup) -> None:
        """Allocate both traces."""
        if not hasattr(syn, "post_spike"):
            raise RuntimeError(
                f"Traces on {syn.name} needs SpikeGather and an Axon on {syn.dst.name}"
            )
        syn.pre_trace = syn.src.state()
        syn.post_trace = syn.dst.state()

    def forward(self, syn: SynapseGroup) -> None:
        """Update both traces."""
        dt = syn.net.dt
        syn.pre_trace = trace_step(
            syn.pre_trace, syn.pre_spike, tau=self.tau_pre, dt=dt, scale=self.scale
        )
        syn.post_trace = trace_step(
            syn.post_trace, syn.post_spike, tau=self.tau_post, dt=dt, scale=self.scale
        )
