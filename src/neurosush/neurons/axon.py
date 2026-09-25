"""Axon behavior for recording spike history."""

from __future__ import annotations

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.buffers import HistoryBuffer
from neurosush.core.network import NeuronGroup
from neurosush.core.order import Order


class Axon(Behavior):
    """Records spike history for a neuron group with delays.

    Args:
        max_delay: Maximum delay in steps.
    """

    order = Order.AXON

    def __init__(self, *, max_delay: int = 1) -> None:
        if max_delay < 1:
            raise ValueError(f"max_delay must be at least 1, got {max_delay}")
        self.max_delay = max_delay

    def initialize(self, group: NeuronGroup) -> None:
        """Check the delays this history must serve and allocate it on the group."""
        for kind, links in (("src_delay", group.efferent), ("dst_delay", group.afferent)):
            for synapses in links.values():
                for syn in synapses:
                    longest = int(getattr(syn, kind).max())
                    if longest >= self.max_delay:
                        raise ValueError(
                            f"{kind} must be less than max_delay={self.max_delay} for "
                            f"{syn.name}, got {longest}"
                        )
        group.spike_history = HistoryBuffer(
            self.max_delay,
            group.size,
            dtype=torch.bool,
            device=group.net.device,
            batch=group.net.batch_size,
        )

    def forward(self, group: NeuronGroup) -> None:
        """Push current spikes into the history buffer."""
        group.spike_history.push(group.spikes)

    def graph_ready(self, group: NeuronGroup) -> bool:
        """Ready without delays: the ring head is a Python int that a graph would freeze."""
        return self.max_delay == 1
