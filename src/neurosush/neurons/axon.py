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
        """Allocate spike history buffer on the group."""
        for synapses in group.efferent.values():
            for syn in synapses:
                if int(syn.src_delay.max()) >= self.max_delay:
                    raise ValueError(f"src_delay must be less than {self.max_delay} for {syn.name}")
        group.spike_history = HistoryBuffer(
            self.max_delay, group.size, dtype=torch.bool, device=group.net.device
        )

    def forward(self, group: NeuronGroup) -> None:
        """Push current spikes into the history buffer."""
        group.spike_history.push(group.spikes)
