"""Input groups whose spikes come from data instead of neuron dynamics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import NeuronGroup
from neurosush.core.order import Order


class SpikeInput(Behavior):
    """Drive a group from spike frames.

    Args:
        frames: An iterable of frames (as tensors) or (frame, label) tuples.
            Items may come from ``neurosush.data.spike_frames``; use
            ``itertools.cycle`` for endless input.
    """

    order = Order.FIRE
    graph_safe = True

    def __init__(self, frames: Iterable[torch.Tensor | tuple[torch.Tensor, Any]]) -> None:
        self.frames = frames

    def initialize(self, group: NeuronGroup) -> None:
        """Allocate spikes, the staging tensor and the label on the group."""
        self._stream = iter(self.frames)
        group.spikes = group.state(False, dtype=torch.bool)
        self._staged = group.state(False, dtype=torch.bool)
        group.label = None

    def prepare(self, group: NeuronGroup) -> None:
        """Read the next frame and stage it into the fixed-address tensor."""
        try:
            item = next(self._stream)
        except StopIteration:
            raise RuntimeError(
                f"SpikeInput on {group.name} ran out of frames at iteration {group.net.iteration}"
            ) from None

        if isinstance(item, tuple):
            frame, label = item
        else:
            frame, label = item, None

        if frame.numel() != group.size * (group.net.batch_size or 1):
            raise ValueError(
                f"SpikeInput on {group.name}: frame size {frame.numel()} "
                f"must match the group's state shape {group.state_shape}"
            )
        frame = frame.reshape(group.state_shape)

        self._staged.copy_(frame)  # copy_ converts dtype and device
        group.label = label

    def forward(self, group: NeuronGroup) -> None:
        """Publish the staged frame."""
        group.spikes = self._staged.clone()
