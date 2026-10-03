"""Input groups whose spikes come from data or rates instead of neuron dynamics."""

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
    independent_ok = True
    graph_safe = True

    def __init__(self, frames: Iterable[torch.Tensor | tuple[torch.Tensor, Any]]) -> None:
        self.frames = frames

    def initialize(self, group: NeuronGroup) -> None:
        """Allocate spikes, the staging tensor and the label on the group."""
        self._stream = iter(self.frames)
        group.spikes = group.state(False, dtype=torch.bool)
        self._staged = group.state(False, dtype=torch.bool)
        group.label = None

    def reset_state(self, group: NeuronGroup) -> None:
        """Silence the group; the stream of frames is not rewound."""
        group.spikes.zero_()
        self._staged.zero_()

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


class PoissonInput(Behavior):
    """Independent random spikes at per-neuron rates, drawn on the network's device.

    Every step each neuron spikes with probability ``rates * dt`` (a Bernoulli
    approximation of a Poisson process, close to it while ``rates * dt`` is small). The
    rates live on the group as ``group.rates`` (shape :attr:`~NeuronGroup.state_shape`, in
    spikes per time unit of ``dt``); change them in place (``group.rates.copy_(...)``) to
    present a new stimulus, which also works while a CUDA graph replays the step.

    Args:
        rates: Initial rates: a number, or a tensor of shape ``(size,)`` or the state shape.
    """

    order = Order.FIRE
    independent_ok = True
    graph_safe = True

    def __init__(self, rates: float | torch.Tensor = 0.0) -> None:
        if isinstance(rates, torch.Tensor):
            if (rates < 0).any():
                raise ValueError("rates must be non-negative")
        elif rates < 0:
            raise ValueError(f"rates must be non-negative, got {rates}")
        self.rates = rates

    def initialize(self, group: NeuronGroup) -> None:
        """Allocate ``group.rates`` and ``group.spikes``."""
        group.rates = group.state()
        if isinstance(self.rates, torch.Tensor):
            if self.rates.shape not in ((group.size,), group.state_shape):
                raise ValueError(
                    f"rates must have shape ({group.size},) or {group.state_shape}, "
                    f"got {tuple(self.rates.shape)}"
                )
            group.rates.copy_(self.rates.expand(group.state_shape))
        else:
            group.rates.fill_(self.rates)
        group.spikes = group.state(False, dtype=torch.bool)

    def reset_state(self, group: NeuronGroup) -> None:
        """Silence the group; ``group.rates`` is kept."""
        group.spikes.zero_()

    def forward(self, group: NeuronGroup) -> None:
        """Draw this step's spikes from ``group.net.generator``."""
        uniform = self.drawn.get("uniform")
        if uniform is None:
            uniform = group.rand()
        group.spikes = uniform < group.rates * group.net.dt

    def draw(self, group: NeuronGroup) -> dict[str, torch.Tensor]:
        """This step's uniform samples, for the compiled stepper."""
        return {"uniform": group.rand()}
