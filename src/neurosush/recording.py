"""Record host attributes over time."""

from __future__ import annotations

from typing import Any

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order


class Recorder(Behavior):
    """Stores copies of attributes of its host every ``interval`` steps.

    Runs after every other behavior, so it sees the state at the end of a step. Attach it
    to a network (for example ``"dopamine"``), a neuron group (``"v"``, ``"spikes"``) or a
    synapse group (``"weights"``).

    Args:
        *attributes: Names of the host attributes to record.
        interval: Record on steps that are multiples of ``interval``.
        device: Where the copies are kept; the CPU by default, to spare device memory.
    """

    order = Order.RECORD

    def __init__(
        self, *attributes: str, interval: int = 1, device: str | torch.device = "cpu"
    ) -> None:
        if not attributes:
            raise ValueError("give at least one attribute to record")
        if interval < 1:
            raise ValueError(f"interval must be positive, got {interval}")
        self.attributes, self.interval, self.device = attributes, interval, torch.device(device)
        self.reset()

    def reset(self) -> None:
        """Drop everything recorded so far."""
        self.steps: list[int] = []
        self._values: dict[str, list[Any]] = {name: [] for name in self.attributes}

    def initialize(self, host: Network | NeuronGroup | SynapseGroup) -> None:
        """Check that every attribute exists once all other behaviors are initialized."""
        missing = [name for name in self.attributes if not hasattr(host, name)]
        if missing:
            raise RuntimeError(f"Recorder on {_name(host)}: no attribute(s) {missing}")

    def forward(self, host: Network | NeuronGroup | SynapseGroup) -> None:
        """Copy the attributes on recording steps."""
        iteration = (host if isinstance(host, Network) else host.net).iteration
        if iteration % self.interval:
            return
        self.steps.append(iteration)
        for name in self.attributes:
            value = getattr(host, name)
            if isinstance(value, torch.Tensor):
                value = value.detach().to(self.device, copy=True)
            self._values[name].append(value)

    def get(self, name: str) -> torch.Tensor:
        """Everything recorded for ``name``, stacked along a new first (time) dimension."""
        if name not in self._values:
            raise KeyError(f"{name!r} is not recorded; recorded: {list(self.attributes)}")
        values = self._values[name]
        if values and isinstance(values[0], torch.Tensor):
            return torch.stack(values)
        return torch.tensor(values, device=self.device)

    def __repr__(self) -> str:
        names = ", ".join(repr(name) for name in self.attributes)
        return f"Recorder({names}, interval={self.interval})"


class SpikeCounter(Behavior):
    """Counts each neuron's spikes in ``group.spike_count`` (float, the group's state shape).

    Adds this step's spikes once per step, so after a sample ``spike_count`` is the response
    of every neuron (per sample in a batch). Unlike :class:`Recorder` it is graph-safe: it
    runs at ``Order.ACTIVITY_HOMEOSTASIS``, right after the spikes of the step are final
    (``Fire`` and the inputs run at ``Order.FIRE``) and below ``Order.RECORD``, so a
    :class:`~neurosush.core.graph.GraphStepper` captures it. :meth:`reset_state` zeros the
    count, which starts the next sample.
    """

    order = Order.ACTIVITY_HOMEOSTASIS
    graph_safe = True

    def initialize(self, group: NeuronGroup) -> None:
        """Allocate ``group.spike_count``; the group must have spikes (a neuron model or input)."""
        if not hasattr(group, "spikes"):
            raise RuntimeError(f"SpikeCounter on {group.name} needs spikes (a neuron model)")
        group.spike_count = group.state()

    def forward(self, group: NeuronGroup) -> None:
        """Add this step's spikes."""
        group.spike_count = group.spike_count + group.spikes.to(group.spike_count.dtype)

    def reset_state(self, group: NeuronGroup) -> None:
        """Zero the count."""
        group.spike_count.zero_()


def _name(host: Network | NeuronGroup | SynapseGroup) -> str:
    return "the network" if isinstance(host, Network) else host.name
