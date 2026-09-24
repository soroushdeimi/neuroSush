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


def _name(host: Network | NeuronGroup | SynapseGroup) -> str:
    return "the network" if isinstance(host, Network) else host.name
