"""homeostatic mechanisms that keep firing rates and voltages in range."""

from __future__ import annotations

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import NeuronGroup
from neurosush.core.order import Order


class ActivityHomeostasis(Behavior):
    """Adjusts threshold to maintain a target spike rate.

    A spike counts +1, a silent step -target/(window-target), so the counter ends a
    window at 0 exactly when the neuron hits its target.
    """

    order = Order.ACTIVITY_HOMEOSTASIS

    def __init__(self, *, target_spikes: int, window: int, rate: float, decay: float = 1.0) -> None:
        """Initialize the ActivityHomeostasis behavior."""
        if not 0 < target_spikes < window:
            raise ValueError(
                f"target_spikes must be between 0 and {window} exclusive, got {target_spikes}"
            )
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if not 0 < decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {decay}")

        self.target_spikes = target_spikes
        self.window = window
        self.rate = rate
        self.decay = decay
        self.silent_penalty = target_spikes / (window - target_spikes)
        self.activity = None  # type: ignore

    def initialize(self, group: NeuronGroup) -> None:
        """Initialize the ActivityHomeostasis behavior."""
        if not hasattr(group, "threshold") or not isinstance(group.threshold, torch.Tensor):
            raise RuntimeError(f"ActivityHomeostasis on {group.name} needs a threshold (LIF model)")
        self.activity = group.vector()

    def forward(self, group: NeuronGroup) -> None:
        """Update the activity and threshold."""
        s = group.spikes.to(torch.float64)
        self.activity = self.activity + s - (1 - s) * self.silent_penalty
        if group.net.iteration % self.window == 0:
            group.threshold = group.threshold + self.activity * self.rate
            self.activity.zero_()
            self.rate *= self.decay


class VoltageHomeostasis(Behavior):
    """Adjusts voltages to stay within a specified range."""

    order = Order.VOLTAGE_HOMEOSTASIS

    def __init__(
        self,
        *,
        target: float | None = None,
        v_min: float | None = None,
        v_max: float | None = None,
        rate: float = 0.001,
    ) -> None:
        """Initialize the VoltageHomeostasis behavior."""
        if (target is not None and (v_min is not None or v_max is not None)) or (
            target is None and (v_min is None or v_max is None)
        ):
            raise ValueError("give either target, or both v_min and v_max")

        if target is not None:
            self.v_min = target
            self.v_max = target
        else:
            # v_min and v_max are both not None here
            self.v_min = v_min  # type: ignore
            self.v_max = v_max  # type: ignore

        if self.v_min > self.v_max:
            raise ValueError(f"v_min must be <= v_max, got v_min={self.v_min}, v_max={self.v_max}")
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")

        self.rate = rate
        self.exhaustion = None  # type: ignore

    def initialize(self, group: NeuronGroup) -> None:
        """Initialize the VoltageHomeostasis behavior."""
        group.exhaustion = group.vector()

    def forward(self, group: NeuronGroup) -> None:
        """Update the exhaustion and voltages."""
        v = group.v
        above = torch.clamp(v - self.v_max, min=0.0)
        below = torch.clamp(v - self.v_min, max=0.0)
        group.exhaustion = group.exhaustion + (above + below) * self.rate
        group.v = group.v - group.exhaustion
