"""Homeostatic mechanisms that keep firing rates and voltages in range."""

from __future__ import annotations

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import NeuronGroup
from neurosush.core.order import Order


class ActivityHomeostasis(Behavior):
    """Moves each neuron's threshold so it fires ``target_spikes`` times per ``window`` steps.

    A spike counts +1 and a silent step ``-target_spikes / (window - target_spikes)``, so the
    counter ends a window at 0 exactly when the neuron hits its target. At the end of each
    window the threshold rises by ``counter * rate`` (falls when negative), the counter
    resets and ``rate *= decay``.

    Args:
        target_spikes: Desired spikes per window, ``0 < target_spikes < window``.
        window: Window length in steps.
        rate: Threshold change per unit of activity.
        decay: Factor applied to ``rate`` after every window, in ``(0, 1]``.
    """

    order = Order.ACTIVITY_HOMEOSTASIS

    def __init__(self, *, target_spikes: int, window: int, rate: float, decay: float = 1.0) -> None:
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

    def initialize(self, group: NeuronGroup) -> None:
        """Check for a per-neuron threshold and allocate the activity counter."""
        if not hasattr(group, "threshold") or not isinstance(group.threshold, torch.Tensor):
            raise RuntimeError(f"ActivityHomeostasis on {group.name} needs a threshold (LIF model)")
        self.activity = group.vector()

    def forward(self, group: NeuronGroup) -> None:
        """Count activity and adjust the threshold at the end of each window."""
        # the threshold is shared by all samples, so a batch counts its mean activity
        s = group.spikes.to(self.activity.dtype).reshape(-1, group.size).mean(0)
        self.activity = self.activity + s - (1 - s) * self.silent_penalty
        if group.net.iteration % self.window == 0:
            group.threshold = group.threshold + self.activity * self.rate
            self.activity.zero_()
            self.rate *= self.decay

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        """The activity counter and the decayed rate."""
        return {"activity": self.activity.clone(), "rate": self.rate}

    def load_state_dict(self, state: dict[str, torch.Tensor | float]) -> None:
        """Restore the counter and rate saved by :meth:`state_dict`."""
        activity = torch.as_tensor(state["activity"])
        if activity.shape != self.activity.shape:
            raise ValueError(
                f"activity shape must be {tuple(self.activity.shape)}, got {tuple(activity.shape)}"
            )
        self.activity = activity.to(self.activity)
        self.rate = float(state["rate"])


class VoltageHomeostasis(Behavior):
    """Pushes voltages back into ``[v_min, v_max]`` through an accumulating exhaustion term.

    ``exhaustion += rate * (v - v_max)`` above the band and ``rate * (v - v_min)`` below it;
    then ``v -= exhaustion``.

    Args:
        target: Sets both ``v_min`` and ``v_max``.
        v_min: Lower edge of the band (with ``v_max``, instead of ``target``).
        v_max: Upper edge of the band.
        rate: Adaptation speed.
    """

    order = Order.VOLTAGE_HOMEOSTASIS

    def __init__(
        self,
        *,
        target: float | None = None,
        v_min: float | None = None,
        v_max: float | None = None,
        rate: float = 0.001,
    ) -> None:
        if (target is not None and (v_min is not None or v_max is not None)) or (
            target is None and (v_min is None or v_max is None)
        ):
            raise ValueError("give either target, or both v_min and v_max")

        if target is not None:
            v_min = v_max = target
        self.v_min, self.v_max = v_min, v_max

        if self.v_min > self.v_max:
            raise ValueError(f"v_min must be <= v_max, got v_min={self.v_min}, v_max={self.v_max}")
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")

        self.rate = rate

    def initialize(self, group: NeuronGroup) -> None:
        """Allocate the exhaustion term."""
        group.exhaustion = group.state()

    def forward(self, group: NeuronGroup) -> None:
        """Update the exhaustion term and apply it to the membrane."""
        v = group.v
        above = torch.clamp(v - self.v_max, min=0.0)
        below = torch.clamp(v - self.v_min, max=0.0)
        group.exhaustion = group.exhaustion + (above + below) * self.rate
        group.v = group.v - group.exhaustion
