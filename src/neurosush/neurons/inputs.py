"""Input groups whose spikes come from data or rates instead of neuron dynamics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons import dynamics


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


def _cuda_graph_ready(group: NeuronGroup) -> bool:
    from neurosush.core.graph import random_numbers_supported  # graph imports behaviors

    return group.net.generator.device.type == "cuda" and random_numbers_supported()


class PoissonDrive(Behavior):
    """Background drive from ``count`` independent Poisson sources per neuron (Brunel 2000).

    Each source fires at ``rate`` (per time unit of ``dt``) and every spike is a delta
    synapse that adds ``jump`` to the membrane. Per step each neuron receives
    ``jump * Poisson(count * rate * dt)``, drawn with ``torch.poisson`` from the network
    generator (or from :meth:`draw` under the compiled stepper). Runs at ``Order.NOISE``,
    after the neuron model integrated and before competition and ``Fire``. The increments
    have mean ``jump * count * rate * dt`` and variance ``jump^2 * count * rate * dt``.

    Args:
        count: Number of sources per neuron (positive; need not be an integer).
        rate: Rate of each source, at least 0.
        jump: Voltage added per source spike (negative for inhibitory sources).
    """

    order = Order.NOISE
    independent_ok = True

    def __init__(self, count: float, rate: float, jump: float) -> None:
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        if rate < 0:
            raise ValueError(f"rate must be non-negative, got {rate}")
        self.count, self.rate, self.jump = float(count), float(rate), float(jump)

    def initialize(self, group: NeuronGroup) -> None:
        """Allocate the Poisson mean of one step."""
        self._lam = group.state(self.count * self.rate * group.net.dt)

    def _draw(self, group: NeuronGroup) -> torch.Tensor:
        return torch.poisson(self._lam, generator=group.net.generator)

    def forward(self, group: NeuronGroup) -> None:
        """Add this step's jumps to the membrane."""
        events = self.drawn.get("events")
        if events is None:
            events = self._draw(group)
        group.v = group.v + self.jump * events

    def draw(self, group: NeuronGroup) -> dict[str, torch.Tensor]:
        """This step's event counts, for the compiled stepper."""
        return {"events": self._draw(group)}

    def moments(self, dt: float) -> tuple[float, float]:
        """Mean and variance of one step's voltage increment."""
        return dynamics.poisson_increment_moments(
            count=self.count, rate=self.rate, jump=self.jump, dt=dt
        )

    def compile_ready(self, group: NeuronGroup) -> bool:
        """Always ready: the compiled stepper draws the counts itself."""
        return True

    def graph_ready(self, group: NeuronGroup) -> bool:
        """Ready when the generator is on CUDA and this torch can replay its draws."""
        return _cuda_graph_ready(group)


class CorrelatedPoissonInput(Behavior):
    """Spike trains with pairwise correlation ``c``: the multiple interaction process.

    A mother train of rate ``rate / c`` is drawn per sample (per batch member), and every
    neuron copies each mother spike independently with probability ``c`` (Kuhn, Aertsen and
    Rotter 2003). Every neuron then fires at ``rate`` and two neurons' spike counts in a bin
    have correlation coefficient ``c`` (exactly so for a Poisson mother; with the per-step
    Bernoulli mother ``c (1 - p) / (1 - c p)``, ``p = rate dt / c``,
    see :func:`~neurosush.neurons.dynamics.correlated_pair_correlation`). ``c = 1`` makes
    identical trains. Sets ``group.spikes`` at ``Order.FIRE``.

    Args:
        rate: Rate of every neuron (positive, per time unit of ``dt``).
        correlation: Pairwise correlation ``c`` in ``(0, 1]``.
    """

    order = Order.FIRE
    independent_ok = True
    graph_safe = True

    def __init__(self, rate: float, correlation: float) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if not 0 < correlation <= 1:
            raise ValueError(f"correlation must be in (0, 1], got {correlation}")
        self.rate, self.correlation = float(rate), float(correlation)

    @property
    def mother_probability(self) -> float:
        """Per-step probability of a mother spike given the last initialized ``dt``."""
        return self._p_mother

    def initialize(self, group: NeuronGroup) -> None:
        """Check that a mother spike per step is a probability and allocate the spikes."""
        p_mother = self.rate / self.correlation * group.net.dt
        if p_mother > 1:
            raise ValueError(
                f"rate / correlation * dt must be at most 1, got {p_mother} "
                f"(rate={self.rate}, correlation={self.correlation}, dt={group.net.dt})"
            )
        self._p_mother = p_mother
        group.spikes = group.state(False, dtype=torch.bool)

    def reset_state(self, group: NeuronGroup) -> None:
        """Silence the group."""
        group.spikes.zero_()

    def _draw(self, group: NeuronGroup) -> dict[str, torch.Tensor]:
        net = group.net
        mother_shape = (*group.state_shape[:-1], 1)
        return {
            "mother": torch.rand(
                mother_shape, generator=net.generator, dtype=net.dtype, device=net.device
            ),
            "copy": group.rand(),
        }

    def draw(self, group: NeuronGroup) -> dict[str, torch.Tensor]:
        """This step's mother and copy samples, for the compiled stepper."""
        return self._draw(group)

    def forward(self, group: NeuronGroup) -> None:
        """Draw the mother spike of each sample and let every neuron copy it with prob. ``c``."""
        drawn = self.drawn if "mother" in self.drawn else self._draw(group)
        group.spikes = (drawn["mother"] < self._p_mother) & (drawn["copy"] < self.correlation)
