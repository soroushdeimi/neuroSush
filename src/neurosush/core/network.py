"""Network objects and ordered behavior scheduling."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING

import torch

from neurosush.core.behavior import Behavior

if TYPE_CHECKING:
    from neurosush.core.buffers import ArrivalBuffer, HistoryBuffer
    from neurosush.neurons.models import LIF
    from neurosush.synapses.currents import _SynapticInput


class Compartment(str, Enum):
    """Dendritic compartments receiving synaptic input."""

    PROXIMAL = "proximal"
    DISTAL = "distal"
    APICAL = "apical"


def _check_name(name: str, groups: Iterable[NeuronGroup | SynapseGroup]) -> None:
    if any(group.name == name for group in groups):
        raise ValueError(f"name {name!r} is already used in this network")


class Network:
    """Own simulation objects, randomness and the behavior schedule.

    Args:
        dt: Time step in the same unit as every time constant.
        dtype: Floating point dtype for network tensors.
        device: Device for network tensors and the random generator.
        seed: Random seed; None draws a nondeterministic seed.
        batch_size: Number of samples simulated in parallel. ``None`` (default) keeps state
            unbatched, shaped ``(size,)``; an int ``B`` shapes every state ``(B, size)``
            while weights and thresholds stay shared.
        behaviors: Behaviors attached to the network.
    """

    # Payoff and Dopamine set the network's modulation state.
    payoff: float
    dopamine: float

    def __init__(
        self,
        *,
        dt: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        batch_size: int | None = None,
        behaviors: Iterable[Behavior] = (),
    ) -> None:
        if batch_size is not None and (isinstance(batch_size, bool) or batch_size < 1):
            raise ValueError(f"batch_size must be a positive int or None, got {batch_size!r}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if not dtype.is_floating_point:
            raise TypeError(f"dtype must be a floating point type, got {dtype}")
        self.dt = float(dt)
        self.dtype = dtype
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.generator = torch.Generator(device=self.device)
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)
        self.groups: list[NeuronGroup] = []
        self.synapses: list[SynapseGroup] = []
        self.iteration = 0
        self.initialized = False
        self.schedule: list[tuple[Network | NeuronGroup | SynapseGroup, Behavior]] = []
        self._preparing: list[tuple[Network | NeuronGroup | SynapseGroup, Behavior]] = []
        self._registrations: list[tuple[Network | NeuronGroup | SynapseGroup, Behavior]] = []
        self._attached_ids: set[int] = set()
        self.behaviors = self._attach(self, behaviors)

    def _attach(
        self, host: Network | NeuronGroup | SynapseGroup, behaviors: Iterable[Behavior]
    ) -> tuple[Behavior, ...]:
        if self.initialized:
            raise RuntimeError("network is already initialized")
        attached = tuple(behaviors)
        pending_ids: set[int] = set()
        for behavior in attached:
            if not isinstance(behavior, Behavior):
                raise TypeError(f"expected a Behavior, got {type(behavior).__name__}")
            if not isinstance(getattr(behavior, "order", None), int):
                raise TypeError(f"{type(behavior).__name__} must define an integer order")
            identity = id(behavior)
            if identity in self._attached_ids or identity in pending_ids:
                raise ValueError(f"{behavior!r} is already attached to an object of this network")
            pending_ids.add(identity)
        self._attached_ids.update(pending_ids)
        self._registrations.extend((host, behavior) for behavior in attached)
        return attached

    def initialize(self) -> None:
        """Initialize all behaviors in execution order once."""
        if self.initialized:
            raise RuntimeError("network is already initialized")
        # Stable sorting preserves registration order for ties.
        self.schedule = sorted(self._registrations, key=lambda pair: pair[1].order)
        self._preparing = [
            (host, behavior)
            for host, behavior in self.schedule
            if type(behavior).prepare is not Behavior.prepare
        ]
        self.initialized = True
        for host, behavior in self.schedule:
            behavior.initialize(host)

    def step(self) -> None:
        """Advance one iteration, running each enabled behavior."""
        if not self.initialized:
            self.initialize()
        self.iteration += 1
        for host, behavior in self._preparing:
            if behavior.enabled:
                behavior.prepare(host)
        for host, behavior in self.schedule:
            if behavior.enabled:
                behavior.forward(host)

    def run(self, steps: int) -> None:
        """Advance the network by the requested number of steps."""
        if steps < 0:
            raise ValueError(f"steps must be non-negative, got {steps}")
        for _ in range(steps):
            self.step()


class NeuronGroup:
    """A shaped population of neurons belonging to one network.

    Args:
        net: Network that owns the group.
        shape: Positive int n for (1, 1, n), or three positive dimensions.
        behaviors: Behaviors attached to the group.
        name: Unique neuron group name; None generates a name.
        tags: Labels stored as a frozenset.
        inhibitory: Whether outgoing currents are made negative.
    """

    # LIF and its subclasses set membrane state and parameters.
    v: torch.Tensor
    tau: float
    resistance: float
    v_rest: float
    v_reset: float
    threshold: torch.Tensor
    model: LIF
    # LIF and SpikeInput set spikes; SpikeInput also sets the label.
    spikes: torch.Tensor
    label: object
    # AdaptiveELIF sets the adaptation current.
    omega: torch.Tensor
    # LIF and DendriteIntegration set the input current.
    I: torch.Tensor  # noqa: E741 - Existing public name for current.
    # DendriteStructure sets the compartment currents.
    dendrite: dict[Compartment, ArrivalBuffer]
    I_proximal: torch.Tensor
    I_distal: torch.Tensor
    I_apical: torch.Tensor
    # Axon records spike history.
    spike_history: HistoryBuffer
    # VoltageHomeostasis sets exhaustion.
    exhaustion: torch.Tensor
    # MinicolumnInhibition sets the steps of inhibition left per minicolumn.
    column_inhibition: torch.Tensor

    def __init__(
        self,
        net: Network,
        shape: int | tuple[int, int, int],
        behaviors: Iterable[Behavior] = (),
        *,
        name: str | None = None,
        tags: Iterable[str] = (),
        inhibitory: bool = False,
    ) -> None:
        if isinstance(shape, int) and not isinstance(shape, bool):
            shape = (1, 1, shape)
        if (
            not isinstance(shape, tuple)
            or len(shape) != 3
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in shape
            )
        ):
            raise ValueError(
                f"shape must be a positive int or a tuple of three positive ints, got {shape!r}"
            )
        self.shape = shape
        self.name = f"ng{len(net.groups)}" if name is None else name
        _check_name(self.name, net.groups)
        self.net = net
        self.tags = frozenset(tags)
        self.inhibitory = bool(inhibitory)
        self.afferent: dict[Compartment, list[SynapseGroup]] = {
            compartment: [] for compartment in Compartment
        }
        self.efferent: dict[Compartment, list[SynapseGroup]] = {
            compartment: [] for compartment in Compartment
        }
        self.behaviors = net._attach(self, behaviors)
        net.groups.append(self)

    @property
    def depth(self) -> int:
        """Number of depth planes."""
        return self.shape[0]

    @property
    def height(self) -> int:
        """Number of rows per depth plane."""
        return self.shape[1]

    @property
    def width(self) -> int:
        """Number of neurons per row."""
        return self.shape[2]

    @property
    def size(self) -> int:
        """Total number of neurons."""
        return self.depth * self.height * self.width

    @property
    def state_shape(self) -> tuple[int, ...]:
        """Shape of per-sample state: ``(size,)``, or ``(batch_size, size)`` when batched."""
        batch = self.net.batch_size
        return (self.size,) if batch is None else (batch, self.size)

    def vector(self, fill: float = 0.0, dtype: torch.dtype | None = None) -> torch.Tensor:
        """A per-neuron parameter tensor of shape ``(size,)``, shared by every sample."""
        return torch.full((self.size,), fill, dtype=dtype or self.net.dtype, device=self.net.device)

    def state(self, fill: float | bool = 0.0, dtype: torch.dtype | None = None) -> torch.Tensor:
        """A per-sample state tensor of shape :attr:`state_shape` filled with ``fill``."""
        return torch.full(
            self.state_shape, fill, dtype=dtype or self.net.dtype, device=self.net.device
        )

    def rand(self) -> torch.Tensor:
        """Uniform samples in ``[0, 1)`` of shape :attr:`state_shape`."""
        return torch.rand(
            self.state_shape,
            generator=self.net.generator,
            dtype=self.net.dtype,
            device=self.net.device,
        )

    def randn(self) -> torch.Tensor:
        """Standard normal samples of shape :attr:`state_shape`."""
        return torch.randn(
            self.state_shape,
            generator=self.net.generator,
            dtype=self.net.dtype,
            device=self.net.device,
        )

    def __repr__(self) -> str:
        return f"NeuronGroup({self.name!r}, shape={self.shape!r})"


class SynapseGroup:
    """Connections between neuron groups targeting one compartment.

    Args:
        net: Network that owns the synapse group and both neuron groups.
        src: Source neuron group.
        dst: Destination neuron group.
        behaviors: Behaviors attached to the synapse group.
        compartment: Target compartment of dst.
        name: Unique synapse group name; None generates a name.
        tags: Labels stored as a frozenset.
    """

    # WeightInit sets sparse edge indices.
    src_idx: torch.Tensor
    dst_idx: torch.Tensor
    # Synaptic input behaviors set connectivity, input and destination current.
    connectivity: str
    input: _SynapticInput
    I: torch.Tensor  # noqa: E741 - Existing public name for current.
    # SpikeGather and synaptic inputs set presynaptic spikes.
    pre_spike: torch.Tensor
    # SpikeGather sets postsynaptic spikes.
    post_spike: torch.Tensor
    # Traces sets the pre- and postsynaptic traces.
    pre_trace: torch.Tensor
    post_trace: torch.Tensor
    # RSTDP sets the eligibility trace.
    eligibility: torch.Tensor
    # ActiveSegments sets the segments, their plateaus and their activity.
    presynaptic: torch.Tensor
    permanence: torch.Tensor
    plateau_steps: torch.Tensor
    active_segments: torch.Tensor
    segment_potential: torch.Tensor
    pre_recent: torch.Tensor
    # SegmentLearning sets when segments started and were used, what started them, and the
    # last spike and win of every cell.
    segment_start: torch.Tensor
    segment_used: torch.Tensor
    activation_synapses: torch.Tensor
    last_spike: torch.Tensor
    last_win: torch.Tensor

    def __init__(
        self,
        net: Network,
        src: NeuronGroup,
        dst: NeuronGroup,
        behaviors: Iterable[Behavior] = (),
        *,
        compartment: Compartment | str = Compartment.PROXIMAL,
        name: str | None = None,
        tags: Iterable[str] = (),
    ) -> None:
        if src.net is not net or dst.net is not net:
            raise ValueError("src and dst must belong to the same network as the synapse")
        try:
            self.compartment = Compartment(compartment)
        except ValueError:
            raise ValueError(
                f"compartment must be one of {[c.value for c in Compartment]}, got {compartment!r}"
            ) from None
        self.name = f"sg{len(net.synapses)}" if name is None else name
        _check_name(self.name, net.synapses)
        self.net = net
        self.src = src
        self.dst = dst
        self.tags = frozenset(tags)
        self.weights: torch.Tensor | None = None
        self.src_delay = torch.zeros(src.size, dtype=torch.long, device=net.device)
        self.dst_delay = torch.zeros(dst.size, dtype=torch.long, device=net.device)
        self.behaviors = net._attach(self, behaviors)
        dst.afferent[self.compartment].append(self)
        src.efferent[self.compartment].append(self)
        net.synapses.append(self)

    def __repr__(self) -> str:
        return f"SynapseGroup({self.name!r}, {self.src.name} -> {self.dst.name})"
