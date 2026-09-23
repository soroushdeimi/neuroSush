"""Network objects and ordered behavior scheduling."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import torch

from neurosush.core.behavior import Behavior


class Compartment(str, Enum):
    """Dendritic compartments receiving synaptic input."""

    PROXIMAL = "proximal"
    DISTAL = "distal"
    APICAL = "apical"


def _check_name(name: str, groups: Iterable[NeuronGroup | SynapseGroup]) -> None:
    if any(group.name == name for group in groups):
        raise ValueError(f"name {name!r} is already used")


class Network:
    """Own simulation objects, randomness and the behavior schedule."""

    def __init__(
        self,
        *,
        dt: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        behaviors: Iterable[Behavior] = (),
    ) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        self.dt = float(dt)
        self.dtype = dtype
        self.device = torch.device(device)
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
                raise TypeError("each behavior must be a Behavior")
            if not isinstance(getattr(behavior, "order", None), int):
                raise TypeError("behavior order must be an int")
            identity = id(behavior)
            if identity in self._attached_ids or identity in pending_ids:
                raise ValueError("behavior is already attached")
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
        self.initialized = True
        for host, behavior in self.schedule:
            behavior.initialize(host)

    def step(self) -> None:
        """Advance one iteration, running each enabled behavior."""
        if not self.initialized:
            self.initialize()
        self.iteration += 1
        for host, behavior in self.schedule:
            if behavior.enabled:
                behavior.forward(host)

    def run(self, steps: int) -> None:
        """Advance the network by the requested number of steps."""
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        for _ in range(steps):
            self.step()


class NeuronGroup:
    """A shaped population of neurons belonging to one network."""

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
            raise ValueError("shape must be a positive int or a tuple of three positive ints")
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

    def vector(self, fill: float = 0.0, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Create a constant vector with one entry per neuron."""
        return torch.full((self.size,), fill, dtype=dtype or self.net.dtype, device=self.net.device)

    def rand(self) -> torch.Tensor:
        """Draw a uniform random vector using the network generator."""
        return torch.rand(
            (self.size,),
            generator=self.net.generator,
            dtype=self.net.dtype,
            device=self.net.device,
        )

    def randn(self) -> torch.Tensor:
        """Draw a standard normal vector using the network generator."""
        return torch.randn(
            (self.size,),
            generator=self.net.generator,
            dtype=self.net.dtype,
            device=self.net.device,
        )

    def __repr__(self) -> str:
        return f"NeuronGroup({self.name!r}, shape={self.shape!r})"


class SynapseGroup:
    """Connections between neuron groups targeting one compartment."""

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
            raise ValueError("source and destination must belong to the network")
        try:
            self.compartment = Compartment(compartment)
        except ValueError:
            raise ValueError(f"unknown compartment: {compartment!r}") from None
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
