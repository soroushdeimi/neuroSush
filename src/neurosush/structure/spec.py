"""Declarative specs of cortical columns, their builder, and JSON serialization.

A spec is plain data: behavior class names from a registry plus JSON-compatible
parameters. Building a spec twice gives two independent structures, which replaces
replication; loading JSON only looks names up in the registry and never runs code.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, TypedDict

from neurosush.core.behavior import Behavior
from neurosush.core.network import Compartment, Network, NeuronGroup, SynapseGroup
from neurosush.modulation import Dopamine
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import KWTA, InherentNoise
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.homeostasis import ActivityHomeostasis, VoltageHomeostasis
from neurosush.neurons.models import ELIF, LIF, AdaptiveELIF, Fire
from neurosush.structure.column import CorticalColumn
from neurosush.structure.connect import connect
from neurosush.structure.layer import Layer
from neurosush.synapses.constraints import CurrentNormalization, WeightClip, WeightNormalization
from neurosush.synapses.currents import (
    AvgPool2dInput,
    Conv2dInput,
    DenseInput,
    LateralInput,
    Local2dInput,
    OneToOneInput,
    SparseInput,
)
from neurosush.synapses.init import DelayInit, WeightInit
from neurosush.synapses.plasticity import ISTDP, RSTDP, STDP
from neurosush.synapses.traces import SpikeGather, Traces

_REGISTRY: dict[str, type[Behavior]] = {}


def register(cls: type[Behavior]) -> type[Behavior]:
    """Make a behavior class available to specs under its class name (usable as a decorator)."""
    known = _REGISTRY.get(cls.__name__)
    if known is not None and known is not cls:
        raise ValueError(f"a different behavior is already registered as {cls.__name__!r}")
    _REGISTRY[cls.__name__] = cls
    return cls


def registered() -> list[str]:
    """Names of all registered behavior classes."""
    return sorted(_REGISTRY)


for _cls in (
    LIF,
    ELIF,
    AdaptiveELIF,
    Fire,
    KWTA,
    InherentNoise,
    Axon,
    DendriteStructure,
    DendriteIntegration,
    ActivityHomeostasis,
    VoltageHomeostasis,
    WeightInit,
    DelayInit,
    DenseInput,
    OneToOneInput,
    SparseInput,
    Conv2dInput,
    Local2dInput,
    LateralInput,
    AvgPool2dInput,
    SpikeGather,
    Traces,
    STDP,
    RSTDP,
    ISTDP,
    WeightClip,
    WeightNormalization,
    CurrentNormalization,
    Dopamine,
):
    register(_cls)


@dataclass(frozen=True)
class BehaviorSpec:
    """A registered behavior class name and its keyword arguments."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Behavior:
        """A new behavior instance."""
        cls = _REGISTRY.get(self.kind)
        if cls is None:
            raise ValueError(f"unknown behavior {self.kind!r}; register it first")
        try:
            return cls(**self.params)
        except TypeError as error:
            raise TypeError(f"{self.kind}: {error}") from error


@dataclass(frozen=True)
class GroupSpec:
    """A neuron group: shape, behaviors, sign and tags."""

    shape: int | tuple[int, int, int]
    behaviors: tuple[BehaviorSpec, ...] = ()
    inhibitory: bool = False
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SynapseSpec:
    """Synapses from ``src`` to ``dst`` (group names; ``"layer.group"`` inside a column)."""

    src: str
    dst: str
    behaviors: tuple[BehaviorSpec, ...]
    compartment: str = "proximal"


@dataclass(frozen=True)
class LayerSpec:
    """Groups, synapses among them, and ports (``None`` keeps the layer defaults)."""

    groups: dict[str, GroupSpec]
    synapses: tuple[SynapseSpec, ...] = ()
    inputs: dict[str, tuple[str, ...]] | None = None
    outputs: dict[str, tuple[str, ...]] | None = None


@dataclass(frozen=True)
class ColumnSpec:
    """Layers, synapses between them, and column ports (``{port: "layer.port"}``)."""

    layers: dict[str, LayerSpec]
    synapses: tuple[SynapseSpec, ...] = ()
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)


def _lookup(groups: dict[str, NeuronGroup], name: str, owner: str) -> NeuronGroup:
    if name not in groups:
        raise ValueError(f"{owner}: unknown group {name!r}")
    return groups[name]


def _synapses(
    net: Network,
    specs: tuple[SynapseSpec, ...],
    groups: dict[str, NeuronGroup],
    owner: str,
) -> list[SynapseGroup]:
    created = []
    for spec in specs:
        src, dst = _lookup(groups, spec.src, owner), _lookup(groups, spec.dst, owner)

        def factory(spec: SynapseSpec = spec) -> list[Behavior]:
            return [b.build() for b in spec.behaviors]

        created += connect(net, [src], [dst], factory, compartment=Compartment(spec.compartment))
    return created


def build_layer(net: Network, name: str, spec: LayerSpec) -> Layer:
    """Create the groups and synapses of a layer; groups are named ``{name}.{group}``."""
    groups = {
        key: NeuronGroup(
            net,
            g.shape,
            [b.build() for b in g.behaviors],
            name=f"{name}.{key}",
            tags=g.tags,
            inhibitory=g.inhibitory,
        )
        for key, g in spec.groups.items()
    }
    synapses = _synapses(net, spec.synapses, groups, f"layer {name!r}")
    return Layer(name, groups, synapses=synapses, inputs=spec.inputs, outputs=spec.outputs)


def build_column(net: Network, name: str, spec: ColumnSpec) -> CorticalColumn:
    """Create a column; groups are named ``{column}.{layer}.{group}``."""
    layers = {key: build_layer(net, f"{name}.{key}", layer) for key, layer in spec.layers.items()}
    groups = {f"{lk}.{gk}": g for lk, layer in layers.items() for gk, g in layer.groups.items()}
    synapses = _synapses(net, spec.synapses, groups, f"column {name!r}")
    return CorticalColumn(name, layers, synapses=synapses, inputs=spec.inputs, outputs=spec.outputs)


def to_json(spec: ColumnSpec) -> str:
    """Serialize a column spec; every parameter must be JSON-compatible."""
    try:
        return json.dumps(dataclasses.asdict(spec), indent=2)
    except TypeError as error:
        raise ValueError(f"spec parameters must be JSON-compatible: {error}") from None


class _BehaviorData(TypedDict):
    kind: str
    params: dict[str, Any]


class _SynapseData(TypedDict):
    src: str
    dst: str
    behaviors: list[_BehaviorData]
    compartment: str


def _behaviors(items: list[_BehaviorData]) -> tuple[BehaviorSpec, ...]:
    return tuple(BehaviorSpec(b["kind"], dict(b["params"])) for b in items)


def _shape(value: int | list[int]) -> int | tuple[int, int, int]:
    if isinstance(value, int):
        return value
    shape = tuple(value)
    assert len(shape) == 3  # Serialized group shapes have depth, height and width.
    return shape


def _synapse_specs(items: list[_SynapseData]) -> tuple[SynapseSpec, ...]:
    return tuple(
        SynapseSpec(s["src"], s["dst"], _behaviors(s["behaviors"]), s["compartment"]) for s in items
    )


def _ports(value: dict[str, list[str]] | None) -> dict[str, tuple[str, ...]] | None:
    return None if value is None else {k: tuple(v) for k, v in value.items()}


def from_json(text: str) -> ColumnSpec:
    """Parse a column spec written by :func:`to_json`."""
    data = json.loads(text)
    layers = {
        name: LayerSpec(
            groups={
                key: GroupSpec(
                    _shape(g["shape"]),
                    _behaviors(g["behaviors"]),
                    g["inhibitory"],
                    tuple(g["tags"]),
                )
                for key, g in layer["groups"].items()
            },
            synapses=_synapse_specs(layer["synapses"]),
            inputs=_ports(layer["inputs"]),
            outputs=_ports(layer["outputs"]),
        )
        for name, layer in data["layers"].items()
    }
    return ColumnSpec(
        layers=layers,
        synapses=_synapse_specs(data["synapses"]),
        inputs=dict(data["inputs"]),
        outputs=dict(data["outputs"]),
    )
