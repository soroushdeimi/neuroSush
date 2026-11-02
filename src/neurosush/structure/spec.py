"""Declarative specs of cortical columns, their builder, and JSON serialization.

A spec is plain data: behavior class names from a registry plus JSON-compatible
parameters. Building a spec twice gives two independent structures, which replaces
replication; loading JSON only looks names up in the registry and never runs code.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
from dataclasses import dataclass, field
from typing import Any, TypedDict

from neurosush.core.behavior import Behavior
from neurosush.core.network import Compartment, Network, NeuronGroup, SynapseGroup
from neurosush.modulation import Dopamine
from neurosush.neurons.adaptation import SpikeTriggeredCurrent
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import KWTA, InherentNoise, MinicolumnInhibition
from neurosush.neurons.dendrite import (
    ConductanceIntegration,
    DendriteIntegration,
    DendriteStructure,
)
from neurosush.neurons.hodgkin_huxley import HodgkinHuxley
from neurosush.neurons.homeostasis import ActivityHomeostasis, AdaptiveThreshold, VoltageHomeostasis
from neurosush.neurons.inputs import CorrelatedPoissonInput, PoissonDrive, PoissonInput
from neurosush.neurons.models import ELIF, LIF, AdaptiveELIF, Fire, Izhikevich, Refractory
from neurosush.recording import SpikeCounter
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
    MaxPool2dInput,
    OneToOneInput,
    SparseInput,
)
from neurosush.synapses.init import DelayInit, WeightInit
from neurosush.synapses.plasticity import ISTDP, RSTDP, STDP
from neurosush.synapses.segment_learning import SegmentLearning
from neurosush.synapses.segments import ActiveSegments
from neurosush.synapses.traces import SpikeGather, Traces
from neurosush.synapses.triplet import TripletSTDP

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
    Izhikevich,
    HodgkinHuxley,
    SpikeTriggeredCurrent,
    PoissonDrive,
    CorrelatedPoissonInput,
    Fire,
    KWTA,
    MinicolumnInhibition,
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
    MaxPool2dInput,
    ActiveSegments,
    SpikeGather,
    Traces,
    STDP,
    RSTDP,
    ISTDP,
    WeightClip,
    WeightNormalization,
    CurrentNormalization,
    Dopamine,
    ConductanceIntegration,
    AdaptiveThreshold,
    Refractory,
    SegmentLearning,
    TripletSTDP,
    SpikeCounter,
    PoissonInput,
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
    if len(value) != 3:
        raise ValueError(f"a group shape is an int or three ints, got {value!r}")
    return value[0], value[1], value[2]


def _synapse_specs(items: list[_SynapseData]) -> tuple[SynapseSpec, ...]:
    return tuple(
        SynapseSpec(s["src"], s["dst"], _behaviors(s["behaviors"]), s["compartment"]) for s in items
    )


def _ports(value: dict[str, list[str]] | None) -> dict[str, tuple[str, ...]] | None:
    return None if value is None else {k: tuple(v) for k, v in value.items()}


def _check_keys(data: dict[str, Any], cls: type[Any], path: str) -> None:
    allowed = [field.name for field in dataclasses.fields(cls)]
    for key in data:
        if key not in allowed:
            matches = difflib.get_close_matches(key, allowed, n=1)
            fix = f"did you mean {matches[0]}?" if matches else f"remove {key!r}"
            raise ValueError(f"{path}.{key}: unknown key; {fix}")


def _check_behaviors(items: list[dict[str, Any]], path: str) -> None:
    for index, item in enumerate(items):
        _check_keys(item, BehaviorSpec, f"{path}[{index}]")


def _check_synapses(items: list[dict[str, Any]], path: str) -> None:
    for index, item in enumerate(items):
        location = f"{path}[{index}]"
        _check_keys(item, SynapseSpec, location)
        _check_behaviors(item["behaviors"], f"{location}.behaviors")


def _check_column(data: dict[str, Any]) -> None:
    _check_keys(data, ColumnSpec, "$")
    _check_synapses(data["synapses"], "$.synapses")
    for name, layer in data["layers"].items():
        path = f"$.layers.{name}"
        _check_keys(layer, LayerSpec, path)
        _check_synapses(layer["synapses"], f"{path}.synapses")
        for name, group in layer["groups"].items():
            location = f"{path}.groups.{name}"
            _check_keys(group, GroupSpec, location)
            _check_behaviors(group["behaviors"], f"{location}.behaviors")


def from_json(text: str) -> ColumnSpec:
    """Parse a column spec written by :func:`to_json`."""
    data = json.loads(text)
    _check_column(data)
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
