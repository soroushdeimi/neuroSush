"""Versioned network specifications and deterministic construction."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.inputs import SpikeInput
from neurosush.recording import Recorder
from neurosush.structure.literals import AssetResolver, decode_literals, pointer
from neurosush.structure.spec import _REGISTRY, BehaviorSpec, ColumnSpec, from_json
from neurosush.viz.trace import TraceRecorder

_LOCAL_BEHAVIORS: dict[str, type[Behavior]] = {
    cls.__name__: cls for cls in (SpikeInput, Recorder, TraceRecorder)
}


@dataclass(frozen=True)
class NetSettings:
    """Simulation settings, expressed with JSON dtype names."""

    dt: float = 1.0
    dtype: str = "float32"
    device: str = "cpu"
    seed: int | None = None
    batch_size: int | None = None
    independent: bool = False
    behaviors: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.dtype not in {"float32", "float64", "float16", "bfloat16"}:
            raise ValueError(f"/network/dtype: invalid dtype {self.dtype!r}")


@dataclass(frozen=True)
class RecordingSpec:
    """Bounded trace recording for one named host."""

    target: str
    attributes: tuple[str, ...]
    interval: int = 1
    max_steps: int = 1000


@dataclass(frozen=True)
class NetworkSpec:
    """A complete version 2 network with embedded v1 column data."""

    network: NetSettings = field(default_factory=NetSettings)
    groups: dict[str, Any] = field(default_factory=dict)
    synapses: tuple[dict[str, Any], ...] = ()
    columns: dict[str, Any] = field(default_factory=dict)
    recordings: tuple[RecordingSpec, ...] = ()
    assets: dict[str, Any] = field(default_factory=dict)
    version: int = 2

    def __post_init__(self) -> None:
        if self.version != 2:
            raise ValueError(f"/version: unsupported version {self.version!r}")


@dataclass(frozen=True)
class Built:
    """Initialized simulation and its named hosts and trace recorders."""

    net: Network
    groups: dict[str, NeuronGroup]
    synapses: dict[str, SynapseGroup]
    recorders: dict[str, TraceRecorder]


def load_spec(text: str) -> ColumnSpec | NetworkSpec:
    """Load v1 column or v2 network JSON without decoding executable objects."""
    data = json.loads(text)
    version = data.get("version", 1)
    if version == 1:
        data.pop("version", None)
        return from_json(json.dumps(data))
    if version != 2:
        raise ValueError(f"/version: unsupported version {version!r}")
    data["network"] = NetSettings(**data.get("network", {}))
    data["recordings"] = tuple(
        RecordingSpec(**{**r, "attributes": tuple(r["attributes"])})
        for r in data.get("recordings", [])
    )
    data["synapses"] = tuple(data.get("synapses", []))
    return NetworkSpec(**data)


def dump_spec(spec: ColumnSpec | NetworkSpec) -> str:
    """Serialize canonically with sorted keys and a final newline."""
    return json.dumps(dataclasses.asdict(spec), indent=2, sort_keys=True, allow_nan=False) + "\n"


def build_network(spec: NetworkSpec, assets: AssetResolver | None = None) -> Built:
    """Build all hosts, attach bounded recorders, then initialize once."""
    settings = dataclasses.asdict(spec.network)
    network_behaviors = settings.pop("behaviors")
    settings["dtype"] = getattr(torch, settings["dtype"])
    net = Network(**settings)

    def behaviors(items: Any, path: str) -> list[Behavior]:
        result = []
        for i, item in enumerate(items):
            if dataclasses.is_dataclass(item) and not isinstance(item, type):
                item = dataclasses.asdict(item)
            here = pointer(path, i)
            params = decode_literals(item.get("params", {}), net, assets, pointer(here, "params"))
            kind = item["kind"]
            cls = _REGISTRY.get(kind, _LOCAL_BEHAVIORS.get(kind))
            if cls is None:
                raise ValueError(f"{here}/kind: unknown behavior {kind!r}; register it first")
            if kind in {"Recorder", "TraceRecorder"}:
                params = dict(params)
                attributes = params.pop("attributes", ())
                behavior = cls(*attributes, **params)
            elif kind in _REGISTRY:
                behavior = BehaviorSpec(kind, params).build()
            else:
                behavior = cls(**params)
            if not isinstance(item.get("enabled", True), bool):
                raise ValueError(f"{here}/enabled: expected a boolean")
            behavior.enabled = item.get("enabled", True)
            result.append(behavior)
        return result

    net.behaviors = net._attach(net, behaviors(network_behaviors, "/network/behaviors"))
    groups: dict[str, NeuronGroup] = {}
    synapses: dict[str, SynapseGroup] = {}

    def add_groups(items: dict[str, Any], prefix: str, path: str) -> None:
        for key, group in items.items():
            name = prefix + key
            here = pointer(path, key)
            shape = group["shape"]
            groups[name] = NeuronGroup(
                net,
                tuple(shape) if isinstance(shape, list) else shape,
                behaviors(group.get("behaviors", []), here + "/behaviors"),
                name=name,
                inhibitory=group.get("inhibitory", False),
                tags=group.get("tags", []),
            )

    def add_synapses(items: Any, prefix: str, path: str) -> None:
        for i, synapse in enumerate(items):
            here = pointer(path, i)
            src, dst = prefix + synapse["src"], prefix + synapse["dst"]
            name = synapse.get("name", f"{src}->{dst}")
            if name in groups or name == "network" or name in synapses:
                raise ValueError(f"{here}/name: duplicate or reserved name {name!r}")
            for endpoint in (src, dst):
                if endpoint not in groups:
                    raise ValueError(f"{here}: unknown group {endpoint!r}")
            synapses[name] = SynapseGroup(
                net,
                groups[src],
                groups[dst],
                behaviors(synapse.get("behaviors", []), here + "/behaviors"),
                name=name,
                compartment=synapse.get("compartment", "proximal"),
                tags=synapse.get("tags", []),
            )

    add_groups(spec.groups, "", "/groups")
    if "network" in groups:
        raise ValueError("/groups/network: reserved target name")
    for column_name, column in spec.columns.items():
        if isinstance(column, ColumnSpec):
            column = dataclasses.asdict(column)
        for layer_name, layer in column["layers"].items():
            prefix = f"{column_name}.{layer_name}."
            path = pointer(pointer("/columns", column_name) + "/layers", layer_name)
            add_groups(layer["groups"], prefix, path + "/groups")
            add_synapses(layer.get("synapses", []), prefix, path + "/synapses")
        add_synapses(column.get("synapses", []), column_name + ".", "/columns/" + column_name)
    add_synapses(spec.synapses, "", "/synapses")
    recorders = {}
    for i, recording in enumerate(spec.recordings):
        path = f"/recordings/{i}/target"
        target = recording.target
        if target in recorders:
            raise ValueError(f"{path}: duplicate recording target {target!r}")
        host: Network | NeuronGroup | SynapseGroup
        if target == "network":
            host = net
        elif target in groups:
            host = groups[target]
        elif target in synapses:
            host = synapses[target]
        else:
            raise ValueError(f"{path}: unknown recording target {target!r}")
        recorder = TraceRecorder(
            *recording.attributes, interval=recording.interval, max_steps=recording.max_steps
        )
        host.behaviors += net._attach(host, [recorder])
        recorders[target] = recorder
    net.initialize()
    return Built(net, groups, synapses, recorders)
