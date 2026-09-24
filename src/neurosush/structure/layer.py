"""Layers: named neuron groups with named input and output ports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from neurosush.core.network import NeuronGroup, SynapseGroup

Ports = Mapping[str, Sequence[str]]


class Layer:
    """Named neuron groups, the synapses among them, and named ports.

    A port is a named tuple of groups that other structures connect to. By default the
    input port ``"in"`` and the output port ``"out"`` expose every group.

    Args:
        name: Layer name.
        groups: Groups by local name.
        synapses: Synapse groups inside the layer.
        inputs: Input ports as ``{port: [group names]}``.
        outputs: Output ports as ``{port: [group names]}``.
    """

    def __init__(
        self,
        name: str,
        groups: Mapping[str, NeuronGroup],
        *,
        synapses: Iterable[SynapseGroup] = (),
        inputs: Ports | None = None,
        outputs: Ports | None = None,
    ) -> None:
        if not groups:
            raise ValueError(f"layer {name!r} needs at least one group")
        self.name = name
        self.groups = dict(groups)
        self.synapses = list(synapses)
        everything = {"in": tuple(self.groups)}
        self._inputs = self._resolve(inputs if inputs is not None else everything)
        self._outputs = self._resolve(
            outputs if outputs is not None else {"out": tuple(self.groups)}
        )

    def _resolve(self, ports: Ports) -> dict[str, tuple[NeuronGroup, ...]]:
        resolved = {}
        for port, names in ports.items():
            missing = [n for n in names if n not in self.groups]
            if missing:
                raise ValueError(
                    f"port {port!r} of layer {self.name!r} names unknown groups {missing}"
                )
            resolved[port] = tuple(self.groups[n] for n in names)
        return resolved

    def __getitem__(self, key: str) -> NeuronGroup:
        return self.groups[key]

    def input_port(self, name: str = "in") -> tuple[NeuronGroup, ...]:
        """Groups of an input port."""
        return _port(self._inputs, name, self.name)

    def output_port(self, name: str = "out") -> tuple[NeuronGroup, ...]:
        """Groups of an output port."""
        return _port(self._outputs, name, self.name)


def _port(
    ports: dict[str, tuple[NeuronGroup, ...]], name: str, owner: str
) -> tuple[NeuronGroup, ...]:
    if name not in ports:
        raise KeyError(f"{owner!r} has no port {name!r}; ports: {sorted(ports)}")
    return ports[name]


class CorticalLayer(Layer):
    """A layer with an excitatory and an inhibitory population.

    Ports: ``"in"`` reaches both populations, ``"out"`` exposes the excitatory one.

    Args:
        name: Layer name.
        exc: Excitatory group (``inhibitory=False``).
        inh: Inhibitory group (``inhibitory=True``).
        synapses: Synapse groups inside the layer.
        inputs: Input ports, replacing the default.
        outputs: Output ports, replacing the default.
    """

    def __init__(
        self,
        name: str,
        *,
        exc: NeuronGroup,
        inh: NeuronGroup,
        synapses: Iterable[SynapseGroup] = (),
        inputs: Ports | None = None,
        outputs: Ports | None = None,
    ) -> None:
        if exc.inhibitory or not inh.inhibitory:
            raise ValueError(
                f"layer {name!r} needs an excitatory exc and an inhibitory inh group "
                f"(got exc.inhibitory={exc.inhibitory}, inh.inhibitory={inh.inhibitory})"
            )
        super().__init__(
            name,
            {"exc": exc, "inh": inh},
            synapses=synapses,
            inputs=inputs if inputs is not None else {"in": ("exc", "inh")},
            outputs=outputs if outputs is not None else {"out": ("exc",)},
        )

    @property
    def exc(self) -> NeuronGroup:
        """Excitatory population."""
        return self.groups["exc"]

    @property
    def inh(self) -> NeuronGroup:
        """Inhibitory population."""
        return self.groups["inh"]
