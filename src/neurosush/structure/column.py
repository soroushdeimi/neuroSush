"""Cortical columns: layers, the synapses between them, and column-level ports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from neurosush.core.network import NeuronGroup, SynapseGroup
from neurosush.structure.layer import Layer


class CorticalColumn:
    """Layers by name plus column ports that point at layer ports.

    Args:
        name: Column name.
        layers: Layers by name.
        synapses: Synapse groups between layers.
        inputs: Column input ports as ``{port: "layer.port"}``.
        outputs: Column output ports as ``{port: "layer.port"}``.
    """

    def __init__(
        self,
        name: str,
        layers: Mapping[str, Layer],
        *,
        synapses: Iterable[SynapseGroup] = (),
        inputs: Mapping[str, str] | None = None,
        outputs: Mapping[str, str] | None = None,
    ) -> None:
        self.name = name
        self.layers = dict(layers)
        self.synapses = list(synapses)
        self._inputs = {k: self._split(v) for k, v in (inputs or {}).items()}
        self._outputs = {k: self._split(v) for k, v in (outputs or {}).items()}
        # Resolve once so that a wrong layer port fails here, not at first use.
        for layer, layer_port in self._inputs.values():
            self.layers[layer].input_port(layer_port)
        for layer, layer_port in self._outputs.values():
            self.layers[layer].output_port(layer_port)

    def _split(self, reference: str) -> tuple[str, str]:
        layer, sep, port = reference.partition(".")
        if not sep:
            raise ValueError(f"port reference {reference!r} must look like 'layer.port'")
        if layer not in self.layers:
            raise ValueError(f"column {self.name!r} has no layer {layer!r}")
        return layer, port

    def input_port(self, name: str) -> tuple[NeuronGroup, ...]:
        """Groups behind a column input port."""
        layer, port = self._inputs[name]
        return self.layers[layer].input_port(port)

    def output_port(self, name: str) -> tuple[NeuronGroup, ...]:
        """Groups behind a column output port."""
        layer, port = self._outputs[name]
        return self.layers[layer].output_port(port)
