"""Spawn-safe library operations for the warm worker."""

from __future__ import annotations

from typing import Any

import torch

import neurosush
from neurosush.core.network import Network, NeuronGroup
from neurosush.neurons.models import Fire
from neurosush.structure.spec import BehaviorSpec
from studio.pool.validation import validate_network


def execute(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a validation or a single-neuron trace using library state."""
    if operation == "validate":
        return validate_network(payload)
    model = payload["model"]
    behavior = BehaviorSpec(model["kind"], model["params"]).build()
    if not callable(getattr(behavior, "fire", None)):
        raise ValueError("/model/kind: choose a neuron model with a fire method")
    behavior.enabled = model["enabled"]
    net = Network(dt=payload["dt"])
    group = NeuronGroup(net, 1, [behavior, Fire()])
    net.initialize()
    group.I.fill_(payload["current"])
    voltages = [group.v.item()]
    spikes = []
    with torch.inference_mode():
        for _ in range(payload["steps"]):
            net.step()
            voltages.append(group.v.item())
            if group.spikes.item():
                spikes.append(net.iteration)
    return {
        "t": [k * net.dt for k in range(len(voltages))],
        "v": voltages,
        "spikes": spikes,
        "provenance": {
            "library": neurosush.__version__,
            "kind": model["kind"],
            "params": model["params"],
            "dt": net.dt,
            "steps": payload["steps"],
            "current": payload["current"],
        },
    }
