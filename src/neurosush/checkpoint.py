"""Save and restore the complete state of an initialized network.

A checkpoint holds the iteration, the random generator, every tensor and delay buffer on
the network, its neuron groups and its synapse groups (weights, delays, voltages, traces
and whatever else behaviors keep there), and the state of behaviors that keep some
themselves. Loading it into a freshly built network of the same structure continues the
simulation exactly where it stopped. Input streams (``SpikeInput``) are not saved: resume
them yourself.
"""

from __future__ import annotations

from collections.abc import Iterator
from os import PathLike
from typing import Any

import torch

from neurosush.core.buffers import _Buffer
from neurosush.core.network import Network

_SCALARS = ("payoff", "dopamine")


def _hosts(net: Network) -> Iterator[tuple[str, Any]]:
    yield "net", net
    for group in net.groups:
        yield f"groups.{group.name}", group
    for syn in net.synapses:
        yield f"synapses.{syn.name}", syn


def _is_buffer_dict(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(v, _Buffer) for v in value.values())
    )


def _entries(net: Network) -> Iterator[tuple[str, Any, str]]:
    """``(key, host, attribute)`` of every piece of saved host state, in a fixed order."""
    for prefix, host in _hosts(net):
        for name, value in sorted(vars(host).items()):
            if host is net and name == "generator":
                continue
            saved = isinstance(value, (torch.Tensor, _Buffer)) or _is_buffer_dict(value)
            if saved or (host is net and name in _SCALARS):
                yield f"{prefix}.{name}", host, name


def _behaviors(net: Network) -> Iterator[tuple[str, Any]]:
    for prefix, host in _hosts(net):
        for index, behavior in enumerate(host.behaviors):
            yield f"{prefix}.{type(behavior).__name__}#{index}", behavior


def _key(key: Any) -> str:
    """Dictionary keys as plain strings (compartments by their value)."""
    return str(getattr(key, "value", key))


def _save(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, _Buffer):
        return value.state_dict()
    if isinstance(value, dict):
        return {_key(k): _save(v) for k, v in value.items()}
    return value


def state_dict(net: Network) -> dict[str, Any]:
    """Everything needed to resume ``net``; the network must be initialized."""
    if not net.initialized:
        raise RuntimeError("initialize the network before saving it")
    state: dict[str, Any] = {
        "iteration": net.iteration,
        "generator": net.generator.get_state(),
    }
    for key, host, name in _entries(net):
        state[key] = _save(getattr(host, name))
    for key, behavior in _behaviors(net):
        saved = behavior.state_dict()
        if saved:
            state[key] = {k: _save(v) for k, v in saved.items()}
    return state


def _restore(key: str, host: Any, name: str, value: Any) -> None:
    current = getattr(host, name)
    if isinstance(current, torch.Tensor):
        if not isinstance(value, torch.Tensor) or value.shape != current.shape:
            got = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ValueError(f"{key}: expected shape {tuple(current.shape)}, got {got}")
        setattr(host, name, value.to(device=current.device, dtype=current.dtype))
    elif isinstance(current, (_Buffer, dict)):
        buffers = {"": current} if isinstance(current, _Buffer) else current
        for compartment, buffer in buffers.items():
            try:
                buffer.load_state_dict(value[_key(compartment)] if compartment else value)
            except ValueError as error:
                raise ValueError(f"{key}: {error}") from None
    else:
        setattr(host, name, value)


def load_state_dict(net: Network, state: dict[str, Any], *, strict: bool = True) -> None:
    """Restore a :func:`state_dict` into a network built the same way.

    Args:
        net: The network to restore; it is initialized first if needed.
        state: A dictionary from :func:`state_dict`.
        strict: Require the keys of ``state`` and of the network to match exactly.
    """
    if not net.initialized:
        net.initialize()
    entries = list(_entries(net))
    behaviors = [(k, b) for k, b in _behaviors(net) if b.state_dict() or k in state]
    expected = {"iteration", "generator"} | {k for k, _, _ in entries} | {k for k, _ in behaviors}
    if strict:
        missing, unexpected = sorted(expected - state.keys()), sorted(state.keys() - expected)
        if missing or unexpected:
            raise KeyError(
                f"state does not match the network: missing {missing}, unexpected {unexpected}"
            )
    net.iteration = int(state.get("iteration", net.iteration))
    if "generator" in state:
        net.generator.set_state(state["generator"].cpu())
    for key, host, name in entries:
        if key in state:
            _restore(key, host, name, state[key])
    for key, behavior in behaviors:
        if key in state:
            behavior.load_state_dict(state[key])


def save(net: Network, path: str | PathLike[str]) -> None:
    """Write :func:`state_dict` of ``net`` to ``path``."""
    torch.save(state_dict(net), path)


def load(net: Network, path: str | PathLike[str], *, strict: bool = True) -> None:
    """Restore ``net`` from a file written by :func:`save`.

    Only tensors and plain containers are unpickled (``weights_only=True``).
    """
    state = torch.load(path, map_location=net.device, weights_only=True)
    load_state_dict(net, state, strict=strict)
