"""Validation and placement of neuron parameters that may differ per neuron."""

from __future__ import annotations

import torch

from neurosush.core.network import NeuronGroup


def positive(**values: float | torch.Tensor) -> None:
    """Raise ValueError if any value (or any element of a tensor value) is <= 0."""
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            if torch.any(value <= 0):
                raise ValueError(f"{name} must be positive, got {value}")
        elif value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def at_least(minimum: float, **values: float | torch.Tensor) -> None:
    """Raise ValueError if any value (or any element of a tensor value) is below ``minimum``."""
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            if torch.any(value < minimum):
                raise ValueError(f"{name} ({value}) must be at least dt ({minimum})")
        elif value < minimum:
            raise ValueError(f"{name} ({value}) must be at least dt ({minimum})")


def keep(value: float | torch.Tensor) -> float | torch.Tensor:
    """A number as ``float``; a tensor unchanged."""
    return value if isinstance(value, torch.Tensor) else float(value)


def per_neuron(group: NeuronGroup, value: float | torch.Tensor, name: str) -> float | torch.Tensor:
    """Place a parameter on ``group``'s device and dtype.

    A number stays a ``float`` (scalar arithmetic is unchanged). A tensor must have shape
    ``(size,)`` or, in an independent network, ``(batch_size, size)``.
    """
    if not isinstance(value, torch.Tensor):
        return float(value)
    net = group.net
    allowed: list[tuple[int, ...]] = [(group.size,)]
    if net.independent:
        allowed.append(group.state_shape)
    if tuple(value.shape) not in allowed:
        raise ValueError(f"{name} must have shape {allowed[-1]}, got {tuple(value.shape)}")
    return value.to(dtype=net.dtype, device=net.device)


def state_like(group: NeuronGroup, value: float | torch.Tensor, name: str) -> torch.Tensor:
    """A state tensor of shape ``group.state_shape`` from a number or a tensor.

    A tensor may have shape ``(size,)`` or the state shape.
    """
    if not isinstance(value, torch.Tensor):
        return group.state(value)
    if tuple(value.shape) not in ((group.size,), group.state_shape):
        raise ValueError(
            f"{name} must have shape ({group.size},) or {group.state_shape}, "
            f"got {tuple(value.shape)}"
        )
    value = value.to(dtype=group.net.dtype, device=group.net.device)
    return value.expand(group.state_shape).clone()
