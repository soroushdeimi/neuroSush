"""Validation shared by the ``load_state_dict`` methods of the HTM models."""

from __future__ import annotations

import torch


def checked(
    name: str, value: object, like: torch.Tensor, *, rows: int | None = None
) -> torch.Tensor:
    """A copy of ``value`` with the shape and dtype of ``like``.

    Args:
        name: Key reported in the error message.
        value: The loaded value.
        like: The tensor it replaces.
        rows: Expected leading dimension instead of ``like``'s (for growable storage).
    """
    shape = tuple(like.shape) if rows is None else (rows, *like.shape[1:])
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        got = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        raise ValueError(f"{name} must have shape {shape}, got {got}")
    return value.to(dtype=like.dtype).clone()
