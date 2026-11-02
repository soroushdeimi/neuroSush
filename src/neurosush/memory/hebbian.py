"""Hebbian attractor memory and associative updates (Tolman-Eichenbaum Machine memory)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def hebbian_update(
    memory: Tensor,
    pattern: Tensor,
    predicted: Tensor,
    *,
    decay: float = 0.9999,
    rate: float = 0.5,
    clip: float | None = 1.0,
) -> Tensor:
    """Compute Tolman-Eichenbaum Machine Hebbian memory update.

    Formula:
        M = clamp(
            decay * memory + rate * outer(pattern - predicted, pattern + predicted),
            -clip,
            clip,
        )
    Shapes:
        memory: (..., n, n)
        pattern / predicted: (..., n)

    Args:
        memory: Current memory tensor of shape (..., n, n).
        pattern: Observed pattern tensor of shape (..., n).
        predicted: Predicted pattern tensor of shape (..., n).
        decay: Decay factor applied to existing memory in [0, 1].
        rate: Hebbian learning rate >= 0.
        clip: Optional maximum absolute value for memory clamping (> 0).

    Returns:
        Updated memory tensor of shape (..., n, n).
    """
    if not (0.0 <= decay <= 1.0):
        raise ValueError(f"decay must be in [0, 1], got {decay}")
    if rate < 0.0:
        raise ValueError(f"rate must be >= 0, got {rate}")
    if clip is not None and clip <= 0.0:
        raise ValueError(f"clip must be > 0, got {clip}")
    if pattern.shape != predicted.shape:
        raise ValueError(
            f"pattern and predicted shapes must match, got {pattern.shape} and {predicted.shape}"
        )
    if (
        memory.ndim < 2
        or memory.shape[-2] != memory.shape[-1]
        or memory.shape[-1] != pattern.shape[-1]
    ):
        raise ValueError(
            "memory shape must end with (n, n) matching pattern dim "
            f"n={pattern.shape[-1]}, got memory shape {memory.shape}"
        )

    diff = pattern - predicted
    sum_p = pattern + predicted
    outer = diff.unsqueeze(-1) * sum_p.unsqueeze(-2)
    updated = decay * memory + rate * outer
    if clip is not None:
        updated = updated.clamp(min=-clip, max=clip)
    return updated


def attractor(
    cue: Tensor,
    memory: Tensor,
    iterations: int = 5,
    *,
    retention: float = 0.8,
    negative_slope: float = 0.01,
    cap: float = 1.0,
    mask: Tensor | None = None,
) -> Tensor:
    """Retrieve pattern from cue using iterative attractor dynamics.

    Formula: h <- leaky_relu(retention * h + memory @ h, negative_slope), capped at cap.
    Shapes:
        cue: (..., n)
        memory: (..., n, n)

    Args:
        cue: Initial cue tensor of shape (..., n).
        memory: Memory weight tensor of shape (..., n, n).
        iterations: Number of attractor iterations (>= 0).
        retention: Retention coefficient for current state h.
        negative_slope: Negative slope for leaky ReLU activation.
        cap: Maximum upper bound for clamping (capped at cap).
        mask: Optional boolean tensor of shape (..., n) keeping
            masked-out entries fixed at cue values.

    Returns:
        Retrieved pattern tensor of shape (..., n).
    """
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if cue.shape[-1] != memory.shape[-1]:
        raise ValueError(f"cue dim {cue.shape[-1]} must match memory dim {memory.shape[-1]}")
    if mask is not None and mask.shape != cue.shape:
        raise ValueError(f"mask shape {mask.shape} must match cue shape {cue.shape}")

    h = cue.to(memory.dtype)
    for _ in range(iterations):
        mh = (memory @ h.unsqueeze(-1)).squeeze(-1)
        updated = F.leaky_relu(retention * h + mh, negative_slope=negative_slope).clamp(max=cap)
        h = torch.where(mask, updated, cue.to(memory.dtype)) if mask is not None else updated
    return h


class HebbianMemory:
    """Stateful wrapper for Hebbian attractor memory supporting storage and retrieval."""

    def __init__(
        self,
        size: int,
        *,
        batch_shape: tuple[int, ...] = (),
        decay: float = 0.9999,
        rate: float = 0.5,
        clip: float | None = 1.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize HebbianMemory.

        Args:
            size: Dimension n of patterns.
            batch_shape: Optional leading batch dimensions.
            decay: Decay factor in [0, 1].
            rate: Learning rate >= 0.
            clip: Optional clip limit > 0.
            dtype: Tensor data type.
            device: Tensor device.
        """
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        self.memory = torch.zeros((*batch_shape, size, size), dtype=dtype, device=device)
        self.decay = decay
        self.rate = rate
        self.clip = clip

    @property
    def decay(self) -> float:
        """Memory decay factor."""
        return self._decay

    @decay.setter
    def decay(self, value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {value}")
        self._decay = value

    @property
    def rate(self) -> float:
        """Hebbian learning rate."""
        return self._rate

    @rate.setter
    def rate(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"rate must be >= 0, got {value}")
        self._rate = value

    @property
    def clip(self) -> float | None:
        """Memory clamp limit."""
        return self._clip

    @clip.setter
    def clip(self, value: float | None) -> None:
        if value is not None and value <= 0.0:
            raise ValueError(f"clip must be > 0, got {value}")
        self._clip = value

    def store(self, pattern: Tensor, predicted: Tensor | None = None) -> None:
        """Store a pattern into memory using Hebbian update in place.

        Args:
            pattern: Observed pattern tensor of shape (..., n).
            predicted: Optional predicted pattern tensor of shape (..., n).
        """
        if predicted is None:
            predicted = torch.zeros_like(pattern)
        updated = hebbian_update(
            self.memory,
            pattern,
            predicted,
            decay=self._decay,
            rate=self._rate,
            clip=self._clip,
        )
        self.memory.copy_(updated)

    def retrieve(self, cue: Tensor, iterations: int = 5, **kwargs: Any) -> Tensor:
        """Retrieve pattern from cue using attractor dynamics.

        Args:
            cue: Initial cue tensor of shape (..., n).
            iterations: Number of attractor iterations.
            **kwargs: Additional keyword arguments passed to attractor.

        Returns:
            Retrieved pattern tensor of shape (..., n).
        """
        return attractor(cue, self.memory, iterations=iterations, **kwargs)

    def reset(self) -> None:
        """Reset memory to zeros in place."""
        self.memory.zero_()
