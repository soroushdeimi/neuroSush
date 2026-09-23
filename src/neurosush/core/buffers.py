"""Buffer implementations for neuroSush."""

from __future__ import annotations

import torch
from torch import Tensor


class _Buffer:
    """Private base class for buffers."""

    def __init__(
        self,
        depth: int,
        size: int,
        *,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> None:
        if depth < 1 or size < 1:
            raise ValueError("depth and size must be positive")
        self._depth = depth
        self._size = size
        self._storage = torch.zeros((depth, size), dtype=dtype, device=device)

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def size(self) -> int:
        return self._size

    def reset(self) -> None:
        self._storage.zero_()

    def _validate_delay(self, delay: int | Tensor) -> None:
        if isinstance(delay, int):
            if delay < 0 or delay >= self._depth:
                raise ValueError(f"delay {delay} is out of range")
        else:
            if (
                delay.ndim != 1
                or delay.shape[0] != self._size
                or torch.any(delay < 0)
                or torch.any(delay >= self._depth)
            ):
                raise ValueError("delay is out of range or has wrong shape")

    def _validate_shape(self, value: Tensor) -> None:
        if value.shape != (self._size,):
            raise ValueError(f"expected shape ({self._size},), got {value.shape}")


class HistoryBuffer(_Buffer):
    """Buffer that stores the last `depth` pushed values."""

    def __init__(
        self,
        depth: int,
        size: int,
        *,
        dtype: torch.dtype = torch.bool,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(depth, size, dtype=dtype, device=device)

    def push(self, value: Tensor) -> None:
        """Push a new value into the buffer."""
        self._validate_shape(value)
        self._storage[1:] = self._storage[:-1].clone()
        self._storage[0] = value.clone()

    def read(self, delay: int | Tensor) -> Tensor:
        """Read values with a given delay."""
        self._validate_delay(delay)
        if isinstance(delay, int):
            return self._storage[delay].clone()

        idx = delay.unsqueeze(0)
        return torch.gather(self._storage, 0, idx).squeeze(0).clone()


class ArrivalBuffer(_Buffer):
    """Buffer for values scheduled to arrive later."""

    def __init__(
        self,
        depth: int,
        size: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(depth, size, dtype=dtype, device=device)

    def add(self, value: Tensor, delay: int | Tensor) -> None:
        """Add values with a given delay."""
        self._validate_shape(value)
        self._validate_delay(delay)

        if isinstance(delay, int):
            self._storage[delay].add_(value)
            return

        idx = delay.unsqueeze(0)
        self._storage.scatter_add_(0, idx, value.unsqueeze(0))

    def current(self) -> Tensor:
        """Return a copy of the current slot."""
        return self._storage[0].clone()

    def advance(self) -> None:
        """Advance the buffer by one step."""
        self._storage[:-1] = self._storage[1:].clone()
        self._storage[-1].zero_()
