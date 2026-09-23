"""Fixed-depth per-neuron buffers that implement transmission delays."""

from __future__ import annotations

import torch
from torch import Tensor


class _Buffer:
    """Storage of shape ``(depth, size)``: one row per slot, one column per neuron."""

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

    def _delays(self, delay: int | Tensor) -> Tensor:
        """Validated per-neuron delays as a long tensor of shape ``(1, size)``."""
        delay = torch.as_tensor(delay, dtype=torch.long, device=self._storage.device)
        if delay.shape not in ((), (self._size,)):
            raise ValueError(f"delay shape must be () or ({self._size},), got {tuple(delay.shape)}")
        if bool(((delay < 0) | (delay >= self._depth)).any()):
            raise ValueError(f"delay must be in [0, {self._depth}), got {delay.tolist()}")
        return delay.expand(self._size).unsqueeze(0)

    def _validate_shape(self, value: Tensor) -> None:
        if value.shape != (self._size,):
            raise ValueError(f"expected shape ({self._size},), got {value.shape}")


class HistoryBuffer(_Buffer):
    """The last ``depth`` values of a per-neuron vector; slot 0 holds the newest."""

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
        """Store a copy of ``value`` as the newest entry, dropping the oldest."""
        self._validate_shape(value)
        self._storage[1:] = self._storage[:-1].clone()
        self._storage[0] = value.clone()

    def read(self, delay: int | Tensor) -> Tensor:
        """Value pushed ``delay[i]`` pushes ago, for each neuron ``i`` (0 is the newest)."""
        return torch.gather(self._storage, 0, self._delays(delay)).squeeze(0)


class ArrivalBuffer(_Buffer):
    """Values scheduled to arrive after a per-neuron delay; slot 0 is due now."""

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
        """Add ``value[i]`` to the slot ``delay[i]`` steps ahead, for each neuron ``i``."""
        self._validate_shape(value)
        index = self._delays(delay)
        self._storage.scatter_add_(0, index, value.to(self._storage.dtype).unsqueeze(0))

    def current(self) -> Tensor:
        """Return a copy of the current slot."""
        return self._storage[0].clone()

    def advance(self) -> None:
        """Move one step forward: drop the due slot and open an empty last slot."""
        self._storage[:-1] = self._storage[1:].clone()
        self._storage[-1].zero_()
