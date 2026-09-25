"""Fixed-depth per-neuron ring buffers that implement transmission delays.

Both buffers move a head index instead of copying rows, so a step costs O(size), not
O(depth * size). Delay tensors are validated once and remembered by identity and version,
which avoids a host-device synchronization on every read or write.
"""

from __future__ import annotations

import torch
from torch import Tensor


class _Buffer:
    """Ring storage of shape ``(depth, *batch, size)``: one row per slot.

    Values have shape ``(size,)``, or ``(batch, size)`` when ``batch`` is given; delays are
    per neuron and shared by every sample.
    """

    def __init__(
        self,
        depth: int,
        size: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        batch: int | None = None,
    ) -> None:
        if depth < 1 or size < 1:
            raise ValueError(f"depth and size must be positive, got depth={depth}, size={size}")
        if batch is not None and batch < 1:
            raise ValueError(f"batch must be positive or None, got {batch}")
        self._value_shape = (size,) if batch is None else (batch, size)
        self._storage = torch.zeros((depth, *self._value_shape), dtype=dtype, device=device)
        self._head = 0
        # id(delay tensor) -> (its version when validated, whether all delays are zero)
        self._checked: dict[int, tuple[int, bool]] = {}

    @property
    def depth(self) -> int:
        """Number of slots."""
        return self._storage.shape[0]

    @property
    def size(self) -> int:
        """Number of neurons."""
        return self._storage.shape[-1]

    def reset(self) -> None:
        """Clear every slot."""
        self._storage.zero_()
        self._head = 0

    def state_dict(self) -> dict[str, Tensor | int]:
        """A copy of the slots and the head position."""
        return {"storage": self._storage.clone(), "head": self._head}

    def load_state_dict(self, state: dict[str, Tensor | int]) -> None:
        """Restore slots and head saved by :meth:`state_dict`."""
        storage = state["storage"]
        if not isinstance(storage, Tensor) or storage.shape != self._storage.shape:
            shape = getattr(storage, "shape", None)
            raise ValueError(f"storage shape must be {tuple(self._storage.shape)}, got {shape}")
        self._storage.copy_(storage)
        self._head = int(state["head"])

    def _slots(self, delay: int | Tensor) -> Tensor:
        """Storage rows for per-neuron delays, shape ``(1, size)``; validates new delays."""
        if isinstance(delay, int):
            if not 0 <= delay < self.depth:
                raise ValueError(f"delay must be in [0, {self.depth}), got {delay}")
            slot = torch.tensor((self._head + delay) % self.depth, device=self._storage.device)
            return slot.expand(1, *self._value_shape)
        self._no_delay(delay)  # validates
        return ((delay + self._head) % self.depth).expand(self._value_shape).unsqueeze(0)

    def _no_delay(self, delay: int | Tensor) -> bool:
        """Whether every delay is zero; validates new delay tensors (once per version)."""
        if isinstance(delay, int):
            return delay == 0
        cached = self._checked.get(id(delay))
        if cached is None or cached[0] != delay._version:
            if delay.shape != (self.size,):
                raise ValueError(f"delay shape must be ({self.size},), got {tuple(delay.shape)}")
            if bool(((delay < 0) | (delay >= self.depth)).any()):
                raise ValueError(f"delay must be in [0, {self.depth}), got {delay.tolist()}")
            cached = (delay._version, not bool(delay.any()))
            self._checked[id(delay)] = cached
        return cached[1]

    def _check_value(self, value: Tensor) -> None:
        if value.shape != self._value_shape:
            raise ValueError(f"value shape must be {self._value_shape}, got {tuple(value.shape)}")


class HistoryBuffer(_Buffer):
    """The last ``depth`` values of a per-neuron vector; delay 0 is the newest."""

    def __init__(
        self,
        depth: int,
        size: int,
        *,
        dtype: torch.dtype = torch.bool,
        device: torch.device | str | None = None,
        batch: int | None = None,
    ) -> None:
        super().__init__(depth, size, dtype=dtype, device=device, batch=batch)

    def push(self, value: Tensor) -> None:
        """Store a copy of ``value`` as the newest entry, overwriting the oldest."""
        self._check_value(value)
        self._head = (self._head - 1) % self.depth
        self._storage[self._head].copy_(value)

    def read(self, delay: int | Tensor) -> Tensor:
        """Value pushed ``delay[i]`` pushes ago, for each neuron ``i`` (0 is the newest)."""
        if self._no_delay(delay):  # the common case: one row, no index arithmetic
            return self._storage[self._head].clone()
        return torch.gather(self._storage, 0, self._slots(delay)).squeeze(0)


class ArrivalBuffer(_Buffer):
    """Values scheduled to arrive after a per-neuron delay; delay 0 is due now."""

    def __init__(
        self,
        depth: int,
        size: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        batch: int | None = None,
    ) -> None:
        super().__init__(depth, size, dtype=dtype, device=device, batch=batch)

    def add(self, value: Tensor, delay: int | Tensor) -> None:
        """Add ``value[i]`` to the slot ``delay[i]`` steps ahead, for each neuron ``i``."""
        self._check_value(value)
        if self._no_delay(delay):
            self._storage[self._head] += value.to(self._storage.dtype)
            return
        source = value.to(self._storage.dtype).unsqueeze(0)
        self._storage.scatter_add_(0, self._slots(delay), source)

    def current(self) -> Tensor:
        """A copy of the total that is due now."""
        return self._storage[self._head].clone()

    def advance(self) -> None:
        """Move one step forward: clear the due slot and make it the last one."""
        self._storage[self._head].zero_()
        self._head = (self._head + 1) % self.depth
