"""Trace data structures, tensor packing, and recording behaviors."""

from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass
from os import PathLike
from typing import Any, ClassVar

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order


def _name(host: Network | NeuronGroup | SynapseGroup) -> str:
    return "the network" if isinstance(host, Network) else host.name


def pack_array(t: torch.Tensor) -> dict[str, Any]:
    """Serialize a tensor into base64 little-endian bytes with shape metadata.

    Args:
        t: Tensor to pack.

    Returns:
        Dictionary with ``dtype``, ``shape``, and ``data`` (base64 string).
    """
    if t.dtype in (torch.float16, torch.float32, torch.float64):
        target_dtype = "float32"
        torch_dtype = torch.float32
        itemsize = 4
    elif t.dtype == torch.bool:
        target_dtype = "uint8"
        torch_dtype = torch.uint8
        itemsize = 1
    elif t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        if t.dtype == torch.int64 and t.numel() > 0:
            t_cpu = t.detach().to(device="cpu")
            min_val = t_cpu.min().item()
            max_val = t_cpu.max().item()
            if min_val < -2147483648 or max_val > 2147483647:
                raise ValueError(
                    f"int64 tensor values outside int32 range [-2147483648, 2147483647]: "
                    f"min {min_val}, max {max_val}"
                )
        target_dtype = "int32"
        torch_dtype = torch.int32
        itemsize = 4
    else:
        raise TypeError(f"unsupported dtype for pack_array: {t.dtype}")

    converted = t.detach().to(device="cpu", dtype=torch_dtype).contiguous()
    if converted.numel() == 0:
        raw_bytes = b""
    else:
        raw = converted.view(-1).view(torch.uint8)
        if sys.byteorder == "big" and itemsize > 1:
            raw = raw.view(-1, itemsize).flip(1).reshape(-1)
        raw_bytes = bytes(raw.tolist())
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return {"dtype": target_dtype, "shape": list(t.shape), "data": encoded}


def unpack_array(d: dict[str, Any]) -> torch.Tensor:
    """Deserialize a dictionary produced by :func:`pack_array` into a tensor.

    Args:
        d: Dictionary with ``dtype``, ``shape``, and ``data``.

    Returns:
        Restored tensor on the CPU.
    """
    dtype_str = d["dtype"]
    shape = tuple(d["shape"])
    raw_bytes = base64.b64decode(d["data"])

    if dtype_str == "float32":
        torch_dtype = torch.float32
        itemsize = 4
        is_bool = False
    elif dtype_str == "uint8":
        torch_dtype = torch.uint8
        itemsize = 1
        is_bool = True
    elif dtype_str == "int32":
        torch_dtype = torch.int32
        itemsize = 4
        is_bool = False
    else:
        raise TypeError(f"unsupported packed dtype: {dtype_str}")

    numel = 1
    for s in shape:
        numel *= s

    if numel == 0:
        return torch.empty(shape, dtype=torch.bool if is_bool else torch_dtype)

    if sys.byteorder == "big" and itemsize > 1:
        raw = (
            torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
            .view(-1, itemsize)
            .flip(1)
            .reshape(-1)
        )
        tensor = raw.view(torch_dtype).reshape(shape).clone()
    else:
        tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch_dtype).reshape(shape).clone()

    if is_bool:
        tensor = tensor.to(torch.bool)
    return tensor


@dataclass
class Trace:
    """Recorded simulation time series with metadata.

    Args:
        meta: JSON-serializable experiment metadata.
        steps: List of simulation step indices.
        series: Named tensors with time on dimension 0 on the CPU.
    """

    meta: dict[str, Any]
    steps: list[int]
    series: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        n_steps = len(self.steps)
        for name, tensor in self.series.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"series {name!r} must be a torch.Tensor, got {type(tensor).__name__}"
                )
            if tensor.device.type != "cpu":
                raise ValueError(f"series {name!r} must be on CPU, got device {tensor.device}")
            if tensor.dim() == 0:
                raise ValueError(
                    f"series {name!r} must have at least 1 dimension (time on dim 0), "
                    f"got shape {tuple(tensor.shape)}"
                )
            if tensor.shape[0] != n_steps:
                raise ValueError(
                    f"series {name!r} length {tensor.shape[0]} does not match "
                    f"steps length {n_steps}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert the trace into a JSON-serializable dictionary."""
        return {
            "meta": self.meta,
            "steps": list(self.steps),
            "series": {name: pack_array(t) for name, t in self.series.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        """Create a Trace from a dictionary produced by :meth:`to_dict`."""
        meta = dict(d.get("meta", {}))
        steps = list(d.get("steps", []))
        series = {name: unpack_array(arr) for name, arr in d.get("series", {}).items()}
        return cls(meta=meta, steps=steps, series=series)

    def save(self, path: str | PathLike[str]) -> None:
        """Save the trace to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> Trace:
        """Load a trace from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def window(self, start: int, stop: int) -> Trace:
        """Return a new Trace covering steps in [start, stop).

        Args:
            start: Inclusive lower bound on step values.
            stop: Exclusive upper bound on step values.
        """
        indices = [i for i, s in enumerate(self.steps) if start <= s < stop]
        new_steps = [self.steps[i] for i in indices]
        new_series = {name: t[indices].clone() for name, t in self.series.items()}
        return Trace(meta=dict(self.meta), steps=new_steps, series=new_series)

    def diff(self, other: Trace, name: str) -> float:
        """Compute the maximum absolute difference of one series against another trace.

        Args:
            other: Other trace to compare against.
            name: Series name to compare.
        """
        if name not in self.series:
            raise KeyError(f"series {name!r} not in this trace")
        if name not in other.series:
            raise KeyError(f"series {name!r} not in other trace")
        t1, t2 = self.series[name], other.series[name]
        if t1.shape != t2.shape:
            raise ValueError(
                f"series {name!r} shapes differ: {tuple(t1.shape)} vs {tuple(t2.shape)}"
            )
        if t1.numel() == 0:
            return 0.0
        return float((t1.to(torch.float64) - t2.to(torch.float64)).abs().max().item())

    def events(self, name: str, *, threshold: float = 0.5) -> list[int]:
        """Return steps at which a series has any nonzero or supra-threshold entry.

        Args:
            name: Series name to check for events.
            threshold: Value threshold for non-boolean series.
        """
        if name not in self.series:
            raise KeyError(f"series {name!r} not in this trace")
        if not self.steps:
            return []
        t = self.series[name]
        mask = t if t.dtype == torch.bool else (t >= threshold)
        has_event = mask.reshape(len(self.steps), -1).any(dim=1)
        return [s for s, ev in zip(self.steps, has_event.tolist(), strict=True) if ev]

    @classmethod
    def make_meta(cls, net: Network, **extra: Any) -> dict[str, Any]:
        """Build standard trace metadata from a Network instance.

        Args:
            net: Simulation network providing dt, device, and versions.
            **extra: Additional metadata keys to include.
        """
        import neurosush

        return {
            "library": neurosush.__version__,
            "torch": torch.__version__,
            "dt": net.dt,
            "seed": getattr(net, "seed", None),
            "device": str(net.device),
            **extra,
        }


class TraceRecorder(Behavior):
    """Preallocates device-side buffers to record host attributes efficiently.

    Schedule inspection expands ``*attributes`` in ``reads`` to the selected names.

    Args:
        *attributes: Names of host attributes to record.
        interval: Recording step interval.
        max_steps: Maximum number of steps to record (preallocated capacity).
    """

    reads: ClassVar[tuple[str, ...]] = ("*attributes",)
    writes: ClassVar[tuple[str, ...]] = ()
    order = Order.RECORD

    def __init__(self, *attributes: str, interval: int = 1, max_steps: int) -> None:
        if not attributes:
            raise ValueError("give at least one attribute to record")
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError(f"interval must be positive, got {interval}")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        self.attributes = attributes
        self.interval = interval
        self.max_steps = max_steps
        self._buffers: dict[str, torch.Tensor] = {}
        self.reset()

    def reset(self) -> None:
        """Reset the recorded steps and pointer to zero."""
        self.steps: list[int] = []
        self._count: int = 0

    def initialize(self, host: Network | NeuronGroup | SynapseGroup) -> None:
        """Check attribute existence and preallocate buffers on host device."""
        missing = [name for name in self.attributes if not hasattr(host, name)]
        if missing:
            raise RuntimeError(f"TraceRecorder on {_name(host)}: no attribute(s) {missing}")

        self._buffers = {}
        for name in self.attributes:
            val = getattr(host, name)
            if isinstance(val, torch.Tensor):
                shape = (self.max_steps, *val.shape)
                self._buffers[name] = torch.empty(shape, dtype=val.dtype, device=val.device)
            else:
                device = getattr(host, "device", torch.device("cpu"))
                val_t = torch.as_tensor(val, device=device)
                shape = (self.max_steps, *val_t.shape)
                self._buffers[name] = torch.empty(shape, dtype=val_t.dtype, device=val_t.device)
        self.reset()

    def forward(self, host: Network | NeuronGroup | SynapseGroup) -> None:
        """Copy attributes into preallocated buffer slice on active intervals."""
        iteration = (host if isinstance(host, Network) else host.net).iteration
        if iteration % self.interval != 0:
            return
        if self._count >= self.max_steps:
            raise RuntimeError(
                f"TraceRecorder on {_name(host)} exceeded max_steps={self.max_steps} "
                f"at iteration {iteration}; increase max_steps to record more samples"
            )
        values: dict[str, torch.Tensor] = {}
        for name in self.attributes:
            buffer = self._buffers[name]
            val = getattr(host, name)
            value = (
                val if isinstance(val, torch.Tensor) else torch.as_tensor(val, device=buffer.device)
            )
            if value.shape != buffer.shape[1:] or value.dtype != buffer.dtype:
                raise RuntimeError(
                    f"TraceRecorder on {_name(host)} attribute {name!r}: "
                    f"expected shape {tuple(buffer.shape[1:])} dtype {buffer.dtype}, "
                    f"got shape {tuple(value.shape)} dtype {value.dtype}"
                )
            values[name] = value
        for name, value in values.items():
            self._buffers[name][self._count].copy_(value)
        self.steps.append(iteration)
        self._count += 1

    def trace(self, **meta: Any) -> Trace:
        """Move recorded slices to CPU in one transfer per attribute and return a Trace.

        Args:
            **meta: Metadata to include in the returned Trace.
        """
        series = {
            name: self._buffers[name][: self._count].detach().to("cpu", copy=True)
            for name in self.attributes
        }
        return Trace(meta=meta, steps=list(self.steps), series=series)

    def __repr__(self) -> str:
        names = ", ".join(repr(name) for name in self.attributes)
        return f"TraceRecorder({names}, interval={self.interval}, max_steps={self.max_steps})"
