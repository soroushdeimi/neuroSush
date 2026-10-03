"""Compile a network's step with ``torch.compile``, optionally replayed as a CUDA graph.

:class:`GraphStepper` removes the launch overhead of a step but still runs one kernel per
tensor operation. :class:`CompiledStepper` traces every captured behavior's ``forward`` into one
function, lets TorchInductor fuse the element-wise work into a few kernels, and (on CUDA)
captures that compiled function as a CUDA graph per ``graph_key``, like :class:`GraphStepper`.

Differences from the eager and graph paths, which stay the bit-exact references:

* Fused kernels round differently, so results match eager stepping to a tolerance
  (``~1e-5`` in float32 on membrane voltages), not bit for bit; a spike can differ when a
  voltage lands within that tolerance of a threshold.
* Random numbers are drawn eagerly, in schedule order, before the compiled call
  (:meth:`~neurosush.core.behavior.Behavior.draw`), so the random stream equals eager stepping.
* Tensor attributes a behavior reassigns are written back into the original tensors inside the
  compiled function (``old.copy_(new)``), so state keeps fixed addresses.
* The first step of each ``graph_key`` compiles (seconds to a minute); it runs uncaptured.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.torch_version import TorchVersion

from neurosush.core.behavior import Behavior
from neurosush.core.graph import (
    _host_name,
    _Key,
    _Pair,
    step_key,
    tensor_snapshot,
)
from neurosush.core.network import Network
from neurosush.core.order import Order

MIN_TORCH_VERSION = "2.3"
"""The oldest torch whose dynamo traces the compiled step (2.2 and older cannot)."""


def torch_supports_compile() -> bool:
    """Whether the installed torch is at least :data:`MIN_TORCH_VERSION`."""
    return bool(TorchVersion(torch.__version__) >= MIN_TORCH_VERSION)


class CompiledStepper:
    """Steps a network through one ``torch.compile``-d function.

    Args:
        net: The network to step; initialized first if needed. On CPU or CUDA.
        warmup: Number of initial steps run through the compiled function without a CUDA graph.
        cuda_graph: Capture the compiled step as a CUDA graph (CUDA networks only).
        mode: ``mode`` argument of ``torch.compile`` (``None`` for the default).

    The tensor attributes written back are those present at construction; a tensor a behavior
    first assigns later (a lazily created attribute) is not written back, and the stepper raises
    ``RuntimeError`` during the first, uncaptured step of a key if it sees one.

    Needs torch 2.3 or newer (:data:`MIN_TORCH_VERSION`): dynamo in 2.2 and older cannot trace
    the step.

    Raises:
        RuntimeError: If the installed torch is older than 2.3.
        ValueError: If a behavior that would be compiled (``order < Order.RECORD``) is not
            :meth:`~neurosush.core.behavior.Behavior.compile_ready`.
    """

    def __init__(
        self,
        net: Network,
        *,
        warmup: int = 3,
        cuda_graph: bool = True,
        mode: str | None = None,
    ) -> None:
        if not torch_supports_compile():
            raise RuntimeError(
                f"CompiledStepper needs torch >= {MIN_TORCH_VERSION} (found {torch.__version__}): "
                "older dynamo cannot trace the compiled step; use GraphStepper or Network.run"
            )
        if not net.initialized:
            net.initialize()
        self.net = net
        self.warmup = warmup
        self.captured: list[_Pair] = [pair for pair in net.schedule if pair[1].order < Order.RECORD]
        self.after: list[_Pair] = [pair for pair in net.schedule if pair[1].order >= Order.RECORD]
        not_ready = [
            f"{type(behavior).__name__} on {_host_name(host)}"
            for host, behavior in self.captured
            if not behavior.compile_ready(host)
        ]
        if not_ready:
            raise ValueError(f"not ready to compile: {', '.join(not_ready)}")
        self._drawing = [pair for pair in self.captured if type(pair[1]).draw is not Behavior.draw]
        self._attrs = [(obj, name, label) for obj, name, _t, label in self._snapshot()]
        for limit in ("recompile_limit", "cache_size_limit"):  # networks of other shapes recompile
            config = torch._dynamo.config
            if hasattr(config, limit) and getattr(config, limit) < 64:
                setattr(config, limit, 64)
        self._compiled: Any = torch.compile(self._whole, fullgraph=True, mode=mode)
        self._use_graphs = cuda_graph and net.device.type == "cuda"
        self._graphs: dict[_Key, torch.cuda.CUDAGraph] = {}
        self._warm: set[_Key] = set()
        self._stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=net.device)  # type: ignore[no-untyped-call]
            if self._use_graphs
            else None
        )
        self._steps = 0

    def step(self) -> None:
        """Advance one iteration: prepare, then a compiled call, a capture or a replay."""
        net = self.net
        net.iteration += 1
        for host, behavior in net._preparing:
            if behavior.enabled:
                behavior.prepare(host)
        key = step_key(self.captured)
        graph = self._graphs.get(key)
        if graph is not None:
            graph.replay()
        elif self._use_graphs and self._steps >= self.warmup and key in self._warm:
            graph = self._capture()
            self._graphs[key] = graph
            graph.replay()
        else:
            first = key not in self._warm
            self._warm.add(key)  # compiles this variant: the next step of this key is captured
            self._call_on_side_stream()
            if first:
                self._check_new_attributes()
        self._steps += 1
        for host, behavior in self.after:
            if behavior.enabled:
                behavior.forward(host)

    def run(self, steps: int) -> None:
        """Call :meth:`step` ``steps`` times.

        Args:
            steps: Number of steps to advance; must be non-negative.
        """
        if steps < 0:
            raise ValueError(f"steps must be non-negative, got {steps}")
        for _ in range(steps):
            self.step()

    def _snapshot(self) -> list[tuple[object, str, torch.Tensor, str]]:
        return tensor_snapshot(self.net, self.captured)

    def _check_new_attributes(self) -> None:
        """Raise if a captured behavior's forward created a tensor attribute not in the snapshot."""
        known = {(id(obj), name) for obj, name, _label in self._attrs}
        for obj, name, _t, label in self._snapshot():
            if (id(obj), name) not in known:
                raise RuntimeError(
                    f"{label}.{name} was created after the CompiledStepper was built, so it "
                    "would not be written back; create state in initialize()"
                )

    def _call(self) -> None:
        """Draw random numbers eagerly, then run the compiled step."""
        drawing = [(b, b.draw(h)) for h, b in self._drawing if b.enabled]
        try:
            for behavior, drawn in drawing:
                behavior.drawn = drawn
            self._compiled()
        finally:
            for behavior, _drawn in drawing:
                del behavior.drawn  # back to the class default

    def _call_on_side_stream(self) -> None:
        if self._stream is None:
            self._call()
            return
        current = torch.cuda.current_stream()
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            self._call()
        current.wait_stream(self._stream)

    def _capture(self) -> torch.cuda.CUDAGraph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._call()
        return graph

    def _whole(self) -> None:
        """The compiled step: every captured ``forward``, then the in-place write-back."""
        olds = [(obj, name, getattr(obj, name), label) for obj, name, label in self._attrs]
        for host, behavior in self.captured:
            if behavior.enabled:
                behavior.forward(host)
        for obj, name, old, label in olds:
            new = getattr(obj, name)
            if new is old:
                continue
            if (
                not isinstance(new, torch.Tensor)
                or new.shape != old.shape
                or new.dtype != old.dtype
            ):
                raise ValueError(f"{label}.{name} changed shape or dtype inside the compiled step")
            old.copy_(new)
            setattr(obj, name, old)
