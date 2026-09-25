"""Capture a network's step as a CUDA graph and replay it.

On a GPU a step of a small network is mostly Python and kernel-launch overhead.
:class:`GraphStepper` captures the step's kernels once with ``torch.cuda.graph`` and replays
them. Behaviors assign new tensors every step (``group.v = ...``); inside the capture, every
replaced tensor attribute is copied back into the original tensor, so state keeps fixed
addresses. Python-side decisions (``Behavior.graph_key``, ``enabled``) select one graph per
key, captured when first needed. A few eager warm-up steps come first, on a side stream, and
behaviors from ``Order.RECORD`` on run eagerly after every step.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order

_Host = Network | NeuronGroup | SynapseGroup
_Pair = tuple[_Host, Behavior]
_Key = tuple[tuple[bool, Hashable], ...]
_Snapshot = tuple[object, str, torch.Tensor, str]


def _register_generators() -> bool:
    """Whether graphs must be told about custom generators (torch before 2.14).

    From torch 2.14 every generator is graph-safe on its own and the registration call is a
    deprecated no-op.
    """
    old = torch.torch_version.TorchVersion(torch.__version__) < "2.14"
    return old and hasattr(torch.cuda.CUDAGraph, "register_generator_state")


def random_numbers_supported() -> bool:
    """Whether draws from a CUDA ``torch.Generator`` can be captured and replayed."""
    return _register_generators() or torch.torch_version.TorchVersion(torch.__version__) >= "2.14"


def _host_name(host: _Host) -> str:
    """``host.name``, or ``"the network"`` when ``host`` is the network itself."""
    return "the network" if isinstance(host, Network) else host.name


def _tensor_attrs(obj: object) -> Iterator[tuple[str, torch.Tensor]]:
    """Direct tensor attributes of ``obj``: the ones a step might reassign."""
    for name, value in vars(obj).items():
        if isinstance(value, torch.Tensor):
            yield name, value


class GraphStepper:
    """Steps a CUDA network by capturing its step as a CUDA graph and replaying it.

    Args:
        net: The network to step; initialized first if needed.
        warmup: Number of steps run eagerly, on a side stream, before any graph is captured.

    Raises:
        ValueError: If ``net`` is not on a CUDA device, or a behavior that would be captured
            (``order < Order.RECORD``) is not
            :meth:`~neurosush.core.behavior.Behavior.graph_ready`.
    """

    def __init__(self, net: Network, *, warmup: int = 3) -> None:
        if not net.initialized:
            net.initialize()
        if net.device.type != "cuda":
            raise ValueError(f"GraphStepper needs a network on cuda, got device {net.device}")
        self.net = net
        self.warmup = warmup
        self.captured: list[_Pair] = [pair for pair in net.schedule if pair[1].order < Order.RECORD]
        self.after: list[_Pair] = [pair for pair in net.schedule if pair[1].order >= Order.RECORD]
        not_ready = [
            f"{type(behavior).__name__} on {_host_name(host)}"
            for host, behavior in self.captured
            if not behavior.graph_ready(host)
        ]
        if not_ready:
            raise ValueError(f"not graph-ready for a CUDA graph: {', '.join(not_ready)}")
        self._register_generator = _register_generators()
        self._graphs: dict[_Key, torch.cuda.CUDAGraph] = {}
        self._stream: torch.cuda.Stream = torch.cuda.Stream(  # type: ignore[no-untyped-call]
            device=net.device
        )  # torch.cuda.Stream.__new__ has no stub annotations
        self._steps = 0

    def step(self) -> None:
        """Advance one iteration: prepare, then an eager warm-up, capture or replay."""
        net = self.net
        net.iteration += 1
        for host, behavior in net._preparing:
            if behavior.enabled:
                behavior.prepare(host)
        if self._steps < self.warmup:
            self._eager_step()
        else:
            key: _Key = tuple(
                (behavior.enabled, behavior.graph_key(host)) for host, behavior in self.captured
            )
            graph = self._graphs.get(key)
            if graph is None:
                graph = self._capture()
                self._graphs[key] = graph
            graph.replay()
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

    def _eager_step(self) -> None:
        """Run the captured behaviors' forward on a side stream (no capture yet)."""
        current = torch.cuda.current_stream()
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            for host, behavior in self.captured:
                if behavior.enabled:
                    behavior.forward(host)
        current.wait_stream(self._stream)

    def _snapshot(self) -> list[_Snapshot]:
        """Every tensor attribute a captured step might reassign, before it runs."""
        net = self.net
        labeled: list[tuple[object, str]] = [(net, "the network")]
        labeled += [(group, group.name) for group in net.groups]
        labeled += [(syn, syn.name) for syn in net.synapses]
        labeled += [
            (behavior, f"{type(behavior).__name__} on {_host_name(host)}")
            for host, behavior in self.captured
        ]
        return [
            (obj, name, tensor, label)
            for obj, label in labeled
            for name, tensor in _tensor_attrs(obj)
        ]

    def _capture(self) -> torch.cuda.CUDAGraph:
        """Capture a new graph for the step ``net`` is currently set up to run.

        Raises:
            RuntimeError: If a captured behavior's ``forward`` raises while capturing.
            ValueError: If a reassigned attribute's shape or dtype changed.
        """
        snapshot = self._snapshot()
        graph = torch.cuda.CUDAGraph()
        if self._register_generator:
            graph.register_generator_state(self.net.generator)
        failure: Exception | None = None
        try:
            with torch.cuda.graph(graph):
                try:
                    self._record(snapshot)
                except Exception as error:
                    failure = error
                    raise
        except Exception:
            # ending a failed capture raises a generic CUDA error; report the cause instead
            if failure is None:
                raise
            raise failure from failure.__cause__
        finally:
            for obj, name, old, _label in snapshot:
                setattr(obj, name, old)
        return graph

    def _record(self, snapshot: list[_Snapshot]) -> None:
        """The captured step: every captured ``forward``, then the copy-back.

        Raises:
            RuntimeError: If a captured behavior's ``forward`` raises.
            ValueError: If a reassigned attribute's shape or dtype changed.
        """
        for host, behavior in self.captured:
            if behavior.enabled:
                try:
                    behavior.forward(host)
                except Exception as error:
                    raise RuntimeError(
                        f"{type(behavior).__name__} on {_host_name(host)} failed while "
                        "capturing a CUDA graph; PyTorch does not recover from a failed "
                        "capture, so restart the process after fixing it"
                    ) from error
        for obj, name, old, label in snapshot:
            new = getattr(obj, name)
            if new is old:
                continue
            if (
                not isinstance(new, torch.Tensor)
                or new.shape != old.shape
                or new.dtype != old.dtype
            ):
                got = (
                    f"{tuple(new.shape)} {new.dtype}"
                    if isinstance(new, torch.Tensor)
                    else type(new).__name__
                )
                raise ValueError(
                    f"{label}.{name} changed shape or dtype while capturing a CUDA graph: "
                    f"expected {tuple(old.shape)} {old.dtype}, got {got}"
                )
            old.copy_(new)
