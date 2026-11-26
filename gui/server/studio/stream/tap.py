"""Copy selected simulation state after recording into display frames."""

from __future__ import annotations

import ctypes
import json
import math
import struct
from pathlib import Path
from typing import Any

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network
from neurosush.core.order import Order
from neurosush.structure.network_spec import Built
from neurosush.viz.schedule import schedule_table
from studio.stream.frames import encode_scalar, encode_spikes, encode_vsample, encode_weights
from studio.stream.ring import Ring


def first(value: torch.Tensor, net: Network) -> torch.Tensor:
    """Select only batch member zero."""
    return value[0] if net.batch_size is not None else value


class StreamTap(Behavior):
    """Buffer copies by step and encode them in batches for the shared ring."""

    order = Order.RECORD + 1
    independent_ok = True

    def __init__(self, built: Built, selection: dict[str, Any], ring: Ring) -> None:
        self.built, self.ring = built, ring
        self.groups = list(built.groups)
        self.synapses = list(built.synapses)
        spikes = selection.get("spikes", [])
        self.spikes = self.groups if spikes == "all" else list(spikes)
        self.vsample = dict(selection.get("vsample", {}))
        self.scalars = list(selection.get("scalars", []))
        self.weights = dict(selection.get("weights", {}))
        self.issues: list[dict[str, str]] = []
        self.metadata: Path | None = None
        self.end_step: int | None = None
        self.count = 0
        self.step0 = 0
        self.flags = 0
        self.buffers: dict[str, list[bytes]] = {}
        self.spike_groups = []
        self.recorded: dict[str, torch.Tensor] = {}
        for name in self.spikes:
            group = built.groups[name]
            if not hasattr(group, "spikes"):
                raise ValueError(f"/stream/spikes: {name} has no spikes")
            self.buffers[name] = []
            self.spike_groups.append((self.buffers[name], group, group.size))
        self.cpu_unbatched = built.net.device.type == "cpu" and built.net.batch_size is None
        if self.cpu_unbatched:
            for name in self.spikes:
                recorder = built.recorders.get(name)
                if (
                    recorder is not None
                    and recorder.enabled
                    and recorder.interval == 1
                    and "spikes" in recorder.attributes
                ):
                    # Reuse the recorder's step copies to avoid copying spikes twice.
                    self.recorded[name] = recorder._buffers["spikes"]

        for name, indices in self.vsample.items():
            self.validate_indices(name, indices)
        for name in self.scalars:
            if name.startswith("rate:"):
                if name[5:] not in built.groups:
                    raise ValueError(f"/stream/scalars: unknown group {name[5:]}")
            elif name not in {"dopamine", "loss", "accuracy", "anomaly"}:
                raise ValueError(f"/stream/scalars: unknown scalar {name}")
            elif not hasattr(built.net, name):
                raise ValueError(f"/stream/scalars: unavailable scalar {name}")
        for name, interval in list(self.weights.items()):
            if name not in built.synapses or type(interval) is not int or interval < 1:
                raise ValueError("/stream/weights: expected known synapse and positive interval")
            synapse = built.synapses[name]
            weights = synapse.weights
            if weights is not None and built.net.independent:
                weights = weights[0]
            shape = tuple(weights.shape) if weights is not None else ()
            layout = getattr(synapse, "connectivity", None)
            if layout is None:
                if hasattr(synapse, "src_idx"):
                    layout = "sparse"
                elif len(shape) == 1:
                    layout = "compact 1-D"
                elif shape == (synapse.src.size, synapse.dst.size):
                    layout = "dense"
                else:
                    layout = f"shape {shape}"
            if layout != "dense" or shape != (synapse.src.size, synapse.dst.size):
                self.issues.append(
                    {
                        "path": "/synapses/" + name.replace("~", "~0").replace("/", "~1"),
                        "message": f"weights not streamable for {layout}",
                        "fix": "stream is limited to dense synapses in this version",
                    }
                )
                del self.weights[name]

        self.voltages: dict[str, list[list[float]]] = {name: [] for name in self.vsample}
        self.values: dict[str, list[float]] = {name: [] for name in self.scalars}
        self.simple = (
            self.cpu_unbatched
            and len(self.spike_groups) == 1
            and not self.vsample
            and not self.scalars
            and not self.weights
        )

    def validate_indices(self, name: str, indices: Any) -> None:
        """Reject invalid voltage selections before indexing tensors."""
        if (
            name not in self.built.groups
            or not isinstance(indices, list)
            or len(indices) > 64
            or any(type(i) is not int or not 0 <= i < self.built.groups[name].size for i in indices)
            or not hasattr(self.built.groups[name], "v")
        ):
            raise ValueError("/stream/vsample: expected group and at most 64 valid neuron indices")

    def select(self, name: str, indices: list[int]) -> None:
        """Flush the old selection before changing its voltage columns."""
        self.validate_indices(name, indices)
        self.flush()
        self.simple = False
        self.vsample[name] = indices
        self.voltages[name] = []
        self.publish()

    def publish(self) -> None:
        """Atomically preserve current metadata for late subscribers."""
        if self.metadata is not None:
            issues = self.metadata.with_name("stream-issues.json")
            temporary = issues.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.issues))
            temporary.replace(issues)
            temporary = self.metadata.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.started()))
            temporary.replace(self.metadata)

    def started(self) -> dict[str, Any]:
        """Describe stream IDs and the actual network schedule."""
        return {
            "dt": self.built.net.dt,
            "groups": [
                {"stream": i, "name": n, "size": g.size, "inhibitory": g.inhibitory}
                for i, (n, g) in enumerate(self.built.groups.items())
            ],
            "synapses": [
                {
                    "stream": i,
                    "name": n,
                    "src": s.src.name,
                    "dst": s.dst.name,
                    "shape": [s.src.size, s.dst.size],
                }
                for i, (n, s) in enumerate(self.built.synapses.items())
            ],
            "scalars": [
                {"id": i, "name": n, "unit": "1/step" if n.startswith("rate:") else None}
                for i, n in enumerate(self.scalars)
            ],
            "vsample": [
                {"stream": self.groups.index(n), "group": n, "indices": indices}
                for n, indices in self.vsample.items()
            ],
            "schedule": schedule_table(self.built.net),
        }

    def forward(self, host: Any) -> None:
        """Copy state before the next simulation step can change it."""
        if self.simple:
            buffer, group, size = self.spike_groups[0]
            if group.name not in self.recorded:
                buffer.append(ctypes.string_at(group.spikes.data_ptr(), size))
            self.count += 1
            if self.count == 1:
                self.step0 = group.net.iteration
            if self.count == 100:
                self.flush(last=group.net.iteration == self.end_step)
            return
        net = self.built.net
        if not self.count:
            self.step0 = net.iteration
        for buffer, group, size in self.spike_groups:
            if group.name in self.recorded:
                continue
            spikes = group.spikes
            if not self.cpu_unbatched:
                spikes = first(spikes, net).to(device="cpu").contiguous()
            buffer.append(ctypes.string_at(spikes.data_ptr(), size))
        for name, indices in self.vsample.items():
            self.voltages[name].append(first(self.built.groups[name].v, net)[indices].tolist())
        for name in self.scalars:
            if name.startswith("rate:"):
                value = first(self.built.groups[name[5:]].spikes, net).float().mean().item()
            else:
                value = float(getattr(net, name))
            self.values[name].append(value)
        self.count += 1
        for name, interval in self.weights.items():
            if net.iteration % interval == 0:
                self.emit_weights(name, net.iteration)
        if self.count == 100:
            self.flush(last=net.iteration == self.end_step)

    def emit_weights(self, name: str, step: int) -> None:
        """Reduce large weight matrices using block means, including edge blocks."""
        synapse = self.built.synapses[name]
        weights = synapse.weights
        if weights is None:
            raise ValueError(f"/stream/weights: {name} has no weights")
        if self.built.net.independent:
            weights = weights[0]
        rows, cols = weights.shape
        rs, cs = math.ceil(rows / 256), math.ceil(cols / 256)
        if rs > 1 or cs > 1:
            weights = torch.stack(
                [
                    torch.stack(
                        [weights[r : r + rs, c : c + cs].float().mean() for c in range(0, cols, cs)]
                    )
                    for r in range(0, rows, rs)
                ]
            )
        self.ring.write(
            encode_weights(
                self.synapses.index(name),
                0,
                step,
                weights.shape[0],
                weights.shape[1],
                rs,
                cs,
                weights.detach().cpu(),
                self.flags
                | int(rs > 1 or cs > 1)
                | (
                    2
                    if self.end_step is not None and step + self.weights[name] > self.end_step
                    else 0
                ),
            )
        )

    def flush(self, *, last: bool = False) -> None:
        """Encode copied steps, preserving little-endian spike bit order."""
        if not self.count:
            return
        flags = self.flags | (2 if last else 0)
        for name in self.spikes:
            width = self.built.groups[name].size
            if name in self.recorded:
                pointer = self.recorded[name].data_ptr() + (self.step0 - 1) * width
                payload = ctypes.string_at(pointer, self.count * width)
            else:
                payload = b"".join(self.buffers[name])
            self.buffers[name].clear()
            batch = max(1, min(65535, (self.ring.slot_size - 36) // width))
            for offset in range(0, self.count, batch):
                count = min(batch, self.count - offset)
                frame = encode_spikes(
                    self.groups.index(name),
                    0,
                    self.step0 + offset,
                    count,
                    struct.pack("<I", width) + payload[offset * width : (offset + count) * width],
                    flags,
                )
                self.ring.write(bytes([129]) + frame[1:])
        for name, indices in self.vsample.items():
            self.ring.write(
                encode_vsample(
                    self.groups.index(name),
                    0,
                    self.step0,
                    indices,
                    self.voltages[name],
                    flags,
                )
            )
            self.voltages[name].clear()
        for i, name in enumerate(self.scalars):
            self.ring.write(encode_scalar(i, 0, self.step0, self.values[name], flags))
            self.values[name].clear()
        self.count = 0


def attach(built: Built, request: dict[str, Any], root: Path) -> StreamTap:
    """Attach to an initialized network without initializing existing behaviors again."""
    root = root.parent / ".streams" / root.name
    root.mkdir(parents=True, exist_ok=True)
    ring = Ring(root / "stream.ring", create=True)
    try:
        tap = StreamTap(built, request.get("stream", {}), ring)
    except Exception:
        ring.close()
        raise
    tap.end_step = request["steps"]
    built.net.schedule.append((built.net, tap))
    built.net.schedule.sort(key=lambda pair: pair[1].order)
    tap.metadata = root / "stream-started.json"
    tap.publish()
    return tap
