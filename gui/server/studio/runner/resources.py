"""Conservative tensor allocation estimates before spawning a run."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from studio.errors import ApiError


def memory_limit() -> int:
    """Read host and cgroup limits, reserving half for the server and runtime."""
    limits = [int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))]
    for filename in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            limits.append(int(Path(filename).read_text().strip()))
        except (OSError, ValueError):
            continue
    return min(limits) // 2


def estimate_bytes(spec: dict[str, Any]) -> int:
    """Budget dense state, learning buffers and bounded recordings without tensors."""
    settings = spec.get("network", {})
    width = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}[
        settings.get("dtype", "float32")
    ]
    batch = settings.get("batch_size") or 1
    groups = dict(spec.get("groups", {}))
    synapses = list(spec.get("synapses", []))
    for column_name, column in spec.get("columns", {}).items():
        for layer_name, layer in column["layers"].items():
            prefix = f"{column_name}.{layer_name}."
            groups.update({prefix + key: value for key, value in layer["groups"].items()})
            synapses.extend(
                {**s, "src": prefix + s["src"], "dst": prefix + s["dst"]}
                for s in layer.get("synapses", [])
            )
        synapses.extend(
            {**s, "src": column_name + "." + s["src"], "dst": column_name + "." + s["dst"]}
            for s in column.get("synapses", [])
        )
    sizes = {
        name: group["shape"] if isinstance(group["shape"], int) else math.prod(group["shape"])
        for name, group in groups.items()
    }
    if any(size < 1 for size in sizes.values()):
        raise ValueError("/groups: shapes must be positive")
    total = sum(sizes.values()) * width * batch * 32
    targets = dict(sizes)
    targets["network"] = 1
    for synapse in synapses:
        count = sizes[synapse["src"]] * sizes[synapse["dst"]]
        targets[synapse.get("name", synapse["src"] + "->" + synapse["dst"])] = count
        total += count * width * 8 * (batch if settings.get("independent") else 1)
    for recording in spec.get("recordings", []):
        total += (
            targets[recording["target"]]
            * len(recording["attributes"])
            * recording.get("max_steps", 1000)
            * width
            * batch
        )
    return total


def preflight(spec: dict[str, Any], reserved: int = 0) -> int:
    """Reject requests whose estimated live allocations exceed available budget."""
    estimate = estimate_bytes(spec)
    limit = memory_limit()
    if estimate + reserved > limit:
        message = f"Estimated {estimate / 2**20:.1f} MB; limit {limit / 2**20:.1f} MB"
        raise ApiError(
            "memory_budget",
            message,
            422,
            [{"path": "/spec", "message": message, "fix": "Reduce shapes or recordings."}],
        )
    return estimate
