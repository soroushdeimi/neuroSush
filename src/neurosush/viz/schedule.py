"""Step schedule inspection and execution lanes for visualization."""

from __future__ import annotations

import inspect
from typing import Any

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order


def _first_doc_line(cls: type) -> str:
    doc = inspect.getdoc(cls)
    if not doc:
        return ""
    return doc.splitlines()[0].strip()


def schedule_lanes() -> list[dict[str, Any]]:
    """Return all Order members as {name, order} dictionaries sorted ascending by order."""
    return [{"name": m.name, "order": int(m.value)} for m in sorted(Order, key=lambda m: m.value)]


def schedule_table(net: Network) -> list[dict[str, Any]]:
    """Return metadata for each behavior in execution order.

    Args:
        net: Simulation network to inspect (initialized if not already).

    Returns:
        List of behavior description dictionaries.
    """
    if not net.initialized:
        net.initialize()

    rows: list[dict[str, Any]] = []
    for index, (host, behavior) in enumerate(net.schedule):
        try:
            lane = Order(behavior.order).name
        except ValueError:
            lane = str(behavior.order)

        if isinstance(host, Network):
            host_name = getattr(host, "name", "network")
            host_kind = "network"
        elif isinstance(host, NeuronGroup):
            host_name = host.name
            host_kind = "neurons"
        elif isinstance(host, SynapseGroup):
            host_name = host.name
            host_kind = "synapses"
        else:
            host_name = getattr(host, "name", str(host))
            host_kind = "unknown"

        summary = _first_doc_line(type(behavior))
        reads = list(getattr(behavior, "reads", ()))
        writes = list(getattr(behavior, "writes", ()))

        rows.append(
            {
                "index": index,
                "order": int(behavior.order),
                "lane": lane,
                "host": host_name,
                "host_kind": host_kind,
                "behavior": type(behavior).__name__,
                "enabled": bool(behavior.enabled),
                "summary": summary,
                "reads": reads,
                "writes": writes,
            }
        )

    return rows
