"""All-pairs connections between two sets of groups."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from neurosush.core.behavior import Behavior
from neurosush.core.network import Compartment, Network, NeuronGroup, SynapseGroup


def connect(
    net: Network,
    src: Sequence[NeuronGroup],
    dst: Sequence[NeuronGroup],
    behaviors: Callable[[], Iterable[Behavior]],
    *,
    compartment: Compartment | str = Compartment.PROXIMAL,
    tags: Iterable[str] = (),
) -> list[SynapseGroup]:
    """Create one synapse group for every ``(source, destination)`` pair.

    Args:
        net: Owning network.
        src: Source groups, for example an output port.
        dst: Destination groups, for example an input port.
        behaviors: Factory called once per pair, so no two synapse groups share state.
        compartment: Dendritic compartment of every destination.
        tags: Tags of every synapse group.
    """
    if not callable(behaviors):
        raise TypeError("behaviors must be a callable that returns fresh Behavior instances")
    tags = tuple(tags)
    return [
        SynapseGroup(net, s, d, behaviors(), compartment=compartment, tags=tags)
        for s in src
        for d in dst
    ]
