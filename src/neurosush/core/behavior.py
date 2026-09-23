"""Base class for everything that runs inside a simulation step."""

from __future__ import annotations

from typing import Any, ClassVar


class Behavior:
    """A unit of state and dynamics attached to a network, neuron group or synapse group.

    Subclasses set ``order`` (see :class:`~neurosush.core.order.Order`), validate their
    arguments in ``__init__``, allocate state in :meth:`initialize` and update it in
    :meth:`forward`. Setting ``enabled = False`` skips :meth:`forward`.
    """

    order: ClassVar[int]
    enabled: bool = True

    def initialize(self, host: Any) -> None:
        """Allocate state on ``host``; called once, in schedule order."""

    def forward(self, host: Any) -> None:
        """Advance ``host`` by one step."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
