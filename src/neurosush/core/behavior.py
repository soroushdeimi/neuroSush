"""Base class for everything that runs inside a simulation step."""

from __future__ import annotations

from typing import Any, ClassVar


class Behavior:
    """A unit of state and dynamics attached to a network, neuron group or synapse group.

    Subclasses set ``order`` (see :class:`~neurosush.core.order.Order`), validate their
    arguments in ``__init__``, allocate state in :meth:`initialize` and update it in
    :meth:`forward`. Setting ``enabled = False`` skips :meth:`forward`. State kept on the
    behavior itself (not on its host) goes through :meth:`state_dict` so that checkpoints
    can restore it.
    """

    order: ClassVar[int]
    enabled: bool = True

    def initialize(self, host: Any) -> None:
        """Allocate state on ``host``; called once, in schedule order."""

    def forward(self, host: Any) -> None:
        """Advance ``host`` by one step."""

    def state_dict(self) -> dict[str, Any]:
        """State held by the behavior itself that a checkpoint must save; none by default."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore what :meth:`state_dict` returned."""
        if state:
            raise KeyError(f"{type(self).__name__} keeps no state, got keys {sorted(state)}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
