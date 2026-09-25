"""Base class for everything that runs inside a simulation step."""

from __future__ import annotations

from collections.abc import Hashable
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
    graph_safe: ClassVar[bool] = False  # True when forward() is unconditionally graph-ready

    def initialize(self, host: Any) -> None:
        """Allocate state on ``host``; called once, in schedule order."""

    def forward(self, host: Any) -> None:
        """Advance ``host`` by one step."""

    def graph_ready(self, host: Any) -> bool:
        """Whether ``forward`` on ``host`` may run inside a captured CUDA graph.

        A ready ``forward`` performs no host-device synchronization (``bool()``, ``.item()``,
        ``.tolist()`` or ``nonzero()`` on a CUDA tensor) and changes no Python-side state; a
        decision it makes in Python must instead be exposed through :meth:`graph_key`.
        Override only when readiness depends on the host; otherwise set :attr:`graph_safe`.

        Args:
            host: The network, neuron group or synapse group this behavior is attached to.

        Returns:
            :attr:`graph_safe` by default.
        """
        return self.graph_safe

    def graph_key(self, host: Any) -> Hashable:
        """The Python-side decision this behavior's upcoming step depends on.

        Called with ``host``'s iteration already set to the step about to run, so the graph
        stepper can capture a separate graph for each distinct value.

        Args:
            host: The network, neuron group or synapse group this behavior is attached to.

        Returns:
            ``None`` by default.
        """
        return None

    def prepare(self, host: Any) -> None:
        """Do host-side work before a step, such as staging the next input frame.

        Called once per step before the schedule runs, for behaviors that override it.

        Args:
            host: The network, neuron group or synapse group this behavior is attached to.
        """

    def state_dict(self) -> dict[str, Any]:
        """State held by the behavior itself that a checkpoint must save; none by default."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore what :meth:`state_dict` returned."""
        if state:
            raise KeyError(f"{type(self).__name__} keeps no state, got keys {sorted(state)}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
