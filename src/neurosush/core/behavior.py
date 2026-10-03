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
    # True when the behavior computes every batch member on its own state in a network created
    # with ``independent=True`` (no mean over the batch, per-member weights); anything else
    # makes such a network refuse to initialize.
    independent_ok: ClassVar[bool] = False
    # Random numbers drawn ahead of this step by :meth:`draw`, set by the compiled stepper only
    # (forward falls back to drawing from the group when a name is missing).
    drawn: dict[str, Any] = {}  # noqa: RUF012 - never mutated; steppers assign a new dict

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

    def compile_ready(self, host: Any) -> bool:
        """Whether ``forward`` on ``host`` may run inside the compiled stepper.

        The same contract as :meth:`graph_ready` (no host-device synchronization, no Python-side
        state changes, Python decisions exposed through :meth:`graph_key`), except that random
        numbers come from :meth:`draw`, so a generator does not matter.

        Args:
            host: The network, neuron group or synapse group this behavior is attached to.

        Returns:
            :meth:`graph_ready` by default.
        """
        return self.graph_ready(host)

    def draw(self, host: Any) -> dict[str, Any]:
        """Random numbers this behavior's next ``forward`` needs, drawn eagerly.

        The compiled stepper cannot trace draws from a ``torch.Generator``, so it calls this
        before the compiled step, in schedule order (the order eager stepping draws in), and
        exposes the result as ``self.drawn`` during ``forward``. A behavior that overrides this
        reads ``self.drawn.get(name)`` in ``forward`` and draws from the group when it is
        ``None``, which keeps eager stepping unchanged.

        Args:
            host: The network, neuron group or synapse group this behavior is attached to.

        Returns:
            Named tensors; empty by default.
        """
        return {}

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

    def reset_state(self, host: Any) -> None:
        """Clear the per-sample dynamic state this behavior keeps on ``host``.

        Clears voltages, currents, traces, spike histories and countdowns in place, and keeps
        learned and parameter state (weights, thresholds, theta, homeostasis counters,
        permanences). In place matters: a captured CUDA graph keeps tensor addresses. A
        buffer reset (:meth:`~neurosush.core.buffers.HistoryBuffer.reset`) also moves a
        Python head index, which a graph would freeze: that is harmless for depth-1 buffers
        (the only ones a graph can capture), but a deeper buffer is reset correctly only
        between eager steps.

        Called by :meth:`~neurosush.core.network.Network.reset_state` for every behavior,
        enabled or not. The default keeps nothing per sample and does nothing.

        Args:
            host: The network, neuron group or synapse group this behavior is attached to.
        """

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
