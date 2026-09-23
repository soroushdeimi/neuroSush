"""Network-wide reward signal and dopamine concentration."""

from __future__ import annotations

from collections.abc import Callable

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network
from neurosush.core.order import Order


class Payoff(Behavior):
    """Sets ``net.payoff`` every step from a user function (reward is positive).

    Args:
        fn: Function that takes a Network and returns a float.
        initial: Initial payoff value.
    """

    order = Order.PAYOFF

    def __init__(self, fn: Callable[[Network], float], *, initial: float = 0.0) -> None:
        self.fn = fn
        self.initial = initial

    def initialize(self, net: Network) -> None:
        """Set the initial payoff: net.payoff = float(initial)."""
        net.payoff = float(self.initial)

    def forward(self, net: Network) -> None:
        """Update the payoff: net.payoff = float(self.fn(net))."""
        net.payoff = float(self.fn(net))


class Dopamine(Behavior):
    """Extracellular dopamine driven by the payoff: ``dd/dt = -d / tau + payoff``.

    Args:
        tau: Decay time constant, in the unit of ``dt``.
        initial: Initial dopamine concentration.
    """

    order = Order.NEUROMODULATOR

    def __init__(self, *, tau: float, initial: float = 0.0) -> None:
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau
        self.initial = initial

    def initialize(self, net: Network) -> None:
        """Set the initial dopamine: net.dopamine = float(initial).

        Raises:
            RuntimeError: If the network does not have a Payoff behavior.
        """
        if not hasattr(net, "payoff"):
            raise RuntimeError("Dopamine needs a Payoff behavior on the network")
        net.dopamine = float(self.initial)

    def forward(self, net: Network) -> None:
        """Update the dopamine: net.dopamine += net.dt * (-net.dopamine / self.tau + net.payoff)."""
        net.dopamine += net.dt * (-net.dopamine / self.tau + net.payoff)
