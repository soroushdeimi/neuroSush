"""Network-wide reward signal and dopamine concentration."""

from __future__ import annotations

from collections.abc import Callable

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network
from neurosush.core.order import Order


class Payoff(Behavior):
    """Network-wide reward signal.

    Args:
        fn: Function that takes a Network and returns a float.
        initial: Initial payoff value.
    """

    order = Order.PAYOFF

    def __init__(self, fn: Callable[[Network], float], *, initial: float = 0.0) -> None:
        """Initialize the Payoff behavior."""
        self.fn = fn
        self.initial = initial
        self.payoff = 0.0

    def initialize(self, net: Network) -> None:
        """Set the initial payoff: net.payoff = float(initial)."""
        net.payoff = float(self.initial)

    def forward(self, net: Network) -> None:
        """Update the payoff: net.payoff = float(self.fn(net))."""
        net.payoff = float(self.fn(net))


class Dopamine(Behavior):
    """Network-wide dopamine concentration.

    Args:
        tau: Time constant of dopamine dynamics.
        initial: Initial dopamine concentration.
    """

    order = Order.NEUROMODULATOR

    def __init__(self, *, tau: float, initial: float = 0.0) -> None:
        """Initialize the Dopamine behavior.

        Raises:
            ValueError: If tau is not positive.
        """
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        self.tau = tau
        self.initial = initial
        self.dopamine = 0.0

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
