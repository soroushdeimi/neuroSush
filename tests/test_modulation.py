import pytest

from neurosush.core.network import Network
from neurosush.core.order import Order
from neurosush.modulation import Dopamine, Payoff


class TestModulation:
    def test_payoff_calls_the_function_every_step(self):
        values = iter([1.0, -2.0])
        net = Network(behaviors=[Payoff(lambda net: next(values), initial=0.5)])
        net.initialize()
        assert net.payoff == 0.5
        net.step()
        assert net.payoff == 1.0
        net.step()
        assert net.payoff == -2.0

    def test_dopamine_dynamics(self):
        net = Network(dt=0.5, behaviors=[Payoff(lambda net: 2.0), Dopamine(tau=4.0, initial=1.0)])
        net.initialize()
        assert net.dopamine == 1.0
        net.step()
        # d += dt * (-d / tau + payoff) = 1 + 0.5 * (-0.25 + 2)
        assert net.dopamine == pytest.approx(1.875)

    def test_dopamine_needs_payoff(self):
        net = Network(behaviors=[Dopamine(tau=1.0)])
        with pytest.raises(RuntimeError, match="Payoff"):
            net.initialize()

    def test_dopamine_tau_must_be_positive(self):
        with pytest.raises(ValueError, match="tau"):
            Dopamine(tau=0.0)


def test_payoff_runs_before_dopamine():
    assert Payoff.order == Order.PAYOFF < Dopamine.order == Order.NEUROMODULATOR
