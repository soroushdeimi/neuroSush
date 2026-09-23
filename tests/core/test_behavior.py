from neurosush.core.behavior import Behavior
from neurosush.core.order import Order


def test_order_values_follow_the_step_sequence():
    sequence = [
        Order.INITIALIZATION,
        Order.PAYOFF,
        Order.NEUROMODULATOR,
        Order.SYNAPTIC_INPUT,
        Order.CURRENT_NORMALIZATION,
        Order.DENDRITE_STRUCTURE,
        Order.DENDRITE_INTEGRATION,
        Order.NEURON_DYNAMICS,
        Order.NOISE,
        Order.COMPETITION,
        Order.VOLTAGE_HOMEOSTASIS,
        Order.FIRE,
        Order.ACTIVITY_HOMEOSTASIS,
        Order.AXON,
        Order.SPIKE_GATHER,
        Order.TRACE,
        Order.PLASTICITY,
        Order.WEIGHT_NORMALIZATION,
        Order.WEIGHT_CLIP,
    ]
    assert list(Order) == sequence
    assert [int(o) for o in sequence] == sorted(int(o) for o in sequence)


def test_order_is_an_int():
    assert Order.FIRE == 340
    assert isinstance(Order.FIRE, int)


def test_base_behavior_hooks_are_no_ops():
    class Plain(Behavior):
        order = Order.FIRE

    b = Plain()
    assert b.initialize(object()) is None
    assert b.forward(object()) is None
    assert b.enabled is True


def test_repr_names_the_class():
    class Plain(Behavior):
        order = Order.FIRE

    assert repr(Plain()) == "Plain()"
