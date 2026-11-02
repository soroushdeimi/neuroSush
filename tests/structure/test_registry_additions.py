import pytest

from neurosush.core.network import Network
from neurosush.introspect import _REGISTRY, describe
from neurosush.structure.spec import BehaviorSpec, registered
from neurosush.viz.trace import Trace


def test_registry_additions() -> None:
    expected = [
        "ConductanceIntegration",
        "AdaptiveThreshold",
        "Refractory",
        "SegmentLearning",
        "TripletSTDP",
        "SpikeCounter",
        "PoissonInput",
    ]
    names = registered()
    for name in expected:
        assert name in names
        cls = _REGISTRY[name]
        desc = describe(cls)
        assert desc["usable"] is True

    assert isinstance(BehaviorSpec("SpikeCounter").build(), _REGISTRY["SpikeCounter"])
    assert isinstance(
        BehaviorSpec("PoissonInput", {"rates": 0.01}).build(), _REGISTRY["PoissonInput"]
    )

    net = Network(seed=7)
    assert net.seed == 7
    assert Trace.make_meta(net)["seed"] == 7


def test_network_seed_none() -> None:
    net = Network(seed=None)
    assert net.seed is None
    assert Trace.make_meta(net)["seed"] is None


def test_unregistered_behavior_raises() -> None:
    with pytest.raises(ValueError, match="unknown behavior"):
        BehaviorSpec("NonExistentBehavior").build()
