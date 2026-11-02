from dataclasses import FrozenInstanceError

import pytest

from neurosush.tem import TEMConfig


def test_defaults_and_frozen():
    config = TEMConfig()
    assert config.modules == (12, 12, 8)
    with pytest.raises(FrozenInstanceError):
        config.sensory = 4


@pytest.mark.parametrize(
    ("kwargs", "value"),
    [
        ({"sensory": 0}, "0"),
        ({"compressed": 3}, "3"),
        ({"modules": ()}, r"\(\)"),
        ({"modules": (2, -1)}, "-1"),
        ({"hidden": 1.5}, "1.5"),
        ({"batch": True}, "True"),
        ({"lr_min": 1.0}, "1.0"),
        ({"memory_rate": -1}, "-1"),
        ({"memory_decay": 1.1}, "1.1"),
        ({"reg_g": float("nan")}, "nan"),
        ({"clip": 0}, "0"),
        ({"iterations": 0}, "0"),
    ],
)
def test_invalid_values(kwargs, value):
    with pytest.raises(ValueError, match=value):
        TEMConfig(**kwargs)


def test_schedules_hand_values():
    config = TEMConfig(decay_ramp=4, rate_ramp=8, latent_ramp=10, lr_every=2)
    assert config.schedules(0) == (0.00094, 0, 0, 0)
    assert config.schedules(2) == pytest.approx((0.00047, 0.49995, 0.125, 0.2))
    assert config.schedules(1000) == (0.00008, 0.9999, 0.5, 1.0)
    with pytest.raises(ValueError, match="-1"):
        config.schedules(-1)
