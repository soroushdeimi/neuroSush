"""The Izhikevich neuron: one Euler step by hand, and the firing classes of the 2003 paper.

Firing classes are checked through signatures (ISI adaptation, bursts) rather than exact
spike times; dt = 0.1 with 5 sub-steps keeps the Euler discretization out of the picture.
"""

from itertools import pairwise

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup
from neurosush.neurons.models import Fire, Izhikevich

from .common import ConstantCurrent

SETS = {
    "regular": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
    "fast": {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0},
    "chattering": {"a": 0.02, "b": 0.2, "c": -50.0, "d": 2.0},
    "bursting": {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0},
    "low_threshold": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0},
}


def isis(params, current, *, steps=10000, dt=0.1, substeps=5):
    """Inter-spike intervals (in time units) of one neuron under constant current."""
    net = Network(dt=dt, dtype=torch.float64)
    group = NeuronGroup(
        net, 1, [ConstantCurrent(current), Izhikevich(**params, substeps=substeps), Fire()]
    )
    net.initialize()
    times = []
    for _ in range(steps):
        net.step()
        if bool(group.spikes.any()):
            times.append(net.iteration * dt)
    return [b - a for a, b in pairwise(times)]


class TestOneStep:
    def test_two_half_steps_then_u_from_the_updated_v(self):
        net = Network(dt=1.0, dtype=torch.float64)
        model = Izhikevich(a=0.02, b=0.2, c=-65.0, d=8.0)
        group = NeuronGroup(net, 1, [ConstantCurrent(10.0), model])
        net.initialize()
        net.step()
        v, u = -65.0, 0.2 * -65.0  # u_init = b v_init = -13
        for _ in range(2):
            v += 0.5 * (0.04 * v * v + 5 * v + 140 - u + 10.0)
        u += 0.02 * (0.2 * v - u)
        assert group.v.item() == pytest.approx(v, abs=1e-12)
        assert group.u.item() == pytest.approx(u, abs=1e-12)
        # by hand: -65 + 0.5 * 7 = -61.5, then -61.5 + 0.5 * 6.79 = -58.105
        assert group.v.item() == pytest.approx(-58.105, abs=1e-12)
        assert group.u.item() == pytest.approx(-13.0 + 0.02 * 1.379, abs=1e-12)

    def test_single_substep_is_forward_euler(self):
        net = Network(dt=0.5, dtype=torch.float64)
        group = NeuronGroup(
            net, 1, [ConstantCurrent(4.0), Izhikevich(**SETS["regular"], substeps=1)]
        )
        net.initialize()
        net.step()
        v = -65.0 + 0.5 * (0.04 * 65.0**2 - 325.0 + 140.0 + 13.0 + 4.0)
        assert group.v.item() == pytest.approx(v, abs=1e-12)
        assert group.u.item() == pytest.approx(-13.0 + 0.5 * 0.02 * (0.2 * v + 13.0), abs=1e-12)

    def test_spike_resets_v_to_c_and_adds_d_to_u(self):
        net = Network(dt=1.0, dtype=torch.float64)
        group = NeuronGroup(
            net,
            2,
            [Izhikevich(a=0.02, b=0.2, c=-65.0, d=8.0, v_init=torch.tensor([35.0, 0.0])), Fire()],
        )
        net.initialize()
        u0 = group.u.clone()
        group.model.fire(group)
        assert group.spikes.tolist() == [True, False]
        assert group.v.tolist() == [-65.0, 0.0]
        assert group.u.tolist() == [u0[0].item() + 8.0, u0[1].item()]


class TestFiringClasses:
    def test_regular_spiking_adapts(self):
        intervals = isis(SETS["regular"], 10.0)
        assert len(intervals) >= 15
        # the first interval is the shortest; the adapted rate is about half the initial one
        assert intervals[0] < 0.6 * intervals[-1]
        assert max(intervals[-10:]) - min(intervals[-10:]) < 0.5  # then tonic

    def test_fast_spiking_does_not_adapt_and_fires_faster(self):
        fast = isis(SETS["fast"], 10.0)
        regular = isis(SETS["regular"], 10.0)
        assert len(fast) > 3 * len(regular)
        steady = fast[5:]
        assert max(steady) / min(steady) < 1.15
        assert fast[-1] < 1.25 * fast[5]  # no slowing down

    def test_chattering_fires_repeated_bursts(self):
        intervals = isis(SETS["chattering"], 10.0)
        long_gaps = [i for i, x in enumerate(intervals) if x > 30.0]
        assert len(long_gaps) >= 2
        # every pause is followed by a burst of short intervals, then the next pause
        assert all(x < 8.0 for x in intervals[: long_gaps[0]])
        assert long_gaps[0] >= 3
        for first, second in pairwise(long_gaps):
            burst = intervals[first + 1 : second]
            assert len(burst) >= 3
            assert all(x < 8.0 for x in burst)

    def test_intrinsically_bursting_starts_with_a_burst_then_fires_tonically(self):
        intervals = isis(SETS["bursting"], 10.0)
        assert all(x < 6.0 for x in intervals[:2])  # initial burst
        assert intervals[2] > 30.0
        tonic = intervals[3:]
        assert len(tonic) >= 10
        assert max(tonic) - min(tonic) < 0.5
        assert min(tonic) > 25.0

    def test_low_threshold_spiking_fires_where_regular_spiking_is_silent(self):
        assert len(isis(SETS["low_threshold"], 2.0)) >= 10
        assert isis(SETS["regular"], 2.0) == []

    def test_one_ms_steps_still_show_adaptation(self):
        intervals = isis(SETS["regular"], 10.0, steps=1000, dt=1.0, substeps=2)
        assert len(intervals) >= 10
        assert intervals[0] < 0.7 * intervals[-1]
