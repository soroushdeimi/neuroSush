"""Active dendritic segments against the temporal memory and their closed forms."""

import math

import pytest
import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order
from neurosush.htm.sdr import random_sdr
from neurosush.htm.temporal_memory import TemporalMemory
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses.segments import ActiveSegments, segment_counts
from neurosush.synapses.traces import SpikeGather

from .common import ScriptedSpikes

f64 = torch.float64


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def tm_segments(tm, per_cell):
    """The memory's segments in the ``(cells, per_cell, synapses)`` layout, and their order."""
    presynaptic = torch.full((tm.n_cells, per_cell, tm.max_synapses), -1, dtype=torch.long)
    permanence = torch.zeros(tm.n_cells, per_cell, tm.max_synapses)
    slot = torch.zeros(tm.n_cells, dtype=torch.long)
    where = []
    for segment in range(tm.n_segments):
        cell = int(tm.segment_cell[segment])
        s = int(slot[cell])
        presynaptic[cell, s] = tm.presynaptic[segment]
        permanence[cell, s] = tm.permanence[segment]
        slot[cell] += 1
        where.append((cell, s))
    return presynaptic, permanence, where


class TestTemporalMemoryEquivalence:
    def test_counts_and_predictions_match_the_temporal_memory(self):
        # the same segments and the same active cells give the same dendrite activity
        tm = TemporalMemory(
            128,
            4,
            activation_threshold=6,
            min_threshold=4,
            max_new_synapses=8,
            max_synapses_per_segment=10,
            initial_permanence=0.51,
            seed=0,
        )
        sequence = random_sdr(128, 8, batch=(6,), generator=gen())
        for _ in range(4):
            tm.reset()
            for x in sequence:
                tm.compute(x)
        per_cell = int(torch.bincount(tm.segment_cell[: tm.n_segments]).max())
        presynaptic, permanence, where = tm_segments(tm, per_cell)
        for x in sequence:
            tm.compute(x, learn=False)
            connected, potential = tm._segment_activity(tm.active_cells)
            got_c, got_p = segment_counts(tm.active_cells, presynaptic, permanence, tm.connected)
            for segment, (cell, s) in enumerate(where):
                assert got_c[cell, s] == connected[segment]
                assert got_p[cell, s] == potential[segment]
            predicted = (got_c >= tm.activation_threshold).any(-1)
            assert torch.equal(predicted, tm.predictive_cells)


class TestCoincidenceDetection:
    @pytest.mark.parametrize(
        ("synapses", "connected_share", "threshold", "p"),
        [
            (20, 1.0, 8, 0.3),
            (20, 0.5, 4, 0.3),
            (32, 0.75, 10, 0.25),
        ],
    )
    def test_spike_probability_is_a_binomial_tail(self, synapses, connected_share, threshold, p):
        # with M_c connected synapses on distinct inputs that fire independently with
        # probability p, a segment spikes with probability P(Bin(M_c, p) >= threshold)
        n_src, n_dst, per_cell, trials = 400, 50, 4, 400
        g = gen(1)
        presynaptic = torch.stack(
            [torch.randperm(n_src, generator=g)[:synapses] for _ in range(n_dst * per_cell)]
        ).reshape(n_dst, per_cell, synapses)
        m_c = round(connected_share * synapses)
        permanence = torch.cat([torch.full((m_c,), 0.8), torch.full((synapses - m_c,), 0.2)])
        permanence = permanence.expand(n_dst, per_cell, synapses)
        spikes = torch.rand(trials, n_src, generator=g) < p
        counts, _ = segment_counts(spikes, presynaptic, permanence, 0.5)
        fired = (counts >= threshold).double()
        expected = sum(
            math.comb(m_c, k) * p**k * (1 - p) ** (m_c - k) for k in range(threshold, m_c + 1)
        )
        error = math.sqrt(expected * (1 - expected) / fired.numel())
        assert abs(fired.mean().item() - expected) < 5 * error


def segment_net(
    volleys,
    *,
    firing=5,
    silent_cells=(),
    plateau,
    dt=1.0,
    threshold=3,
    cells=2,
    drive=None,
    amplitude=1.0,
    distal_gain=None,
):
    """Five inputs onto ``cells`` cells with one segment each (all five synapses connected).

    ``volleys`` are the steps at which the first ``firing`` inputs fire; ``silent_cells``
    get no synapses.
    """
    net = Network(dt=dt, dtype=f64)
    src = NeuronGroup(net, 5, [ScriptedSpikes({t: list(range(firing)) for t in volleys}), Axon()])
    behaviors = [DendriteStructure(), DendriteIntegration(distal_gain=distal_gain)]
    if drive is not None:
        behaviors.append(drive)
    behaviors += [LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0), Fire()]
    dst = NeuronGroup(net, cells, behaviors)
    presynaptic = torch.arange(5).expand(cells, 1, 5).clone()
    presynaptic[list(silent_cells)] = -1
    syn = SynapseGroup(
        net,
        src,
        dst,
        [
            ActiveSegments(
                segments=1,
                synapses=5,
                activation_threshold=threshold,
                plateau=plateau,
                amplitude=amplitude,
                presynaptic=presynaptic,
                permanence=torch.ones(cells, 1, 5),
            ),
            SpikeGather(),
        ],
        compartment="distal",
    )
    return net, dst, syn


class TestPlateau:
    @pytest.mark.parametrize(("plateau", "dt"), [(5.0, 1.0), (7.5, 0.5), (3.2, 1.0)])
    def test_plateau_lasts_ceil_of_its_duration_in_steps(self, plateau, dt):
        # a volley at step 3 is gathered at step 3 and read by the segments at step 4
        net, _, syn = segment_net([3], plateau=plateau, dt=dt)
        on = []
        for _ in range(40):
            net.step()
            if bool(syn.I[0] > 0):
                on.append(net.iteration)
        duration = math.ceil(plateau / dt - 1e-9)
        assert on == list(range(4, 4 + duration))

    def test_a_new_dendritic_spike_restarts_the_plateau(self):
        net, _, syn = segment_net([3, 6], plateau=5.0)
        on = []
        for _ in range(30):
            net.step()
            if bool(syn.I[0] > 0):
                on.append(net.iteration)
        assert on == list(range(4, 12))  # 4..8 from the first, restarted at 7 for 5 steps

    def test_below_threshold_input_does_nothing(self):
        net, _, syn = segment_net([3], plateau=5.0, threshold=5, firing=4)  # 4 of 5
        for _ in range(20):
            net.step()
            assert not bool(syn.I.any())


class ProximalDrive(Behavior):
    """A constant proximal current (after DendriteStructure, before integration)."""

    order = Order.DENDRITE_STRUCTURE + 1

    def __init__(self, value):
        self.value = value

    def forward(self, group):
        group.I_proximal = group.state(self.value)


class TestSomaticCoupling:
    """While a plateau holds I_distal = A, the LIF Euler step is linear below the priming
    limit L = v_rest + g (theta - v_rest):
    v_{n+1} = v_n (1 - dt/tau - dt k) + dt (v_rest + R I) / tau + dt k L with k = tanh(A),
    so v_n = v* + (v_0 - v*) (1 - dt/tau - dt k)^n towards
    v* = ((v_rest + R I) / tau + k L) / (1 / tau + k)."""

    GAIN, AMPLITUDE = 0.8, 0.5

    def fixed_point(self, drive):
        k, limit = math.tanh(self.AMPLITUDE), -65.0 + self.GAIN * 10.0
        return ((-65.0 + drive) / 10.0 + k * limit) / (1 / 10.0 + k), k

    def test_a_predicted_cell_is_depolarized_but_does_not_fire(self):
        net, dst, _ = segment_net(
            range(1, 400), plateau=5.0, amplitude=self.AMPLITUDE, distal_gain=self.GAIN, cells=1
        )
        v_star, k = self.fixed_point(0.0)
        net.run(1)  # the plateau starts at step 2
        v0 = dst.v.item()
        for n in range(1, 300):
            net.step()
            assert not bool(dst.spikes.any())
            expected = v_star + (v0 - v_star) * (1 - 1 / 10.0 - k) ** n
            assert dst.v.item() == pytest.approx(expected, abs=1e-9)
        assert v_star < -55.0  # it settles below the threshold, above rest
        assert v_star > -65.0

    def test_a_predicted_cell_fires_first_by_the_predicted_number_of_steps(self):
        # both cells get the same proximal drive; only cell 0 has a plateau
        drive = 12.0  # v_inf = -53 > theta: both fire eventually
        net, dst, _ = segment_net(
            range(1, 200),
            plateau=5.0,
            amplitude=self.AMPLITUDE,
            distal_gain=self.GAIN,
            drive=ProximalDrive(drive),
            cells=2,
            silent_cells=(1,),
        )
        first = [None, None]
        for _ in range(200):
            net.step()
            for cell in (0, 1):
                if first[cell] is None and bool(dst.spikes[cell]):
                    first[cell] = net.iteration

        def crossing(k, limit, start_step):
            # iterate the exact piecewise-linear map from v = v_rest until theta
            v, n = -65.0, 0
            while v < -55.0:
                n += 1
                active = n >= start_step
                prime = k * max(limit - v, 0.0) if active else 0.0
                v = v + 0.1 * ((-65.0 - v) + drive) + prime
            return n

        k, limit = math.tanh(self.AMPLITUDE), -65.0 + self.GAIN * 10.0
        assert first[1] == crossing(0.0, limit, 10**9)
        assert first[0] == crossing(k, limit, 2)
        assert first[0] < first[1]
