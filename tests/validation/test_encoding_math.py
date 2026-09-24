"""Spike encoders against their distributions, and delays against exact arrival times."""

import math

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.encoding import interval_poisson, rate_poisson
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import DendriteStructure
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import DelayInit, WeightInit
from neurosush.synapses.traces import SpikeGather

from .common import ScriptedSpikes


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def intervals(train):
    """Inter-spike intervals (in steps) of every column of a ``(steps, n)`` spike train."""
    out = []
    for column in train.T:
        times = column.nonzero().flatten()
        out.append(times.diff())
    return torch.cat(out)


class TestRatePoisson:
    def test_counts_are_binomial(self):
        # every step is an independent Bernoulli(p) trial: counts ~ Binomial(steps, p)
        p, steps, n = 0.07, 400, 3000
        counts = rate_poisson(torch.full((n,), p), steps, generator=gen()).sum(0).double()
        mean, var = steps * p, steps * p * (1 - p)
        assert abs(counts.mean().item() - mean) < 5 * math.sqrt(var / n)
        # the sample variance of n draws has standard error ~ var * sqrt(2 / n)
        assert abs(counts.var().item() - var) < 5 * var * math.sqrt(2 / n)

    def test_intervals_are_geometric(self):
        # P(ISI = k) = p (1 - p)^(k - 1); a chi-square statistic with df bins has mean df and
        # standard deviation sqrt(2 df)
        p = 0.1
        isi = intervals(rate_poisson(torch.full((500,), p), 1000, generator=gen(1)))
        bins = 30
        observed = torch.bincount(isi.clamp(max=bins), minlength=bins + 1)[1:].double()
        probs = torch.tensor(
            [p * (1 - p) ** (k - 1) for k in range(1, bins)] + [(1 - p) ** (bins - 1)]
        )
        expected = probs.double() * len(isi)
        chi2 = ((observed - expected) ** 2 / expected).sum().item()
        df = bins - 1
        assert abs(chi2 - df) < 5 * math.sqrt(2 * df)

    def test_elements_are_independent(self):
        # the correlation of two independent trains is zero within its sampling error
        train = rate_poisson(torch.full((2,), 0.2), 20000, generator=gen(2)).double()
        r = torch.corrcoef(train.T)[0, 1].item()
        assert abs(r) < 5 / math.sqrt(20000)


class TestIntervalPoisson:
    @pytest.mark.parametrize("rate", [0.05, 0.2, 0.5])
    def test_mean_interval(self, rate):
        # intervals are Poisson(1 / rate) with zeros raised to one step:
        # E = 1 / rate + P(0) = 1 / rate + exp(-1 / rate)
        lam = 1 / rate
        isi = intervals(interval_poisson(torch.full((400,), rate), 2000, generator=gen(3))).double()
        expected = lam + math.exp(-lam)
        # Var of max(1, Poisson) is at most lam + 1
        assert abs(isi.mean().item() - expected) < 5 * math.sqrt((lam + 1) / len(isi))


class TestDelays:
    def test_spikes_arrive_after_the_axonal_plus_dendritic_delay(self):
        # a spike at step t0 is gathered after src_delay steps, becomes current on the next
        # step and waits dst_delay more steps in the dendrite: it arrives at
        # t0 + src_delay + dst_delay + 1, with its full weight
        n, t0 = 12, 3
        g = gen(4)
        src_delay = torch.randint(0, 5, (n,), generator=g)
        dst_delay = torch.randint(0, 4, (n,), generator=g)
        weights = torch.diag(torch.rand(n, generator=g) + 0.5)
        net = Network(dtype=torch.float64)
        src = NeuronGroup(net, n, [ScriptedSpikes({t0: list(range(n))}), Axon(max_delay=5)])
        dst = NeuronGroup(net, n, [DendriteStructure(proximal_depth=4)])
        SynapseGroup(
            net,
            src,
            dst,
            [
                WeightInit(weights=weights),
                DelayInit(delays=src_delay, side="src"),
                DelayInit(delays=dst_delay, side="dst"),
                DenseInput(),
                SpikeGather(),
            ],
        )
        arrivals = torch.full((n,), -1)
        amplitude = torch.zeros(n, dtype=torch.float64)
        for _ in range(t0 + 12):
            net.step()
            current = dst.I_proximal
            new = (current != 0) & (arrivals < 0)
            arrivals[new] = net.iteration
            amplitude[new] = current[new]
        assert arrivals.tolist() == (t0 + src_delay + dst_delay + 1).tolist()
        torch.testing.assert_close(amplitude, weights.diag().double())
