"""Triplet STDP against closed forms built from the Euler trace recursions.

Every trace decays by ``c = 1 - dt / tau`` per step, so a spike ``k`` steps ago contributes
``c^k``. Traces include the current step's spike for ``r1`` and ``o1`` (like the pair
``STDP``) while ``r2`` and ``o2`` enter the triplet terms as their values before the
current increment.
"""

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.synapses.currents import DenseInput, OneToOneInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces
from neurosush.synapses.triplet import TripletSTDP, triplet_dense, triplet_one_to_one

from .common import ScriptedSpikes

# visual cortex, minimal-ish parameters in the style of Pfister and Gerstner (2006)
P = {
    "a2_plus": 0.005,
    "a3_plus": 0.006,
    "a2_minus": 0.007,
    "a3_minus": 0.0023,
    "tau_plus": 16.8,
    "tau_minus": 33.7,
    "tau_x": 101.0,
    "tau_y": 125.0,
}


def c(tau, k=1):
    return (1 - 1.0 / tau) ** k


def run_triplet(pre, post, steps, params=None, n_src=1, n_dst=1, w0=0.5, **extra):
    """Final weights of a dense synapse driven by scripted ``{step: [neurons]}`` spike trains."""
    net = Network(dtype=torch.float64)
    src = NeuronGroup(net, n_src, [ScriptedSpikes(pre), Axon()])
    dst = NeuronGroup(net, n_dst, [ScriptedSpikes(post), Axon()])
    syn = SynapseGroup(
        net,
        src,
        dst,
        [
            WeightInit(weights=torch.full((n_src, n_dst), w0, dtype=torch.float64)),
            DenseInput(),
            SpikeGather(),
            TripletSTDP(**(params or P), **extra),
        ],
    )
    net.run(steps)
    return syn.weights - w0


class TestPairLimit:
    def test_without_triplet_terms_equals_pair_stdp(self):
        g = torch.Generator().manual_seed(3)
        steps = 80
        spikes = torch.rand(steps, 5, generator=g) < 0.15
        pre = {s + 1: [i for i in range(3) if spikes[s, i]] for s in range(steps)}
        post = {s + 1: [i for i in range(2) if spikes[s, 3 + i]] for s in range(steps)}
        # force simultaneous spikes
        pre[40], post[40] = [0, 1], [0, 1]

        pair = {**P, "a3_plus": 0.0, "a3_minus": 0.0}
        triplet = run_triplet(pre, post, steps, pair, n_src=3, n_dst=2)

        net = Network(dtype=torch.float64)
        src = NeuronGroup(net, 3, [ScriptedSpikes(pre), Axon()])
        dst = NeuronGroup(net, 2, [ScriptedSpikes(post), Axon()])
        syn = SynapseGroup(
            net,
            src,
            dst,
            [
                WeightInit(weights=torch.full((3, 2), 0.5, dtype=torch.float64)),
                DenseInput(),
                SpikeGather(),
                Traces(tau_pre=P["tau_plus"], tau_post=P["tau_minus"]),
                STDP(a_plus=P["a2_plus"], a_minus=P["a2_minus"]),
            ],
        )
        net.run(steps)
        torch.testing.assert_close(triplet, syn.weights - 0.5, rtol=1e-12, atol=1e-14)


class TestTripletTerms:
    def test_post_pre_post_adds_the_triplet_potentiation(self):
        # post at 5, pre at 10, post at 10 + lag: the second post spike sees r1 = c+^lag and
        # o2_before = c_y^(lag + 5); the pre spike at 10 depresses with o1 = c-^5, r2_before = 0
        lag = 7
        got = run_triplet({10: [0]}, {5: [0], 10 + lag: [0]}, 10 + lag + 1).item()
        ltp = c(P["tau_plus"], lag) * (P["a2_plus"] + P["a3_plus"] * c(P["tau_y"], lag + 5))
        ltd = c(P["tau_minus"], 5) * P["a2_minus"]
        assert got == pytest.approx(ltp - ltd, abs=1e-14)

    def test_pre_post_pair_has_no_triplet_term(self):
        lag = 7
        got = run_triplet({10: [0]}, {10 + lag: [0]}, 10 + lag + 1).item()
        assert got == pytest.approx(P["a2_plus"] * c(P["tau_plus"], lag), abs=1e-14)

    def test_pre_post_pre_adds_the_triplet_depression(self):
        # pre at 5, post at 10, pre at 10 + lag: depression o1 = c-^lag * (a2- + a3- c_x^(lag+5))
        # and the first pre spike potentiates the post spike with r1 = c+^5
        lag = 4
        got = run_triplet({5: [0], 10 + lag: [0]}, {10: [0]}, 10 + lag + 1).item()
        ltp = P["a2_plus"] * c(P["tau_plus"], 5)
        ltd = c(P["tau_minus"], lag) * (P["a2_minus"] + P["a3_minus"] * c(P["tau_x"], lag + 5))
        assert got == pytest.approx(ltp - ltd, abs=1e-14)

    def test_simultaneous_pair_contributes_both_terms(self):
        # post at 7, then pre and post together at 10: LTP uses r1 = 1 (the current pre spike)
        # and o2_before = c_y^3; LTD uses o1 = 1 + c-^3 (the current post spike included)
        got = run_triplet({10: [0]}, {7: [0], 10: [0]}, 11).item()
        ltp = P["a2_plus"] + P["a3_plus"] * c(P["tau_y"], 3)
        ltd = (1 + c(P["tau_minus"], 3)) * P["a2_minus"]
        assert got == pytest.approx(ltp - ltd, abs=1e-14)

    def test_lone_simultaneous_pair_is_the_pair_difference(self):
        got = run_triplet({10: [0]}, {10: [0]}, 11).item()
        assert got == pytest.approx(P["a2_plus"] - P["a2_minus"], abs=1e-15)


def expected_total(period, lag, n, p):
    """Closed form of the total change of ``n`` pre-post pairings (``all`` interaction).

    Pre spike ``j`` at ``j * period`` and post spike ``j`` at ``j * period + lag``.
    Returns the per-pairing (LTP, LTD) lists.
    """
    ltp, ltd = [], []
    for k in range(n):
        r1 = sum(c(p["tau_plus"], lag + (k - j) * period) for j in range(k + 1))
        o2 = sum(c(p["tau_y"], (k - j) * period) for j in range(k))
        ltp.append(r1 * (p["a2_plus"] + p["a3_plus"] * o2))
        o1 = sum(c(p["tau_minus"], (k - j) * period - lag) for j in range(k))
        r2 = sum(c(p["tau_x"], (k - j) * period) for j in range(k))
        ltd.append(o1 * (p["a2_minus"] + p["a3_minus"] * r2))
    return ltp, ltd


def pairing_train(period, lag, n, start=5):
    pre = {start + j * period: [0] for j in range(n)}
    post = {start + j * period + lag: [0] for j in range(n)}
    return pre, post, start + (n - 1) * period + lag + 1


class TestFrequencyDependence:
    @pytest.mark.parametrize("period", [100, 50, 20, 10])
    def test_total_change_matches_the_closed_form(self, period):
        n, lag = 6, 3
        pre, post, steps = pairing_train(period, lag, n)
        got = run_triplet(pre, post, steps).item()
        ltp, ltd = expected_total(period, lag, n, P)
        assert got == pytest.approx(sum(ltp) - sum(ltd), abs=1e-12)

    def test_potentiation_per_pairing_grows_with_frequency_only_with_triplets(self):
        n, lag = 6, 3
        potentiation = {}
        for a3 in (P["a3_plus"], 0.0):
            params = {**P, "a3_plus": a3, "a2_minus": 0.0, "a3_minus": 0.0}
            for period in (100, 20):
                pre, post, steps = pairing_train(period, lag, n)
                got = run_triplet(pre, post, steps, params).item() / n
                ltp, _ = expected_total(period, lag, n, params)
                assert got == pytest.approx(sum(ltp) / n, abs=1e-12)
                potentiation[a3, period] = got
        triplet_low, triplet_high = potentiation[P["a3_plus"], 100], potentiation[P["a3_plus"], 20]
        pair_low, pair_high = potentiation[0.0, 100], potentiation[0.0, 20]
        assert triplet_high > triplet_low  # the triplet rule potentiates more at 5x the rate
        # the pair rule gains only the small pile-up of r1; the triplet gain is far larger
        assert triplet_high - triplet_low > 3 * (pair_high - pair_low)

    def test_pair_rule_depression_dominates_at_low_frequency(self):
        n, lag, period = 6, 3, 100
        params = {**P, "a3_plus": 0.0, "a3_minus": 0.0}
        pre, post, steps = pairing_train(period, lag, n)
        got = run_triplet(pre, post, steps, params).item()
        ltp, ltd = expected_total(period, lag, n, params)
        assert got == pytest.approx(sum(ltp) - sum(ltd), abs=1e-12)
        # a2_minus > a2_plus and the post-pre pairing in the train is nearly as close
        assert sum(ltd) > 0


NU_PRE, NU_POST = 0.0001, 0.01
DC = {
    "a2_plus": 0.0,
    "a3_plus": NU_POST,
    "a2_minus": NU_PRE,
    "a3_minus": 0.0,
    "tau_plus": 20.0,
    "tau_minus": 20.0,
    "tau_x": 20.0,
    "tau_y": 40.0,
}


class TestNearestDiehlCook:
    def test_ltp_is_nu_post_r1_o2_before_with_reset_traces(self):
        # two pre spikes (4 and 10) then a post at 10 + lag; an earlier post at 5
        lag = 6
        pre = {4: [0], 10: [0]}
        post = {5: [0], 10 + lag: [0]}
        got = run_triplet(pre, post, 10 + lag + 1, DC, interaction="nearest").item()
        r1 = c(20.0, lag)  # nearest: only the latest pre spike counts
        o2_before = c(40.0, lag + 5)  # o2 was reset to 1 at step 5
        ltp = NU_POST * r1 * o2_before
        # depression at pre spikes: step 4 sees o1 = 0, step 10 sees o1 = c^5
        ltd = NU_PRE * c(20.0, 5)
        assert got == pytest.approx(ltp - ltd, abs=1e-15)

    def test_all_to_all_would_accumulate_instead(self):
        lag = 6
        pre = {4: [0], 10: [0]}
        post = {5: [0], 10 + lag: [0]}
        nearest = run_triplet(pre, post, 10 + lag + 1, DC, interaction="nearest").item()
        every = run_triplet(pre, post, 10 + lag + 1, DC, interaction="all").item()
        # all-to-all r1 includes the older pre spike (step 4), so potentiation is larger
        extra = NU_POST * c(20.0, lag + 6) * c(40.0, lag + 5)
        assert every - nearest == pytest.approx(extra, abs=1e-15)


class TestBatchMean:
    def test_dense_is_the_mean_of_the_samples(self):
        g = torch.Generator().manual_seed(1)
        f64 = torch.float64
        shape = (4, 3), (4, 2)
        args = {
            "pre_spike": torch.rand(shape[0], generator=g) < 0.5,
            "post_spike": torch.rand(shape[1], generator=g) < 0.5,
            "r1": torch.rand(shape[0], generator=g, dtype=f64),
            "o1": torch.rand(shape[1], generator=g, dtype=f64),
            "r2_before": torch.rand(shape[0], generator=g, dtype=f64),
            "o2_before": torch.rand(shape[1], generator=g, dtype=f64),
        }
        rates = {k: P[k] for k in ("a2_plus", "a3_plus", "a2_minus", "a3_minus")}
        batched = triplet_dense(**args, **rates)
        per_sample = torch.stack(
            [triplet_dense(**{k: v[i] for k, v in args.items()}, **rates) for i in range(4)]
        )
        torch.testing.assert_close(batched, per_sample.mean(0), rtol=1e-12, atol=1e-15)
        assert batched.shape == (3, 2)

    def test_one_to_one_is_the_mean_of_the_samples(self):
        g = torch.Generator().manual_seed(2)
        f64 = torch.float64
        args = {
            "pre_spike": torch.rand(5, 3, generator=g) < 0.5,
            "post_spike": torch.rand(5, 3, generator=g) < 0.5,
            "r1": torch.rand(5, 3, generator=g, dtype=f64),
            "o1": torch.rand(5, 3, generator=g, dtype=f64),
            "r2_before": torch.rand(5, 3, generator=g, dtype=f64),
            "o2_before": torch.rand(5, 3, generator=g, dtype=f64),
        }
        rates = {k: P[k] for k in ("a2_plus", "a3_plus", "a2_minus", "a3_minus")}
        batched = triplet_one_to_one(**args, **rates)
        per_sample = torch.stack(
            [triplet_one_to_one(**{k: v[i] for k, v in args.items()}, **rates) for i in range(5)]
        )
        torch.testing.assert_close(batched, per_sample.mean(0), rtol=1e-12, atol=1e-15)

    def test_batched_network_with_identical_samples_matches_unbatched(self):
        pre, post = {10: [0], 14: [0]}, {5: [0], 12: [0]}
        single = run_triplet(pre, post, 20)
        net = Network(dtype=torch.float64, batch_size=3)
        src = NeuronGroup(net, 1, [ScriptedSpikes(pre), Axon()])
        dst = NeuronGroup(net, 1, [ScriptedSpikes(post), Axon()])
        syn = SynapseGroup(
            net,
            src,
            dst,
            [
                WeightInit(weights=torch.full((1, 1), 0.5, dtype=torch.float64)),
                DenseInput(),
                SpikeGather(),
                TripletSTDP(**P),
            ],
        )
        net.run(20)
        torch.testing.assert_close(syn.weights - 0.5, single, rtol=1e-12, atol=1e-15)


def test_one_to_one_matches_dense_diagonal():
    pre, post = {10: [0, 1], 14: [1]}, {5: [0], 12: [0, 1]}
    dense = run_triplet(pre, post, 20, n_src=2, n_dst=2)
    net = Network(dtype=torch.float64)
    src = NeuronGroup(net, 2, [ScriptedSpikes(pre), Axon()])
    dst = NeuronGroup(net, 2, [ScriptedSpikes(post), Axon()])
    syn = SynapseGroup(
        net,
        src,
        dst,
        [
            WeightInit(weights=torch.full((2,), 0.5, dtype=torch.float64), shape=(2,)),
            OneToOneInput(),
            SpikeGather(),
            TripletSTDP(**P),
        ],
    )
    net.run(20)
    torch.testing.assert_close(syn.weights - 0.5, dense.diagonal(), rtol=1e-12, atol=1e-15)
