"""Traces, STDP and homeostasis against their closed forms.

A trace decays by ``c = 1 - dt / tau`` per step and gains one per spike, so its response to a
spike ``k`` steps ago is exactly ``c^k`` (``~ exp(-k dt / tau)``). Pair-based STDP pairs every
postsynaptic spike with the presynaptic trace (potentiation, ``a_plus``) and every
presynaptic spike with the postsynaptic trace (depression, ``a_minus``).
"""

import math

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.homeostasis import ActivityHomeostasis
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import ISTDP, STDP
from neurosush.synapses.traces import SpikeGather, Traces, trace_step

from .common import BernoulliSpikes, ConstantCurrent, ScriptedSpikes

A_PLUS, A_MINUS, TAU_PLUS, TAU_MINUS = 0.01, 0.012, 20.0, 10.0


class TestTraces:
    def test_impulse_response_is_geometric(self):
        trace, c = torch.zeros(1), 1 - 0.5 / 8.0
        values = []
        for step in range(30):
            trace = trace_step(trace, torch.tensor([step == 0]), tau=8.0, dt=0.5)
            values.append(trace.item())
        assert values == pytest.approx([c**k for k in range(30)], rel=1e-6)

    def test_traces_are_linear_in_the_spike_train(self):
        # the trace of a spike train is the sum of the impulse responses of its spikes
        g = torch.Generator().manual_seed(0)
        spikes = torch.rand(200, 5, generator=g) < 0.1
        trace = torch.zeros(5, dtype=torch.float64)
        for s in spikes:
            trace = trace_step(trace, s, tau=6.0, dt=1.0)
        c = 1 - 1 / 6.0
        ages = torch.arange(199, -1, -1, dtype=torch.float64).unsqueeze(1)
        expected = (spikes.double() * c**ages).sum(0)
        torch.testing.assert_close(trace, expected)


def stdp_pair(pre_step, post_step, dt=1.0):
    """Weight change of one synapse after one pre and one post spike."""
    net = Network(dt=dt, dtype=torch.float64)
    pre = NeuronGroup(net, 1, [ScriptedSpikes({pre_step: [0]}), Axon()])
    post = NeuronGroup(net, 1, [ScriptedSpikes({post_step: [0]}), Axon()])
    syn = SynapseGroup(
        net,
        pre,
        post,
        [
            WeightInit(weights=torch.tensor([[0.5]])),
            DenseInput(),
            SpikeGather(),
            Traces(tau_pre=TAU_PLUS, tau_post=TAU_MINUS),
            STDP(a_plus=A_PLUS, a_minus=A_MINUS),
        ],
    )
    net.run(max(pre_step, post_step) + 1)
    return syn.weights.item() - 0.5


class TestSTDPWindow:
    @pytest.mark.parametrize("lag", [1, 2, 5, 10, 25, 60])
    def test_pre_before_post_potentiates_by_the_exponential_window(self, lag):
        c = 1 - 1.0 / TAU_PLUS
        assert stdp_pair(10, 10 + lag) == pytest.approx(A_PLUS * c**lag, rel=1e-9)

    @pytest.mark.parametrize("lag", [1, 2, 5, 10, 25, 60])
    def test_post_before_pre_depresses_by_the_exponential_window(self, lag):
        c = 1 - 1.0 / TAU_MINUS
        assert stdp_pair(10 + lag, 10) == pytest.approx(-A_MINUS * c**lag, rel=1e-9)

    def test_coincident_spikes_get_both_terms(self):
        assert stdp_pair(10, 10) == pytest.approx(A_PLUS - A_MINUS, rel=1e-9)

    def test_window_approaches_the_continuous_exponential_as_dt_shrinks(self):
        # a 5 ms lag: the discrete window (1 - dt/tau)^(5/dt) tends to exp(-5 / tau)
        errors = []
        for dt in (1.0, 0.5, 0.25):
            lag_steps = round(5.0 / dt)
            change = stdp_pair(10, 10 + lag_steps, dt=dt)
            errors.append(abs(change - A_PLUS * math.exp(-5.0 / TAU_PLUS)))
        assert errors[0] > errors[1] > errors[2]
        assert errors[1] / errors[2] == pytest.approx(2.0, rel=0.1)  # first order in dt


class TestSTDPDrift:
    def test_uncorrelated_poisson_trains_drift_as_predicted(self):
        # independent Bernoulli spikes with probabilities p_pre, p_post: the expected trace
        # after n steps is p (1 - c^n) / (1 - c), so after N steps every weight changes by
        # E[dw] = p_pre p_post sum_n [a_plus (1 - c+^n)/(1 - c+) - a_minus (1 - c-^n)/(1 - c-)]
        p_pre, p_post, steps = 0.05, 0.08, 1000
        net = Network(seed=0, dtype=torch.float64)
        pre = NeuronGroup(net, 40, [BernoulliSpikes(p_pre), Axon()])
        post = NeuronGroup(net, 40, [BernoulliSpikes(p_post), Axon()])
        syn = SynapseGroup(
            net,
            pre,
            post,
            [
                WeightInit(mode=0.5),
                DenseInput(),
                SpikeGather(),
                Traces(tau_pre=TAU_PLUS, tau_post=TAU_MINUS),
                STDP(a_plus=A_PLUS, a_minus=A_MINUS),
            ],
        )
        net.run(steps)
        c_plus, c_minus = 1 - 1 / TAU_PLUS, 1 - 1 / TAU_MINUS
        expected = (
            p_pre
            * p_post
            * sum(
                A_PLUS * (1 - c_plus**n) / (1 - c_plus) - A_MINUS * (1 - c_minus**n) / (1 - c_minus)
                for n in range(1, steps + 1)
            )
        )
        change = syn.weights - 0.5
        # rows share a presynaptic train and columns a postsynaptic one, so the 40 x 40
        # changes are not independent; the row means are (nearly) independent samples
        rows = change.mean(1)
        error = rows.std().item() / math.sqrt(len(rows))
        assert abs(change.mean().item() - expected) < 5 * error
        assert expected > 0  # a_plus tau_plus = 0.2 exceeds a_minus tau_minus = 0.12


class TestHomeostasis:
    def test_firing_converges_to_the_target_count_per_window(self):
        # the counter ends a window at spikes (1 + penalty) - window * penalty, which is zero
        # exactly at the target, so the threshold stops moving there
        target, window = 4, 50
        net = Network(dtype=torch.float64)
        group = NeuronGroup(
            net,
            3,
            [
                ConstantCurrent(30.0),
                LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
                Fire(),
                ActivityHomeostasis(target_spikes=target, window=window, rate=0.05),
            ],
        )
        counts = []
        for _ in range(80):
            spikes = 0
            for _ in range(window):
                net.step()
                spikes += int(group.spikes[0])
            counts.append(spikes)
        assert counts[0] >= 2 * target
        assert all(abs(count - target) <= 1 for count in counts[-20:])


class TestInhibitorySTDP:
    """Vogels et al. (2011): inhibitory plasticity balances excitation so that the
    postsynaptic rate settles at ``rho``, from above (inhibition grows) or from below
    (inhibition shrinks)."""

    def rates(self, rho, initial_inhibition, blocks=12):
        net = Network(seed=0)
        exc = NeuronGroup(net, 80, [BernoulliSpikes(0.02), Axon()])
        inh = NeuronGroup(net, 20, [BernoulliSpikes(0.05), Axon()], inhibitory=True)
        post = NeuronGroup(
            net,
            5,
            [
                DendriteStructure(),
                DendriteIntegration(),
                LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
                Fire(),
                Axon(),
            ],
        )
        SynapseGroup(net, exc, post, [WeightInit(mode=1.0), DenseInput(coef=4.0), SpikeGather()])
        SynapseGroup(
            net,
            inh,
            post,
            [
                WeightInit(mode=initial_inhibition),
                DenseInput(coef=4.0),
                SpikeGather(),
                Traces(tau_pre=20.0),
                ISTDP(lr=0.02, rho=rho),
            ],
        )
        rates = []
        for _ in range(blocks):
            spikes = 0
            for _ in range(1000):
                net.step()
                spikes += int(post.spikes.sum())
            rates.append(spikes / (1000 * post.size))
        return rates

    @pytest.mark.parametrize(("rho", "initial_inhibition"), [(0.005, 0.0), (0.02, 0.3)])
    def test_rate_settles_at_the_target(self, rho, initial_inhibition):
        rates = self.rates(rho, initial_inhibition)
        settled = sum(rates[-6:]) / 6
        assert settled == pytest.approx(rho, rel=0.25)
        # it had to move there: the first block is far from the target
        assert abs(rates[0] - rho) > 0.3 * rho
