"""Reward-modulated STDP: the three-factor rule in closed form, and distal reward.

With eligibility ``c_n = c_{n-1} (1 - dt/tau_c) + stdp_n`` and ``w += dt d_n c_n``, one
pairing that sets ``c = s`` at step ``q`` and one reward ``P`` at step ``r > q`` (so
``d_n = P dt (1 - dt/tau_d)^(n - r)`` from step ``r`` on) change the weight by the geometric
series ``dt^2 P s e_c^(r - q) (1 - (e_c e_d)^K) / (1 - e_c e_d)`` over ``K`` steps, where
``e_c = 1 - dt/tau_c`` and ``e_d = 1 - dt/tau_d``.
"""

import random

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.modulation import Dopamine, Payoff
from neurosush.neurons.axon import Axon
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import RSTDP
from neurosush.synapses.traces import SpikeGather, Traces

from .common import BernoulliSpikes, ScriptedSpikes

TAU_TRACE, TAU_C, TAU_D, A_PLUS, A_MINUS = 10.0, 50.0, 8.0, 0.2, 0.3


def rewarded_pair(pre_step, post_step, reward_step, reward, steps, dt=1.0):
    net = Network(
        dt=dt,
        dtype=torch.float64,
        behaviors=[
            Payoff(lambda n: reward / dt if n.iteration == reward_step else 0.0),
            Dopamine(tau=TAU_D),
        ],
    )
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
            Traces(tau_pre=TAU_TRACE),
            RSTDP(tau_c=TAU_C, a_plus=A_PLUS, a_minus=A_MINUS),
        ],
    )
    net.run(steps)
    return syn.weights.item() - 0.5


def closed_form(pairing, pairing_step, reward_step, reward, steps, dt=1.0):
    e_c, e_d = 1 - dt / TAU_C, 1 - dt / TAU_D
    k = steps - reward_step + 1  # steps from the reward on
    # the payoff reward / dt raises dopamine by dt * reward / dt = reward at the reward step
    return (
        dt
        * reward
        * pairing
        * e_c ** (reward_step - pairing_step)
        * (1 - (e_c * e_d) ** k)
        / (1 - e_c * e_d)
    )


class TestThreeFactorRule:
    @pytest.mark.parametrize(("pre", "post"), [(10, 14), (10, 25), (14, 10)])
    def test_single_pairing_and_reward(self, pre, post):
        e_trace = 1 - 1 / TAU_TRACE
        lag = post - pre
        pairing = A_PLUS * e_trace**lag if lag > 0 else -A_MINUS * e_trace**-lag
        reward_step = max(pre, post) + 30
        got = rewarded_pair(pre, post, reward_step, reward=1.5, steps=200)
        assert got == pytest.approx(
            closed_form(pairing, max(pre, post), reward_step, 1.5, 200), rel=1e-9
        )

    def test_no_reward_no_change(self):
        assert rewarded_pair(10, 14, reward_step=10**6, reward=1.0, steps=100) == 0.0

    def test_credit_decays_with_the_reward_delay(self):
        # the later the reward, the smaller the change, by exactly e_c per step of delay
        early = rewarded_pair(10, 14, 40, reward=1.0, steps=400)
        late = rewarded_pair(10, 14, 60, reward=1.0, steps=400)
        e_c = 1 - 1 / TAU_C
        # both runs stop at step 400, so the tails differ slightly; compare the closed forms
        assert late / early == pytest.approx(
            closed_form(1.0, 14, 60, 1.0, 400) / closed_form(1.0, 14, 40, 1.0, 400), rel=1e-9
        )
        assert late / early == pytest.approx(e_c**20, rel=1e-3)


class TestDistalReward:
    """Izhikevich (2007): a reward that arrives 50-150 steps after one particular pre-post
    pairing is credited to that synapse among all the synapses that saw the same spikes.

    Paired design: the control run has identical spikes (same seed) and the same number of
    rewards at random times, so the difference between the runs isolates the credit given
    by contingent reward."""

    def run(self, reward_times=None, seed=0, steps=12000, n=10):
        rng = random.Random(seed)
        pending, given, last_pre = [], [], [-(10**9)]

        def payoff(net):
            t = net.iteration
            pre, post = net.groups[0].spikes, net.groups[1].spikes  # the previous step's
            if bool(pre[0]):
                last_pre[0] = t
            if reward_times is None and bool(post[0]) and 0 < t - last_pre[0] <= 10:
                pending.append(t + rng.randint(50, 150))
            if reward_times is not None and t in reward_times:
                pending.append(t)
            due = pending.count(t)
            pending[:] = [x for x in pending if x != t]
            given.extend([t] * due)
            return 0.05 * due

        rates = torch.full((n,), 0.01)
        rates[0] = 0.03  # the rewarded pair fires more, so pairings happen often enough
        net = Network(seed=seed, behaviors=[Payoff(payoff), Dopamine(tau=20.0)])
        pre = NeuronGroup(net, n, [BernoulliSpikes(rates), Axon()])
        post = NeuronGroup(net, n, [BernoulliSpikes(rates), Axon()])
        syn = SynapseGroup(
            net,
            pre,
            post,
            [
                WeightInit(mode=0.5),
                DenseInput(),
                SpikeGather(),
                Traces(tau_pre=10.0),
                RSTDP(tau_c=200.0, a_plus=0.1, a_minus=0.1),
            ],
        )
        net.run(steps)
        return (syn.weights - 0.5).flatten(), given

    def test_contingent_reward_is_credited_to_its_synapse(self):
        contingent, given = self.run()
        times = set(random.Random(100).sample(range(200, 12000), len(given)))
        control, _ = self.run(reward_times=times)
        credit = contingent - control
        others = credit[1:]
        z = (credit[0] - others.mean()) / others.std()
        assert len(given) > 40
        assert z.item() > 3  # measured 3.9 to 5.7 over three seeds
        assert abs(others.mean().item()) < others.std().item()  # no net drift elsewhere
