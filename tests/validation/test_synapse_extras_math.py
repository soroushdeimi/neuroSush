"""Nearest-spike traces and pairing, fixed in-degree, delta coefficient and OR pooling.

Each case is checked against a closed form or a direct reimplementation. A trace decays by
``c = 1 - dt / tau`` per step, so a spike ``k`` steps ago contributes ``c^k`` when the trace
adds up (``"all"``) and only the latest spike counts when it is set to one (``"nearest"``).
"""

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses.currents import DenseInput, MaxPool2dInput, OneToOneInput, delta_coef
from neurosush.synapses.init import WeightInit, fixed_in_degree
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces

from .common import ScriptedSpikes

A_PLUS, A_MINUS, TAU_PLUS, TAU_MINUS = 0.01, 0.012, 20.0, 10.0
C_PLUS, C_MINUS = 1 - 1 / TAU_PLUS, 1 - 1 / TAU_MINUS


def run_stdp(pre_steps, post_steps, *, interaction, pairing="all", connectivity="dense"):
    """Weight change of one synapse for the spikes at the given steps."""
    net = Network(dtype=torch.float64)
    pre = NeuronGroup(net, 1, [ScriptedSpikes({s: [0] for s in pre_steps}), Axon()])
    post = NeuronGroup(net, 1, [ScriptedSpikes({s: [0] for s in post_steps}), Axon()])
    dense = connectivity == "dense"
    syn = SynapseGroup(
        net,
        pre,
        post,
        [
            WeightInit(
                weights=torch.tensor([[0.5]] if dense else [0.5]), shape=None if dense else (1,)
            ),
            DenseInput() if dense else OneToOneInput(),
            SpikeGather(),
            Traces(tau_pre=TAU_PLUS, tau_post=TAU_MINUS, interaction=interaction),
            STDP(a_plus=A_PLUS, a_minus=A_MINUS, pairing=pairing),
        ],
    )
    net.run(max(*pre_steps, *post_steps) + 3)
    return syn.weights.item() - 0.5


class TestNearestTraces:
    def test_potentiation_uses_the_latest_presynaptic_spike(self):
        # pre at 10 and 12, post at 15: all adds c^5 + c^3, nearest keeps c^3
        all_ = run_stdp([10, 12], [15], interaction="all")
        nearest = run_stdp([10, 12], [15], interaction="nearest")
        assert all_ == pytest.approx(A_PLUS * (C_PLUS**5 + C_PLUS**3), rel=1e-9)
        assert nearest == pytest.approx(A_PLUS * C_PLUS**3, rel=1e-9)

    def test_depression_uses_the_latest_postsynaptic_spike(self):
        # post at 10 and 12, pre at 15
        all_ = run_stdp([15], [10, 12], interaction="all")
        nearest = run_stdp([15], [10, 12], interaction="nearest")
        assert all_ == pytest.approx(-A_MINUS * (C_MINUS**5 + C_MINUS**3), rel=1e-9)
        assert nearest == pytest.approx(-A_MINUS * C_MINUS**3, rel=1e-9)

    def test_trace_is_set_to_scale_and_decays(self):
        net = Network(dtype=torch.float64)
        pre = NeuronGroup(net, 1, [ScriptedSpikes({2: [0], 3: [0]}), Axon()])
        post = NeuronGroup(net, 1, [ScriptedSpikes({}), Axon()])
        syn = SynapseGroup(
            net,
            pre,
            post,
            [SpikeGather(), Traces(tau_pre=4.0, scale=2.0, interaction="nearest")],
        )
        net.initialize()
        seen = []
        for _ in range(6):
            net.run(1)
            seen.append(syn.pre_trace.item())
        # spikes at steps 2 and 3 (seen at indices 1, 2): 2, 2, then decay by 0.75
        assert seen[1:5] == pytest.approx([2.0, 2.0, 1.5, 1.125], rel=1e-12)

    def test_single_spike_pairs_agree_with_all(self):
        assert run_stdp([10], [14], interaction="nearest") == pytest.approx(
            run_stdp([10], [14], interaction="all"), rel=1e-12
        )

    def test_invalid_interaction(self):
        with pytest.raises(ValueError, match="interaction"):
            Traces(tau_pre=5.0, interaction="first")


@pytest.mark.parametrize("connectivity", ["dense", "one_to_one"])
class TestNearestPairing:
    def test_a_postsynaptic_spike_depresses_only_the_next_presynaptic_spike(self, connectivity):
        # post at 10, pre at 12 and 14: the pair (10, 14) is consumed by (10, 12)
        kw = {"interaction": "nearest", "connectivity": connectivity}
        nearest = run_stdp([12, 14], [10], pairing="nearest", **kw)
        all_ = run_stdp([12, 14], [10], pairing="all", **kw)
        assert nearest == pytest.approx(-A_MINUS * C_MINUS**2, rel=1e-9)
        assert all_ == pytest.approx(-A_MINUS * (C_MINUS**2 + C_MINUS**4), rel=1e-9)

    def test_nothing_is_armed_before_the_first_postsynaptic_spike(self, connectivity):
        # pre at 5 and 7, post at 10, pre at 12: potentiation from the latest pre spike (7),
        # depression from the pair (10, 12) only
        got = run_stdp(
            [5, 7, 12],
            [10],
            interaction="nearest",
            pairing="nearest",
            connectivity=connectivity,
        )
        assert got == pytest.approx(A_PLUS * C_PLUS**3 - A_MINUS * C_MINUS**2, rel=1e-9)

    def test_coincident_spikes_potentiate_and_arm_for_the_next_pre_spike(self, connectivity):
        # pre and post at 10 (potentiation 1, no armed depression), pre again at 12
        got = run_stdp(
            [10, 12],
            [10],
            interaction="nearest",
            pairing="nearest",
            connectivity=connectivity,
        )
        assert got == pytest.approx(A_PLUS - A_MINUS * C_MINUS**2, rel=1e-9)

    def test_a_second_postsynaptic_spike_rearms(self, connectivity):
        # post 10, pre 12 (armed), pre 14 (not), post 16, pre 18 (armed again)
        got = run_stdp(
            [12, 14, 18],
            [10, 16],
            interaction="nearest",
            pairing="nearest",
            connectivity=connectivity,
        )
        ltp = A_PLUS * C_PLUS**2  # post 16 sees the latest pre spike (14); post 10 sees none
        ltd = A_MINUS * (C_MINUS**2 + C_MINUS**2)  # (10, 12) and (16, 18)
        assert got == pytest.approx(ltp - ltd, rel=1e-9)


def reference_nearest_pair(pre, post, a_plus, a_minus, tau_pre, tau_post, w0):
    """Direct loop of the nearest-pair rule with an explicit armed flag per pair."""
    steps, n_pre = pre.shape
    n_post = post.shape[1]
    w = w0.clone()
    pre_trace = torch.zeros(n_pre, dtype=w.dtype)
    post_trace = torch.zeros(n_post, dtype=w.dtype)
    armed = torch.zeros(n_pre, n_post, dtype=torch.bool)
    for t in range(steps):
        pre_trace = torch.where(pre[t], 1.0, pre_trace * (1 - 1 / tau_pre))
        post_trace = torch.where(post[t], 1.0, post_trace * (1 - 1 / tau_post))
        dw = torch.zeros_like(w)
        for i in range(n_pre):
            for j in range(n_post):
                if post[t, j]:
                    dw[i, j] += a_plus * pre_trace[i]
                if pre[t, i] and armed[i, j]:
                    dw[i, j] -= a_minus * post_trace[j]
        w += dw
        for i in range(n_pre):
            for j in range(n_post):
                if pre[t, i]:
                    armed[i, j] = False
                if post[t, j]:
                    armed[i, j] = True
    return w


def test_nearest_pairing_matches_a_direct_implementation_on_random_spikes():
    g = torch.Generator().manual_seed(3)
    steps, n_pre, n_post = 120, 4, 3
    pre = torch.rand(steps, n_pre, generator=g) < 0.15
    post = torch.rand(steps, n_post, generator=g) < 0.1
    w0 = torch.full((n_pre, n_post), 0.5, dtype=torch.float64)
    expected = reference_nearest_pair(pre, post, A_PLUS, A_MINUS, TAU_PLUS, TAU_MINUS, w0)

    net = Network(dtype=torch.float64)
    # the scripts are 1-based steps
    pre_group = NeuronGroup(
        net,
        n_pre,
        [
            ScriptedSpikes({t + 1: pre[t].nonzero().flatten().tolist() for t in range(steps)}),
            Axon(),
        ],
    )
    post_group = NeuronGroup(
        net,
        n_post,
        [
            ScriptedSpikes({t + 1: post[t].nonzero().flatten().tolist() for t in range(steps)}),
            Axon(),
        ],
    )
    syn = SynapseGroup(
        net,
        pre_group,
        post_group,
        [
            WeightInit(weights=w0),
            DenseInput(),
            SpikeGather(),
            Traces(tau_pre=TAU_PLUS, tau_post=TAU_MINUS, interaction="nearest"),
            STDP(a_plus=A_PLUS, a_minus=A_MINUS, pairing="nearest"),
        ],
    )
    net.run(steps)
    torch.testing.assert_close(syn.weights, expected, rtol=1e-9, atol=1e-12)


def test_nearest_pairing_reset_state_disarms():
    net = Network(dtype=torch.float64)
    pre = NeuronGroup(net, 1, [ScriptedSpikes({12: [0]}), Axon()])
    post = NeuronGroup(net, 1, [ScriptedSpikes({10: [0]}), Axon()])
    syn = SynapseGroup(
        net,
        pre,
        post,
        [
            WeightInit(weights=torch.tensor([[0.5]])),
            DenseInput(),
            SpikeGather(),
            Traces(tau_pre=TAU_PLUS, tau_post=TAU_MINUS, interaction="nearest"),
            STDP(a_plus=A_PLUS, a_minus=A_MINUS, pairing="nearest"),
        ],
    )
    net.run(11)  # post spiked: armed
    net.reset_state()
    w = syn.weights.clone()
    net.run(3)  # steps 12-14 (iteration counter goes on): the pre spike finds nothing armed
    torch.testing.assert_close(syn.weights, w)


class IndependentScript(ScriptedSpikes):
    independent_ok = True


def test_nearest_pairing_with_independent_members_gives_each_the_same_change():
    net = Network(dtype=torch.float64, batch_size=2, independent=True)
    pre = NeuronGroup(net, 1, [IndependentScript({12: [0], 14: [0]}), Axon()])
    post = NeuronGroup(net, 1, [IndependentScript({10: [0]}), Axon()])
    syn = SynapseGroup(
        net,
        pre,
        post,
        [
            WeightInit(weights=torch.tensor([[0.5]])),
            DenseInput(),
            SpikeGather(),
            Traces(tau_pre=TAU_PLUS, tau_post=TAU_MINUS, interaction="nearest"),
            STDP(a_plus=A_PLUS, a_minus=A_MINUS, pairing="nearest"),
        ],
    )
    net.run(17)
    assert syn.weights.shape == (2, 1, 1)
    torch.testing.assert_close(
        syn.weights.flatten(), torch.full((2,), 0.5 - A_MINUS * C_MINUS**2, dtype=torch.float64)
    )


# ---------------------------------------------------------------------------------------------
# fixed in-degree
# ---------------------------------------------------------------------------------------------


def sparse_synapse(n_src, n_dst, k, seed=0, mode="ones"):
    net = Network(seed=seed)
    syn = SynapseGroup(
        net,
        NeuronGroup(net, n_src, [Axon()]),
        NeuronGroup(net, n_dst),
        [WeightInit(mode=mode, sparse=True, in_degree=k)],
    )
    net.initialize()
    return syn


class TestFixedInDegree:
    def test_every_destination_has_exactly_k_distinct_sources(self):
        n_src, n_dst, k = 30, 40, 7
        syn = sparse_synapse(n_src, n_dst, k)
        assert syn.src_idx.numel() == syn.dst_idx.numel() == syn.weights.numel() == n_dst * k
        pairs = syn.src_idx * n_dst + syn.dst_idx
        assert pairs.unique().numel() == n_dst * k  # no repeated connection
        assert torch.equal(syn.dst_idx.bincount(minlength=n_dst), torch.full((n_dst,), k))
        assert int(syn.src_idx.min()) >= 0
        assert int(syn.src_idx.max()) < n_src

    def test_all_sources_when_in_degree_equals_the_source_count(self):
        syn = sparse_synapse(5, 3, 5)
        for d in range(3):
            assert sorted(syn.src_idx[syn.dst_idx == d].tolist()) == [0, 1, 2, 3, 4]

    def test_sources_are_uniform(self):
        # every source appears n_dst * k / n_src times on average; the Pearson statistic of the
        # counts is about chi-square with n_src - 1 degrees of freedom (mean 49, sd 10)
        n_src, n_dst, k = 50, 4000, 5
        counts = sparse_synapse(n_src, n_dst, k).src_idx.bincount(minlength=n_src).double()
        expected = n_dst * k / n_src
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        assert chi2 < 49 + 6 * (2 * 49) ** 0.5
        assert counts.sum() == n_dst * k

    def test_weight_values_and_dtype(self):
        syn = sparse_synapse(6, 4, 2, mode=2.5)
        assert syn.weights.tolist() == [2.5] * 8
        assert syn.weights.dtype == torch.get_default_dtype()

    def test_reproducible_and_seed_dependent(self):
        a, b, c = (sparse_synapse(20, 10, 4, seed=s) for s in (1, 1, 2))
        assert torch.equal(a.src_idx, b.src_idx)
        assert not torch.equal(a.src_idx, c.src_idx)

    def test_chunked_draw_still_exact(self):
        # 2^23 sources: two destinations per chunk, so the draw is split in two
        src, dst = fixed_in_degree((1 << 23), 3, 4, generator=torch.Generator().manual_seed(0))
        assert torch.equal(dst.bincount(), torch.full((3,), 4))
        assert (src * 3 + dst).unique().numel() == 12

    def test_validation(self):
        with pytest.raises(ValueError, match="needs sparse"):
            WeightInit(mode="ones", in_degree=3)
        with pytest.raises(ValueError, match="exclusive"):
            WeightInit(mode="ones", sparse=True, in_degree=3, density=0.5)
        for bad in (0, -1, 2.5, True):
            with pytest.raises(ValueError, match="in_degree"):
                WeightInit(mode="ones", sparse=True, in_degree=bad)
        with pytest.raises(ValueError, match="in_degree"):
            sparse_synapse(5, 3, 6)  # more than there are sources

    def test_independent_networks_refuse_sparse(self):
        net = Network(batch_size=2, independent=True)
        SynapseGroup(
            net,
            NeuronGroup(net, 4, [Axon()]),
            NeuronGroup(net, 3),
            [WeightInit(mode="ones", sparse=True, in_degree=2)],
        )
        with pytest.raises(NotImplementedError, match="sparse"):
            net.initialize()


# ---------------------------------------------------------------------------------------------
# delta coefficient
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tau", "dt", "jump", "resistance"),
    [(20.0, 0.5, 0.5, 1.0), (10.0, 0.1, 0.2, 2.0), (5.0, 1.0, 1.0, 0.5)],
)
def test_delta_coef_makes_one_spike_jump_the_membrane_by_jump(tau, dt, jump, resistance):
    assert delta_coef(tau, dt, jump, resistance) == pytest.approx(jump * tau / (dt * resistance))
    net = Network(dt=dt, dtype=torch.float64)
    pre = NeuronGroup(net, 1, [ScriptedSpikes({3: [0]}), Axon()])
    post = NeuronGroup(
        net,
        1,
        [
            DendriteStructure(),
            DendriteIntegration(),
            LIF(tau=tau, threshold=100.0, v_reset=0.0, v_rest=0.0, resistance=resistance),
            Fire(),
            Axon(),
        ],
    )
    SynapseGroup(
        net,
        pre,
        post,
        [
            WeightInit(weights=torch.tensor([[1.0]])),
            DenseInput(coef=delta_coef(tau, dt, jump, resistance)),
            SpikeGather(),
        ],
    )
    net.initialize()
    v = []
    for _ in range(8):
        net.run(1)
        v.append(post.v.item())
    assert max(v) == pytest.approx(jump, rel=1e-12)  # exactly one jump from rest
    # afterwards the membrane leaks by (1 - dt / tau) per step
    peak = v.index(max(v))
    assert v[peak + 1] == pytest.approx(jump * (1 - dt / tau), rel=1e-12)


@pytest.mark.parametrize("args", [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.0)])
def test_delta_coef_validation(args):
    with pytest.raises(ValueError, match="positive"):
        delta_coef(*args)


# ---------------------------------------------------------------------------------------------
# OR pooling
# ---------------------------------------------------------------------------------------------


def direct_or_pool(spikes, kernel, stride):
    """Loop over every window of every depth plane and OR the spikes inside."""
    depth, height, width = spikes.shape
    out_h = (height - kernel[0]) // stride[0] + 1
    out_w = (width - kernel[1]) // stride[1] + 1
    out = torch.zeros(depth, out_h, out_w)
    for d in range(depth):
        for i in range(out_h):
            for j in range(out_w):
                window = spikes[
                    d,
                    i * stride[0] : i * stride[0] + kernel[0],
                    j * stride[1] : j * stride[1] + kernel[1],
                ]
                out[d, i, j] = float(window.any())
    return out


def pool_synapse(src_shape, dst_shape, pool, independent=False, batch_size=None):
    net = Network(batch_size=batch_size, independent=independent)
    src = NeuronGroup(net, src_shape, [Axon()])
    dst = NeuronGroup(net, dst_shape)
    syn = SynapseGroup(net, src, dst, [pool, SpikeGather()])
    net.initialize()
    return syn


@pytest.mark.parametrize(
    ("kernel", "stride", "dst_hw"),
    [(2, 2, (3, 3)), (3, 1, (4, 4)), ((2, 3), (2, 1), (3, 4)), (2, None, (3, 3))],
)
def test_max_pool_is_a_window_or(kernel, stride, dst_hw):
    g = torch.Generator().manual_seed(0)
    syn = pool_synapse((2, 6, 6), (2, *dst_hw), MaxPool2dInput(kernel, stride))
    pool = syn.behaviors[0]
    for density in (0.0, 0.05, 0.3, 1.0):
        spikes = torch.rand(2, 6, 6, generator=g) < density
        syn.pre_spike = spikes.flatten()
        pool.forward(syn)
        expected = direct_or_pool(spikes, pool.kernel_size, pool.stride)
        assert syn.I.tolist() == expected.flatten().tolist()


def test_max_pool_counts_a_window_once_and_applies_coef_and_sign():
    syn = pool_synapse((1, 4, 4), (1, 2, 2), MaxPool2dInput(2, coef=3.0))
    syn.pre_spike = torch.ones(16, dtype=torch.bool)
    syn.behaviors[0].forward(syn)
    assert syn.I.tolist() == [3.0] * 4  # four spikes per window, still one
    net = Network()
    src = NeuronGroup(net, (1, 4, 4), [Axon()], inhibitory=True)
    syn = SynapseGroup(net, src, NeuronGroup(net, (1, 2, 2)), [MaxPool2dInput(2), SpikeGather()])
    net.initialize()
    syn.pre_spike = torch.ones(16, dtype=torch.bool)
    syn.behaviors[0].forward(syn)
    assert syn.I.tolist() == [-1.0] * 4


def test_max_pool_batch_and_independent_members_pool_separately():
    g = torch.Generator().manual_seed(1)
    spikes = torch.rand(3, 1, 4, 4, generator=g) < 0.2
    for independent in (False, True):
        syn = pool_synapse((1, 4, 4), (1, 2, 2), MaxPool2dInput(2), independent, 3)
        syn.pre_spike = spikes.reshape(3, 16)
        syn.behaviors[0].forward(syn)
        for b in range(3):
            assert syn.I[b].tolist() == direct_or_pool(spikes[b], (2, 2), (2, 2)).flatten().tolist()


def test_max_pool_validation():
    with pytest.raises(ValueError, match="height"):
        pool_synapse((1, 4, 4), (1, 3, 3), MaxPool2dInput(2))  # stride 2 gives 2, not 3
    with pytest.raises(ValueError, match="depth"):
        pool_synapse((2, 4, 4), (1, 2, 2), MaxPool2dInput(2))
    with pytest.raises(ValueError, match="exceeds"):
        pool_synapse((1, 2, 2), (1, 1, 1), MaxPool2dInput(3))
    for bad in (0, -1, 1.5, (1, 2, 3)):
        with pytest.raises(ValueError, match="kernel_size"):
            MaxPool2dInput(bad)
    with pytest.raises(ValueError, match="stride"):
        MaxPool2dInput(2, 0)
