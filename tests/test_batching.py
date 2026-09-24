"""Batched simulation: every sample behaves like its own unbatched run."""

import itertools

import pytest
import torch

from neurosush.core.buffers import ArrivalBuffer, HistoryBuffer
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import KWTA, kwta_losers
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.homeostasis import ActivityHomeostasis
from neurosush.neurons.inputs import SpikeInput
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses import currents, plasticity
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces

G = torch.Generator().manual_seed(0)


def spikes(*shape, p=0.3):
    return torch.rand(*shape, generator=G) < p


class TestState:
    def test_batch_size_shapes_state_but_not_parameters(self):
        net = Network(batch_size=3)
        group = NeuronGroup(net, 4)
        assert group.state_shape == (3, 4)
        assert group.state().shape == (3, 4)
        assert group.vector().shape == (4,)
        assert group.rand().shape == (3, 4)

    def test_unbatched_is_the_default(self):
        group = NeuronGroup(Network(), 4)
        assert group.state_shape == (4,)

    @pytest.mark.parametrize("batch_size", [0, -1, True])
    def test_invalid_batch_size(self, batch_size):
        with pytest.raises(ValueError, match="batch_size"):
            Network(batch_size=batch_size)


class TestBuffers:
    def test_history_is_per_sample_with_shared_delays(self):
        buf = HistoryBuffer(depth=2, size=2, batch=2)
        buf.push(torch.tensor([[True, False], [False, True]]))
        buf.push(torch.tensor([[False, False], [False, False]]))
        assert buf.read(torch.tensor([1, 0])).tolist() == [[True, False], [False, False]]

    def test_arrival_is_per_sample(self):
        buf = ArrivalBuffer(depth=2, size=1, batch=3)
        buf.add(torch.tensor([[1.0], [2.0], [3.0]]), torch.tensor([1]))
        buf.advance()
        assert buf.current().tolist() == [[1.0], [2.0], [3.0]]

    def test_value_shape_must_include_the_batch(self):
        with pytest.raises(ValueError, match="shape"):
            HistoryBuffer(depth=1, size=2, batch=2).push(torch.tensor([True, False]))


def per_sample(fn, batched_args, sample_count):
    """Stack fn applied to each sample of every batched argument."""
    return torch.stack([fn(*(a[i] for a in batched_args)) for i in range(sample_count)])


class TestCurrents:
    def test_dense(self):
        s, w = spikes(4, 5), torch.rand(5, 3, generator=G)
        assert torch.allclose(
            currents.dense_current(s, w), per_sample(lambda x: currents.dense_current(x, w), [s], 4)
        )

    def test_sparse(self):
        s = spikes(3, 4)
        src, dst, values = (
            torch.tensor([0, 1, 3]),
            torch.tensor([2, 0, 2]),
            torch.rand(3, generator=G),
        )
        batched = currents.sparse_current(s, values, src, dst, 3)
        expected = per_sample(lambda x: currents.sparse_current(x, values, src, dst, 3), [s], 3)
        assert torch.allclose(batched, expected)

    def test_conv_local_lateral_pool(self):
        s = spikes(2, 2 * 4 * 4)
        conv_w = torch.rand(3, 2, 2, 2, generator=G)
        local_w = torch.rand(3, 9, 8, generator=G)
        lateral_w = torch.rand(1, 1, 1, 3, 3, generator=G)
        cases = [
            lambda x: currents.conv2d_current(x, conv_w, src_shape=(2, 4, 4), stride=1, padding=0),
            lambda x: currents.local2d_current(
                x, local_w, src_shape=(2, 4, 4), kernel_size=(2, 2), stride=1, padding=0
            ),
            lambda x: currents.lateral_current(x, lateral_w, shape=(2, 4, 4)),
            lambda x: currents.avg_pool_current(x, src_shape=(2, 4, 4), out_size=(2, 2)),
        ]
        for fn in cases:
            assert torch.allclose(fn(s), per_sample(fn, [s], 2), atol=1e-6)


class TestLearningKernels:
    def activity(self, n_pre, n_post, batch):
        return {
            "pre_spike": spikes(batch, n_pre),
            "pre_trace": torch.rand(batch, n_pre, generator=G),
            "post_spike": spikes(batch, n_post),
            "post_trace": torch.rand(batch, n_post, generator=G),
        }

    def test_dense_stdp_is_the_mean_of_samples(self):
        act = self.activity(5, 4, 3)
        batched = plasticity.stdp_dense(**act, a_plus=0.2, a_minus=0.1)
        samples = [
            plasticity.stdp_dense(**{k: v[i] for k, v in act.items()}, a_plus=0.2, a_minus=0.1)
            for i in range(3)
        ]
        assert torch.allclose(batched, torch.stack(samples).mean(0))

    def test_conv_and_local_stdp_are_the_mean_of_samples(self):
        geometry = {
            "src_shape": (1, 3, 3),
            "dst_shape": (2, 2, 2),
            "kernel_size": (2, 2),
            "stride": (1, 1),
            "padding": (0, 0),
        }
        act = self.activity(9, 8, 2)
        for fn in (plasticity.stdp_conv2d, plasticity.stdp_local2d):
            batched = fn(**act, a_plus=0.3, a_minus=0.1, **geometry)
            samples = [
                fn(**{k: v[i] for k, v in act.items()}, a_plus=0.3, a_minus=0.1, **geometry)
                for i in range(2)
            ]
            assert torch.allclose(batched, torch.stack(samples).mean(0), atol=1e-6)

    def test_one_to_one_and_istdp(self):
        act = self.activity(4, 4, 3)
        for fn, kwargs in (
            (plasticity.stdp_one_to_one, {"a_plus": 0.2, "a_minus": 0.1}),
            (plasticity.istdp_dense, {"lr": 0.1, "alpha": 0.2}),
            (plasticity.istdp_one_to_one, {"lr": 0.1, "alpha": 0.2}),
        ):
            batched = fn(**act, **kwargs)
            samples = [fn(**{k: v[i] for k, v in act.items()}, **kwargs) for i in range(3)]
            assert torch.allclose(batched, torch.stack(samples).mean(0), atol=1e-6)


def test_kwta_competes_within_each_sample():
    v = torch.tensor([[3.0, 1.0, 2.0], [0.5, 4.0, 5.0]])
    assert kwta_losers(v, 0.0, k=1).tolist() == [[False, True, True], [True, True, False]]


def network(batch_size, frames, learn=True):
    net = Network(seed=1, batch_size=batch_size)
    src = NeuronGroup(net, 6, [SpikeInput(frames), Axon()])
    dst = NeuronGroup(
        net,
        3,
        [
            DendriteStructure(),
            DendriteIntegration(),
            LIF(tau=5.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            KWTA(k=2),
            Fire(),
            ActivityHomeostasis(target_spikes=2, window=10, rate=0.5),
            Axon(),
        ],
    )
    rules = [STDP(a_plus=0.05, a_minus=0.02, bound="soft")] if learn else []
    syn = SynapseGroup(
        net,
        src,
        dst,
        [
            WeightInit(mode="uniform"),
            DenseInput(coef=15.0),
            SpikeGather(),
            Traces(tau_pre=5.0),
            *rules,
        ],
    )
    return net, dst, syn


class TestNetwork:
    def test_identical_samples_learn_exactly_like_one(self):
        trains = spikes(40, 6, p=0.4)
        net1, dst1, syn1 = network(None, iter(trains))
        net4, dst4, syn4 = network(4, iter(trains.unsqueeze(1).expand(40, 4, 6)))
        net1.run(40)
        net4.run(40)
        assert torch.allclose(syn4.weights, syn1.weights, atol=1e-6)
        assert torch.allclose(dst4.threshold, dst1.threshold)
        assert torch.equal(dst4.spikes, dst1.spikes.expand(4, 3))

    def test_samples_do_not_interact_without_learning(self):
        trains = [spikes(30, 6, p=0.4) for _ in range(3)]
        runs = []
        for train in trains:
            net, dst, _ = network(None, iter(train), learn=False)
            record = []
            for _ in range(30):
                net.step()
                record.append(dst.v.clone())
            runs.append(torch.stack(record))
        net, dst, _ = network(3, iter(torch.stack(trains, 1)), learn=False)
        record = []
        for _ in range(30):
            net.step()
            record.append(dst.v.clone())
        batched = torch.stack(record)
        # homeostasis couples samples through the shared threshold only after a window
        for i in range(3):
            assert torch.allclose(batched[:10, i], runs[i][:10])

    def test_frames_must_match_the_batch(self):
        net, _, _ = network(2, itertools.repeat(torch.zeros(6, dtype=torch.bool)))
        with pytest.raises(ValueError, match="state shape"):
            net.step()
