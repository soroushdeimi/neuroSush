import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.synapses.constraints import CurrentNormalization, WeightClip, WeightNormalization
from neurosush.synapses.currents import (
    Conv2dInput,
    DenseInput,
    Local2dInput,
    OneToOneInput,
    SparseInput,
)
from neurosush.synapses.init import DelayInit, WeightInit
from neurosush.synapses.traces import SpikeGather, Traces, trace_step


def test_trace_step_decays_then_adds():
    trace = torch.tensor([1.0, 0.0])
    out = trace_step(trace, torch.tensor([False, True]), tau=10.0, dt=1.0, scale=2.0)
    assert out.tolist() == pytest.approx([0.9, 2.0])
    assert trace.tolist() == [1.0, 0.0]


def test_single_spike_is_full_at_its_step():
    trace = torch.zeros(1)
    values = []
    for spike in (True, False, False):
        trace = trace_step(trace, torch.tensor([spike]), tau=10.0, dt=1.0)
        values.append(round(trace.item(), 4))
    assert values == [1.0, 0.9, 0.81]


def gather_net(src_delay=0, dst_delay=0, max_delay=3):
    from neurosush.core.behavior import Behavior
    from neurosush.core.order import Order

    class Spikes(Behavior):
        order = Order.FIRE

        def __init__(self, frames):
            self.frames = frames

        def initialize(self, group):
            group.spikes = group.vector(False, dtype=torch.bool)

        def forward(self, group):
            step = group.net.iteration - 1
            value = self.frames[step] if step < len(self.frames) else False
            group.spikes = group.vector(value, dtype=torch.bool)

    net = Network()
    src = NeuronGroup(net, 1, behaviors=[Spikes([True]), Axon(max_delay=max_delay)])
    dst = NeuronGroup(net, 1, behaviors=[Spikes([False, True]), Axon(max_delay=max_delay)])
    syn = SynapseGroup(
        net,
        src,
        dst,
        behaviors=[
            DelayInit(delays=src_delay),
            DelayInit(delays=dst_delay, side="dst"),
            SpikeGather(),
            Traces(tau_pre=10.0, tau_post=5.0),
        ],
    )
    return net, syn


class TestSpikeGatherAndTraces:
    def test_zero_delay_reads_this_steps_spikes(self):
        net, syn = gather_net()
        net.step()
        assert syn.pre_spike.tolist() == [True]
        assert syn.post_spike.tolist() == [False]
        assert syn.pre_trace.tolist() == [1.0]
        net.step()
        assert syn.post_spike.tolist() == [True]
        assert syn.pre_trace.tolist() == pytest.approx([0.9])
        assert syn.post_trace.tolist() == [1.0]

    def test_source_delay(self):
        net, syn = gather_net(src_delay=2)
        seen = []
        for _ in range(3):
            net.step()
            seen.append(bool(syn.pre_spike[0]))
        assert seen == [False, False, True]

    def test_needs_axons(self):
        net = Network()
        syn = SynapseGroup(net, NeuronGroup(net, 1), NeuronGroup(net, 1), behaviors=[SpikeGather()])
        with pytest.raises(RuntimeError, match="Axon"):
            net.initialize()
        assert syn.name == "sg0"

    def test_postsynaptic_spikes_need_an_axon_on_the_destination(self):
        net = Network()
        src = NeuronGroup(net, 1, [Axon()])
        syn = SynapseGroup(net, src, NeuronGroup(net, 1), behaviors=[SpikeGather()])
        net.initialize()
        assert syn.pre_spike.tolist() == [False]
        assert not hasattr(syn, "post_spike")

    def test_traces_need_postsynaptic_spikes(self):
        net = Network()
        src = NeuronGroup(net, 1, [Axon()])
        SynapseGroup(
            net,
            src,
            NeuronGroup(net, 1, name="out"),
            behaviors=[SpikeGather(), Traces(tau_pre=5.0)],
        )
        with pytest.raises(RuntimeError, match="an Axon on out"):
            net.initialize()

    def test_traces_need_gathered_spikes(self):
        net = Network()
        SynapseGroup(net, NeuronGroup(net, 1), NeuronGroup(net, 1), behaviors=[Traces(tau_pre=5.0)])
        with pytest.raises(RuntimeError, match="SpikeGather"):
            net.initialize()

    def test_invalid_tau(self):
        with pytest.raises(ValueError, match="tau_pre"):
            Traces(tau_pre=0.0)


def with_weights(src, dst, init, input_behavior, *extra):
    net = Network()
    syn = SynapseGroup(
        net,
        NeuronGroup(net, src, [Axon()]),
        NeuronGroup(net, dst),
        behaviors=[init, input_behavior, *extra, SpikeGather()],
    )
    net.initialize()
    return syn


class TestWeightClip:
    def test_clamps_including_negative_range(self):
        w = torch.tensor([[-2.0, 0.5], [3.0, -0.5]])
        syn = with_weights(
            2, 2, WeightInit(weights=w), DenseInput(), WeightClip(w_min=-1.0, w_max=1.0)
        )
        syn.behaviors[2].forward(syn)
        assert syn.weights.tolist() == [[-1.0, 0.5], [1.0, -0.5]]

    def test_invalid_interval(self):
        with pytest.raises(ValueError, match="w_min"):
            WeightClip(w_min=1.0, w_max=1.0)


class TestWeightNormalization:
    def test_dense_columns_sum_to_norm(self):
        w = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
        syn = with_weights(2, 2, WeightInit(weights=w), DenseInput(), WeightNormalization(norm=2.0))
        syn.behaviors[2].forward(syn)
        assert syn.weights.tolist() == [[0.5, 0.0], [1.5, 0.0]]

    def test_conv_kernels_sum_to_norm(self):
        net = Network()
        syn = SynapseGroup(
            net,
            NeuronGroup(net, (1, 3, 3), [Axon()]),
            NeuronGroup(net, (2, 2, 2)),
            behaviors=[
                WeightInit(mode="ones", shape=(2, 1, 2, 2)),
                Conv2dInput(),
                WeightNormalization(norm=1.0),
                SpikeGather(),
            ],
        )
        net.initialize()
        syn.behaviors[2].forward(syn)
        assert syn.weights.sum(dim=(1, 2, 3)).tolist() == pytest.approx([1.0, 1.0])

    def test_sparse_and_local_and_one_to_one(self):
        syn = with_weights(
            4,
            4,
            WeightInit(mode="ones", density=0.5, sparse=True),
            SparseInput(),
            WeightNormalization(),
        )
        syn.behaviors[2].forward(syn)
        sums = torch.zeros(4).index_add_(0, syn.dst_idx, syn.weights)
        assert all(s == pytest.approx(1.0) or s == 0.0 for s in sums.tolist())
        syn = with_weights(
            3, 3, WeightInit(mode=2.0, shape=(3,)), OneToOneInput(), WeightNormalization(norm=0.5)
        )
        syn.behaviors[2].forward(syn)
        assert syn.weights.tolist() == [0.5, 0.5, 0.5]
        net = Network()
        syn = SynapseGroup(
            net,
            NeuronGroup(net, (1, 2, 2), [Axon()]),
            NeuronGroup(net, (1, 1, 2)),
            behaviors=[
                WeightInit(mode="ones", shape=(1, 2, 2)),
                Local2dInput(kernel_size=(2, 1)),
                WeightNormalization(),
                SpikeGather(),
            ],
        )
        net.initialize()
        syn.behaviors[2].forward(syn)
        assert syn.weights.sum(-1).tolist() == [[1.0, 1.0]]


class TestCurrentNormalization:
    def test_scales_current_by_incoming_weight_sum(self):
        w = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
        syn = with_weights(
            2, 2, WeightInit(weights=w), DenseInput(), CurrentNormalization(norm=2.0)
        )
        syn.I = torch.tensor([4.0, 5.0])
        syn.behaviors[2].forward(syn)
        # column sums (4, 0): zero sums are left alone
        assert syn.I.tolist() == [2.0, 5.0]

    def test_needs_an_input_behavior(self):
        net = Network()
        SynapseGroup(
            net,
            NeuronGroup(net, 1),
            NeuronGroup(net, 1),
            behaviors=[WeightInit(mode="ones"), CurrentNormalization()],
        )
        with pytest.raises(RuntimeError, match="input"):
            net.initialize()
