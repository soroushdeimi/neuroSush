import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.modulation import Dopamine, Payoff
from neurosush.neurons.axon import Axon
from neurosush.synapses.currents import AvgPool2dInput, DenseInput, OneToOneInput, SparseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import ISTDP, RSTDP, STDP
from neurosush.synapses.traces import SpikeGather, Traces


def constant(value):
    return lambda net: value


def learning_synapse(rule, *, weights=None, input_behavior=None, traces=None, net=None, size=2):
    net = net or Network()
    src = NeuronGroup(net, size, behaviors=[Axon()])
    dst = NeuronGroup(net, size, behaviors=[Axon()])
    if weights is None:
        init = WeightInit(mode=0.5)
    else:
        init = WeightInit(weights=weights, shape=tuple(weights.shape))
    behaviors = [
        init,
        input_behavior or DenseInput(),
        SpikeGather(),
        traces or Traces(tau_pre=10.0),
        rule,
    ]
    syn = SynapseGroup(net, src, dst, behaviors=behaviors)
    net.initialize()
    return net, syn


def set_activity(syn, pre_spike, pre_trace, post_spike, post_trace):
    syn.pre_spike = torch.tensor(pre_spike)
    syn.pre_trace = torch.tensor(pre_trace)
    syn.post_spike = torch.tensor(post_spike)
    syn.post_trace = torch.tensor(post_trace)


class TestSTDP:
    def test_dense_update(self):
        _, syn = learning_synapse(STDP(a_plus=0.1, a_minus=0.2))
        set_activity(syn, [True, False], [0.0, 2.0], [True, False], [0.0, 0.5])
        syn.behaviors[-1].forward(syn)
        torch.testing.assert_close(syn.weights, torch.tensor([[0.5, 0.4], [0.7, 0.5]]))

    def test_soft_bound_scales_the_update(self):
        _, syn = learning_synapse(
            STDP(a_plus=1.0, a_minus=1.0, bound="soft"), weights=torch.tensor([[0.25]]), size=1
        )
        set_activity(syn, [False], [1.0], [True], [0.0])
        syn.behaviors[-1].forward(syn)
        torch.testing.assert_close(syn.weights, torch.tensor([[0.25 + 0.75]]))

    def test_one_to_one_and_sparse(self):
        _, syn = learning_synapse(
            STDP(a_plus=1.0, a_minus=1.0),
            weights=torch.tensor([0.5, 0.5]),
            input_behavior=OneToOneInput(),
        )
        set_activity(syn, [True, False], [0.0, 3.0], [False, True], [1.0, 0.0])
        syn.behaviors[-1].forward(syn)
        assert syn.weights.tolist() == pytest.approx([-0.5, 3.5])

    def test_sparse_updates_values(self):
        net = Network(seed=0)
        src = NeuronGroup(net, 3, behaviors=[Axon()])
        dst = NeuronGroup(net, 3, behaviors=[Axon()])
        syn = SynapseGroup(
            net,
            src,
            dst,
            behaviors=[
                WeightInit(mode="ones", density=0.5, sparse=True),
                SparseInput(),
                SpikeGather(),
                Traces(tau_pre=5.0),
                STDP(a_plus=1.0, a_minus=0.0),
            ],
        )
        net.initialize()
        set_activity(syn, [False] * 3, [1.0, 1.0, 1.0], [True] * 3, [0.0] * 3)
        syn.behaviors[-1].forward(syn)
        assert syn.weights.tolist() == [2.0] * syn.weights.numel()

    def test_needs_traces(self):
        net = Network()
        src = NeuronGroup(net, 1, behaviors=[Axon()])
        SynapseGroup(
            net,
            src,
            src,
            behaviors=[
                WeightInit(mode=0.5),
                DenseInput(),
                SpikeGather(),
                STDP(a_plus=0.1, a_minus=0.1),
            ],
        )
        with pytest.raises(RuntimeError, match="Traces"):
            net.initialize()

    def test_unsupported_connectivity(self):
        net = Network()
        src = NeuronGroup(net, (1, 2, 2), behaviors=[Axon()])
        dst = NeuronGroup(net, (1, 1, 1), behaviors=[Axon()])
        SynapseGroup(
            net,
            src,
            dst,
            behaviors=[
                AvgPool2dInput(),
                SpikeGather(),
                Traces(tau_pre=5.0),
                STDP(a_plus=0.1, a_minus=0.1),
            ],
        )
        with pytest.raises(ValueError, match="avg_pool"):
            net.initialize()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"a_plus": -0.1, "a_minus": 0.1}, "a_plus"),
            ({"a_plus": 0.1, "a_minus": -0.1}, "a_minus"),
            ({"a_plus": 0.1, "a_minus": 0.1, "w_min": 1.0, "w_max": 1.0}, "w_min"),
            ({"a_plus": 0.1, "a_minus": 0.1, "bound": "tanh"}, "bound"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            STDP(**kwargs)

    def test_not_graph_ready_on_cpu(self):
        _, syn = learning_synapse(STDP(a_plus=0.1, a_minus=0.1))
        assert syn.behaviors[-1].graph_ready(syn) is False

    @pytest.mark.gpu
    def test_graph_ready_on_cuda(self):
        net = Network(device="cuda")
        _, syn = learning_synapse(STDP(a_plus=0.1, a_minus=0.1), net=net)
        assert syn.behaviors[-1].graph_ready(syn) is True


class TestRSTDP:
    def make(self, payoff=1.0, dt=1.0):
        net = Network(dt=dt, behaviors=[Payoff(constant(payoff)), Dopamine(tau=10.0)])
        return learning_synapse(RSTDP(a_plus=1.0, a_minus=1.0, tau_c=4.0), net=net, size=1)

    def test_eligibility_then_weight(self):
        net, syn = self.make()
        net.dopamine = 2.0
        set_activity(syn, [False], [1.0], [True], [0.0])
        syn.behaviors[-1].forward(syn)
        # c = c * (1 - dt / tau_c) + dw = 1; w += dt * dopamine * c
        assert syn.eligibility.tolist() == [[1.0]]
        assert syn.weights.tolist() == [[0.5 + 2.0]]
        set_activity(syn, [False], [0.0], [False], [0.0])
        syn.behaviors[-1].forward(syn)
        assert syn.eligibility.tolist() == [[0.75]]
        torch.testing.assert_close(syn.weights, torch.tensor([[2.5 + 1.5]]))

    def test_no_dopamine_no_weight_change(self):
        net, syn = self.make()
        net.dopamine = 0.0
        set_activity(syn, [False], [1.0], [True], [0.0])
        syn.behaviors[-1].forward(syn)
        assert syn.weights.tolist() == [[0.5]]

    def test_needs_dopamine(self):
        with pytest.raises(RuntimeError, match="Dopamine"):
            learning_synapse(RSTDP(a_plus=1.0, a_minus=1.0, tau_c=4.0))

    def test_tau_c_must_exceed_dt(self):
        net = Network(dt=2.0, behaviors=[Payoff(constant(0.0)), Dopamine(tau=10.0)])
        with pytest.raises(ValueError, match="tau_c"):
            learning_synapse(RSTDP(a_plus=1.0, a_minus=1.0, tau_c=2.0), net=net)

    def test_never_graph_ready(self):
        _, syn = self.make()
        assert syn.behaviors[-1].graph_ready(syn) is False


class TestISTDP:
    def test_alpha_from_target_rate(self):
        _, syn = learning_synapse(ISTDP(lr=0.1, rho=0.005), traces=Traces(tau_pre=20.0, scale=2.0))
        # alpha = 2 * rho * tau * scale
        assert syn.behaviors[-1].alpha == pytest.approx(0.4)

    def test_dense_update(self):
        _, syn = learning_synapse(ISTDP(lr=0.5, alpha=0.2))
        set_activity(syn, [True, False], [0.0, 0.4], [False, True], [0.3, 0.1])
        syn.behaviors[-1].forward(syn)
        torch.testing.assert_close(syn.weights, torch.tensor([[0.55, 0.45], [0.5, 0.7]]))

    def test_needs_symmetric_traces(self):
        with pytest.raises(ValueError, match="symmetric"):
            learning_synapse(ISTDP(lr=0.1, rho=0.01), traces=Traces(tau_pre=10.0, tau_post=20.0))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"lr": 0.1}, "rho"),
            ({"lr": 0.1, "rho": 0.1, "alpha": 0.1}, "rho"),
            ({"lr": 0.0, "rho": 0.1}, "lr"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ISTDP(**kwargs)
