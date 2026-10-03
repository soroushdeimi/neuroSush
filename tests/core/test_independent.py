"""Independent batches: B members behave as B separate networks."""

import pytest
import torch

from neurosush import checkpoint
from neurosush.core.behavior import Behavior
from neurosush.core.graph import GraphStepper
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import KWTA, InherentNoise
from neurosush.neurons.dendrite import (
    ConductanceIntegration,
    DendriteIntegration,
    DendriteStructure,
)
from neurosush.neurons.homeostasis import ActivityHomeostasis, AdaptiveThreshold
from neurosush.neurons.inputs import PoissonInput, SpikeInput
from neurosush.neurons.models import LIF, Fire, Refractory
from neurosush.recording import SpikeCounter
from neurosush.synapses.constraints import CurrentNormalization, WeightClip, WeightNormalization
from neurosush.synapses.currents import (
    AvgPool2dInput,
    Conv2dInput,
    DenseInput,
    LateralInput,
    Local2dInput,
    OneToOneInput,
    SparseInput,
)
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import ISTDP, RSTDP, STDP
from neurosush.synapses.traces import SpikeGather, Traces
from neurosush.synapses.triplet import TripletSTDP

N_IN, N_EXC, MEMBERS, STEPS = 12, 5, 3, 60


def scripted(seed=0, members=MEMBERS, steps=STEPS):
    """Spike frames ``(steps, members, N_IN)``, fixed and independent of any network's RNG."""
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(steps, members, N_IN, generator=gen) < 0.4


def build(frames, *, batch=None, independent=False, rule="pair", dtype=torch.float64, device="cpu"):
    """Input -> exc (conductances, adaptive threshold) <-> inh; frames are ``(steps, B, N_IN)``."""
    net = Network(dtype=dtype, device=device, seed=1, batch_size=batch, independent=independent)
    stream = [f if batch is not None else f[0] for f in frames]
    inp = NeuronGroup(net, N_IN, [SpikeInput(stream), Axon()], name="input")
    exc = NeuronGroup(
        net,
        N_EXC,
        [
            ConductanceIntegration(e_exc=0.0, e_inh=-100.0, tau_exc=1.0, tau_inh=2.0),
            LIF(tau=20.0, threshold=-52.0, v_reset=-65.0, v_rest=-65.0),
            Refractory(3.0),
            Fire(),
            AdaptiveThreshold(increment=0.5, tau=100.0),
            SpikeCounter(),
            Axon(),
        ],
        name="exc",
    )
    inh = NeuronGroup(
        net,
        N_EXC,
        [
            ConductanceIntegration(e_exc=0.0, e_inh=-85.0, tau_exc=1.0, tau_inh=2.0),
            LIF(tau=10.0, threshold=-40.0, v_reset=-45.0, v_rest=-60.0),
            Fire(),
            ActivityHomeostasis(target_spikes=2, window=10, rate=0.1, decay=0.9),
            Axon(),
        ],
        inhibitory=True,
        name="inh",
    )
    learn = (
        [Traces(tau_pre=20.0, tau_post=20.0), STDP(a_plus=0.05, a_minus=0.01)]
        if rule == "pair"
        else [
            TripletSTDP(
                a2_plus=0.01,
                a3_plus=0.05,
                a2_minus=0.02,
                a3_minus=0.01,
                tau_plus=20.0,
                tau_minus=20.0,
                tau_x=30.0,
                tau_y=40.0,
                interaction="all",
            )
        ]
    )
    syn = SynapseGroup(
        net,
        inp,
        exc,
        [
            WeightInit(mode="uniform", scale=0.5),
            DenseInput(),
            SpikeGather(),
            *learn,
            WeightClip(w_min=0.0, w_max=1.0),
            WeightNormalization(norm=2.0),
        ],
        name="input->exc",
    )
    to_inh = SynapseGroup(
        net,
        exc,
        inh,
        [
            WeightInit(mode="uniform", scale=6.0, offset=2.0, shape=(N_EXC,)),
            OneToOneInput(),
            SpikeGather(),
            Traces(tau_pre=10.0),
            STDP(a_plus=0.01, a_minus=0.01, bound="soft"),
        ],
        name="exc->inh",
    )
    back = SynapseGroup(
        net,
        inh,
        exc,
        [
            WeightInit(mode="uniform", scale=3.0),
            DenseInput(),
            SpikeGather(),
            Traces(tau_pre=10.0),
            ISTDP(lr=0.01, rho=0.05),
        ],
        name="inh->exc",
    )
    net.initialize()
    return net, syn, to_inh, back, exc, inh


def _states(net):
    out = {}
    for group in net.groups:
        for name in ("v", "spikes", "I", "g_exc", "g_inh", "refractory", "threshold", "theta"):
            if hasattr(group, name):
                out[f"{group.name}.{name}"] = getattr(group, name)
    for syn in net.synapses:
        for name in ("weights", "pre_trace", "post_trace"):
            if hasattr(syn, name):
                out[f"{syn.name}.{name}"] = getattr(syn, name)
    return out


@pytest.mark.parametrize("rule", ["pair", "triplet"])
def test_independent_batch_equals_separate_networks(rule):
    frames = scripted()
    batch = build(frames, batch=MEMBERS, independent=True, rule=rule)
    singles = [build(frames[:, b : b + 1], rule=rule) for b in range(MEMBERS)]
    for b, single in enumerate(singles):
        for key in ("weights",):
            for i in (1, 2, 3):  # input->exc, exc->inh, inh->exc
                getattr(single[i], key).copy_(getattr(batch[i], key)[b])
    # members must start different, otherwise the test proves little
    assert not torch.equal(batch[1].weights[0], batch[1].weights[1])
    for _ in range(STEPS):
        batch[0].step()
        for single in singles:
            single[0].step()
    spikes = 0
    for b, single in enumerate(singles):
        got, want = _states(batch[0]), _states(single[0])
        assert got.keys() == want.keys()
        for key in got:
            torch.testing.assert_close(got[key][b], want[key], rtol=0, atol=1e-12, msg=key)
        spikes += int(single[4].spike_count.sum())
    assert spikes > 0
    assert batch[1].weights.shape == (MEMBERS, N_IN, N_EXC)
    assert batch[2].weights.shape == (MEMBERS, N_EXC)
    assert batch[4].theta.shape == (MEMBERS, N_EXC)
    assert (batch[4].theta > 0).any()
    assert (batch[1].weights != batch[1].weights[:1]).any()


def test_members_do_not_interact():
    frames = scripted()
    other = frames.clone()
    other[:, 1] = scripted(seed=9)[:, 1]
    a = build(frames, batch=MEMBERS, independent=True)
    b = build(other, batch=MEMBERS, independent=True)
    b[1].weights.copy_(a[1].weights)
    b[2].weights.copy_(a[2].weights)
    b[3].weights.copy_(a[3].weights)
    a[0].run(STEPS)
    b[0].run(STEPS)
    for key, value in _states(a[0]).items():
        other_value = _states(b[0])[key]
        for member in (0, 2):
            assert torch.equal(value[member], other_value[member]), key
    assert not torch.equal(a[1].weights[1], b[1].weights[1])


def test_shared_mode_still_shares():
    net = Network(batch_size=3, dtype=torch.float64)
    group = NeuronGroup(net, 4)
    assert group.vector().shape == (4,)
    assert not net.independent
    indep = Network(batch_size=3, independent=True)
    assert NeuronGroup(indep, 4).vector().shape == (3, 4)


def test_requires_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        Network(independent=True)


def test_explicit_weights_and_threshold_per_member():
    net = Network(batch_size=2, independent=True, dtype=torch.float64)
    inp = NeuronGroup(net, 3, [SpikeInput([torch.ones(2, 3)] * 3), Axon()])
    thr = torch.tensor([[-50.0, -50.0], [-60.0, -60.0]], dtype=torch.float64)
    out = NeuronGroup(
        net,
        2,
        [
            DendriteStructure(),
            DendriteIntegration(),
            LIF(tau=5.0, threshold=thr, v_reset=-70.0, v_rest=-65.0),
            Fire(),
            Axon(),
        ],
    )
    w = torch.rand(2, 3, 2, dtype=torch.float64)
    SynapseGroup(net, inp, out, [WeightInit(weights=w), DenseInput(), SpikeGather()])
    net.initialize()
    assert torch.equal(out.threshold, thr)
    assert torch.equal(net.synapses[0].weights, w)
    n2 = Network(batch_size=2, independent=True)
    a = NeuronGroup(n2, 3, [SpikeInput([torch.ones(2, 3)]), Axon()])
    b = NeuronGroup(n2, 2, [Axon()])
    SynapseGroup(n2, a, b, [WeightInit(weights=torch.rand(5, 3, 2)), DenseInput(), SpikeGather()])
    with pytest.raises(ValueError, match="weights shape"):
        n2.initialize()
    # one shared weight matrix is copied to every member
    n3 = Network(batch_size=2, independent=True)
    a = NeuronGroup(n3, 3, [SpikeInput([torch.ones(2, 3)]), Axon()])
    b = NeuronGroup(n3, 2, [Axon()])
    s = SynapseGroup(n3, a, b, [WeightInit(weights=torch.rand(3, 2)), DenseInput(), SpikeGather()])
    n3.initialize()
    assert torch.equal(s.weights[0], s.weights[1])


def _one_group(extra_group=(), extra_syn=(), *, kind="dense"):
    net = Network(batch_size=2, independent=True)
    a = NeuronGroup(net, (1, 4, 4), [PoissonInput(0.1), Axon()])
    b = NeuronGroup(
        net,
        (1, 4, 4),
        [LIF(tau=5.0, threshold=1.0, v_reset=0.0, v_rest=0.0), Fire(), Axon(), *extra_group],
    )
    SynapseGroup(net, a, b, [SpikeGather(), *extra_syn])
    return net


@pytest.mark.parametrize(
    ("extra_group", "extra_syn", "name"),
    [
        ((), [WeightInit(mode="uniform", density=0.5, sparse=True), SparseInput()], "SparseInput"),
        (
            (),
            [WeightInit(mode="uniform", shape=(1, 1, 3, 3)), Conv2dInput(padding=1)],
            "Conv2dInput",
        ),
        (
            (),
            [WeightInit(mode="uniform", shape=(1, 16, 9)), Local2dInput(kernel_size=3, padding=1)],
            "Local2dInput",
        ),
        ((), [AvgPool2dInput()], "AvgPool2dInput"),
        (
            (),
            [
                WeightInit(mode="uniform", shape=(16, 16)),
                DenseInput(),
                Traces(tau_pre=5.0),
                RSTDP(a_plus=0.1, a_minus=0.1, tau_c=10.0),
            ],
            "RSTDP",
        ),
    ],
)
def test_unsupported_behaviors_raise(extra_group, extra_syn, name):
    net = _one_group(extra_group, extra_syn)
    with pytest.raises(NotImplementedError, match=name):
        net.initialize()


def test_lateral_and_custom_behaviors_raise():
    net = Network(batch_size=2, independent=True)
    g = NeuronGroup(net, (1, 4, 4), [PoissonInput(0.1), Axon()])
    SynapseGroup(
        net, g, g, [WeightInit(mode="ones", shape=(1, 1, 1, 3, 3)), LateralInput(), SpikeGather()]
    )
    with pytest.raises(NotImplementedError, match="LateralInput"):
        net.initialize()

    class Custom(Behavior):
        order = Order.NOISE

    net = Network(batch_size=2, independent=True, behaviors=[Custom()])
    with pytest.raises(NotImplementedError, match="Custom on the network"):
        net.initialize()
    # the same network is fine when shared
    Network(batch_size=2, behaviors=[Custom()]).initialize()


def test_normalization_kwta_noise_and_current_normalization_per_member():
    net = Network(batch_size=2, independent=True, dtype=torch.float64, seed=0)
    a = NeuronGroup(net, 6, [PoissonInput(0.5), Axon()])
    b = NeuronGroup(
        net,
        4,
        [
            DendriteStructure(),
            DendriteIntegration(),
            LIF(tau=5.0, threshold=0.5, v_reset=0.0, v_rest=0.0),
            InherentNoise(scale=0.1),
            KWTA(2),
            Fire(),
            Axon(),
        ],
    )
    SynapseGroup(
        net,
        a,
        b,
        [WeightInit(mode="uniform"), DenseInput(), SpikeGather(), CurrentNormalization(norm=1.0)],
    )
    net.initialize()
    net.run(20)
    assert b.v.shape == (2, 4)
    assert int(b.spikes.sum(-1).max()) <= 2  # KWTA per member


def test_checkpoint_and_reset_roundtrip(tmp_path):
    frames = scripted()
    a = build(frames, batch=MEMBERS, independent=True)
    a[0].run(30)
    path = tmp_path / "net.pt"
    checkpoint.save(a[0], path)
    b = build(frames[30:], batch=MEMBERS, independent=True)
    b[1].weights.zero_()
    checkpoint.load(b[0], path)
    a[0].run(30)
    b[0].run(30)
    for key, value in _states(a[0]).items():
        assert torch.equal(value, _states(b[0])[key]), key
    weights = a[1].weights.clone()
    theta = a[4].theta.clone()
    a[0].reset_state()
    assert torch.equal(a[1].weights, weights)
    assert torch.equal(a[4].theta, theta)
    assert not a[4].spike_count.any()
    assert not a[1].pre_trace.any()


@pytest.mark.gpu
@pytest.mark.parametrize("rule", ["pair", "triplet"])
def test_graph_stepper_matches_eager(rule):
    frames = scripted(steps=40).cuda()
    eager = build(
        frames, batch=MEMBERS, independent=True, rule=rule, dtype=torch.float32, device="cuda"
    )
    graphed = build(
        frames, batch=MEMBERS, independent=True, rule=rule, dtype=torch.float32, device="cuda"
    )
    for i in (1, 2, 3):
        graphed[i].weights.copy_(eager[i].weights)
    stepper = GraphStepper(graphed[0], warmup=2)
    eager[0].run(40)
    stepper.run(40)
    assert int(eager[4].spike_count.sum()) > 0
    for key, value in _states(eager[0]).items():
        assert torch.equal(value, _states(graphed[0])[key]), key


def test_sparse_weight_init_raises():
    net = Network(batch_size=2, independent=True)
    a = NeuronGroup(net, 4, [PoissonInput(0.1), Axon()])
    b = NeuronGroup(net, 4, [Axon()])
    SynapseGroup(net, a, b, [WeightInit(mode="uniform", density=0.5, sparse=True)])
    with pytest.raises(NotImplementedError, match="sparse"):
        net.initialize()
