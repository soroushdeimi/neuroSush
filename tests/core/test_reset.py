import itertools

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.modulation import Dopamine, Payoff
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import MinicolumnInhibition
from neurosush.neurons.dendrite import (
    ConductanceIntegration,
    DendriteIntegration,
    DendriteStructure,
)
from neurosush.neurons.homeostasis import ActivityHomeostasis, AdaptiveThreshold
from neurosush.neurons.inputs import PoissonInput, SpikeInput
from neurosush.neurons.models import LIF, AdaptiveELIF, Fire, Refractory
from neurosush.recording import Recorder
from neurosush.synapses.currents import DenseInput, OneToOneInput
from neurosush.synapses.init import DelayInit, WeightInit
from neurosush.synapses.plasticity import RSTDP, STDP
from neurosush.synapses.segment_learning import SegmentLearning
from neurosush.synapses.segments import ActiveSegments
from neurosush.synapses.traces import SpikeGather, Traces

SAMPLE = 8  # two periods of the four input frames


def frames():
    """Four fixed frames over 4 inputs; a sample of 8 steps shows each twice."""
    return [
        torch.tensor(f, dtype=torch.bool)
        for f in ([1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1], [0, 0, 0, 1])
    ]


def build_delayed():
    """LIF with dendritic delays (depth 3), traces and STDP; deterministic input."""
    net = Network(seed=0)
    src = NeuronGroup(net, 4, [SpikeInput(itertools.cycle(frames())), Axon(max_delay=3)])
    dst = NeuronGroup(
        net,
        3,
        [
            DendriteStructure(proximal_depth=3),
            DendriteIntegration(tau_current=4.0),
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0, v_init=-60.0),
            Refractory(2.0),
            Fire(),
            Axon(max_delay=3),
        ],
    )
    syn = SynapseGroup(
        net,
        src,
        dst,
        [
            WeightInit(weights=torch.full((4, 3), 8.0)),
            DelayInit(delays=torch.tensor([0, 1, 2]), side="dst"),
            DelayInit(delays=2),
            DenseInput(),
            SpikeGather(),
            Traces(tau_pre=5.0),
            STDP(a_plus=0.05, a_minus=0.02),
        ],
    )
    net.initialize()
    return net, src, dst, syn


def build_conductance():
    """Excitatory conductance group with adaptation, inhibition and RSTDP."""
    net = Network(
        seed=0, behaviors=[Payoff(lambda n: 1.0, initial=0.5), Dopamine(tau=10.0, initial=0.2)]
    )
    src = NeuronGroup(net, 4, [SpikeInput(itertools.cycle(frames())), Axon()])
    dst = NeuronGroup(
        net,
        3,
        [
            ConductanceIntegration(e_exc=0.0, e_inh=-100.0, tau_exc=2.0, tau_inh=2.0),
            AdaptiveELIF(
                tau=20.0,
                threshold=-52.0,
                v_reset=-65.0,
                v_rest=-65.0,
                delta=2.0,
                theta_rh=-50.0,
                alpha=0.1,
                beta=0.5,
                tau_w=30.0,
                omega_init=0.25,
                v_init=-62.0,
            ),
            Refractory(3.0),
            Fire(),
            Axon(),
        ],
    )
    inh = NeuronGroup(
        net,
        3,
        [
            ConductanceIntegration(e_exc=0.0, e_inh=-85.0, tau_exc=1.0, tau_inh=2.0),
            LIF(tau=10.0, threshold=-40.0, v_reset=-45.0, v_rest=-60.0),
            Fire(),
            Axon(),
        ],
        inhibitory=True,
    )
    syn = SynapseGroup(
        net,
        src,
        dst,
        [
            WeightInit(weights=torch.full((4, 3), 3.0)),
            DenseInput(),
            SpikeGather(),
            Traces(tau_pre=5.0),
            RSTDP(tau_c=20.0, a_plus=0.05, a_minus=0.02),
        ],
    )
    SynapseGroup(
        net,
        dst,
        inh,
        [WeightInit(weights=torch.full((3,), 20.0), shape=(3,)), OneToOneInput(), SpikeGather()],
    )
    SynapseGroup(
        net, inh, dst, [WeightInit(weights=torch.full((3, 3), 5.0)), DenseInput(), SpikeGather()]
    )
    net.initialize()
    return net, src, dst, syn


def build_segments():
    """Minicolumns with a distal ActiveSegments input and a proximal driver."""
    net = Network(seed=0)
    src = NeuronGroup(net, 4, [SpikeInput(itertools.cycle(frames())), Axon()])
    dst = NeuronGroup(
        net,
        4,
        [
            DendriteStructure(),
            DendriteIntegration(distal_gain=0.5),
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            MinicolumnInhibition(cells_per_column=2, duration=2.0),
            Fire(),
            Axon(),
        ],
    )
    segments = ActiveSegments(
        segments=1,
        synapses=2,
        activation_threshold=2,
        plateau=3.0,
        amplitude=5.0,
        presynaptic=torch.tensor([[[0, 1]], [[1, 2]], [[2, 3]], [[0, 3]]]),
        permanence=torch.ones(4, 1, 2),
    )
    syn = SynapseGroup(
        net,
        src,
        dst,
        [WeightInit(weights=torch.full((4, 4), 20.0)), DenseInput(), SpikeGather()],
    )
    SynapseGroup(net, src, dst, [segments, SpikeGather()], compartment="distal")
    net.initialize()
    return net, src, dst, syn


BUILDERS = [build_delayed, build_conductance, build_segments]


def run_sample(net, dst):
    """Per-step ``v``, spikes and (when present) ``I`` of ``dst`` over one sample."""
    log = []
    for _ in range(SAMPLE):
        net.step()
        log.append((dst.v.clone(), dst.spikes.clone(), dst.I.clone()))
    return log


def snapshot(net):
    """Every tensor attribute of the groups and synapses, by name."""
    return {
        (obj.name, name): value
        for obj in (*net.groups, *net.synapses)
        for name, value in vars(obj).items()
        if isinstance(value, torch.Tensor)
    }


@pytest.mark.parametrize("build", BUILDERS)
def test_a_reset_network_repeats_a_fresh_one(build):
    fresh_net, _, fresh_dst, fresh_syn = build()
    expected = run_sample(fresh_net, fresh_dst)

    net, _, dst, syn = build()
    weights = syn.weights.clone()
    first = run_sample(net, dst)
    assert any(spikes.any() for _, spikes, _ in first)  # the sample does something
    net.reset_state()
    syn.weights.copy_(weights)  # learning moved them; only the dynamics are under test
    got = run_sample(net, dst)

    for (v, spikes, current), (e_v, e_spikes, e_current) in zip(got, expected, strict=True):
        assert torch.equal(v, e_v)
        assert torch.equal(spikes, e_spikes)
        assert torch.equal(current, e_current)
    if hasattr(syn, "pre_trace"):
        assert torch.equal(syn.pre_trace, fresh_syn.pre_trace)
        assert torch.equal(syn.post_trace, fresh_syn.post_trace)
    if hasattr(syn, "eligibility"):
        assert torch.equal(syn.eligibility, fresh_syn.eligibility)
    assert torch.equal(syn.weights, fresh_syn.weights)


def test_reset_clears_the_state_to_its_initial_values():
    net, src, dst, syn = build_delayed()
    net.run(SAMPLE)
    net.reset_state()
    assert (dst.v == -60.0).all()  # v_init
    assert not dst.spikes.any()
    assert (dst.I == 0).all()
    assert (dst.refractory == 0).all()
    assert not src.spikes.any()
    assert not src.spike_history.read(0).any()
    assert all((buffer.current() == 0).all() for buffer in dst.dendrite.values())
    assert (dst.I_proximal == 0).all()
    assert (syn.I == 0).all()
    assert (syn.pre_trace == 0).all()
    assert (syn.post_trace == 0).all()
    assert not syn.pre_spike.any()
    assert not syn.post_spike.any()


def test_reset_returns_adaptation_modulation_and_conductances_to_their_initial_values():
    net, _, dst, syn = build_conductance()
    net.run(SAMPLE)
    assert (dst.omega != 0.25).any()
    assert net.dopamine != 0.2
    assert syn.eligibility.any()
    net.reset_state()
    assert (dst.omega == 0.25).all()
    assert (dst.v == -62.0).all()
    assert (dst.g_exc == 0).all()
    assert (dst.g_inh == 0).all()
    assert (syn.eligibility == 0).all()
    assert net.payoff == 0.5
    assert net.dopamine == 0.2


def test_reset_ends_plateaus_and_inhibition():
    net, _, dst, _ = build_segments()
    syn = net.synapses[1]  # the segments
    net.run(SAMPLE)
    assert syn.pre_recent.any()
    assert syn.plateau_steps.any()
    assert dst.column_inhibition.any()
    net.reset_state()
    assert (syn.plateau_steps == 0).all()
    assert (syn.pre_recent == 0).all()
    assert not syn.active_segments.any()
    assert (syn.segment_potential == 0).all()
    assert (dst.column_inhibition == 0).all()
    assert (syn.permanence == 1.0).all()


def test_segment_learning_forgets_times_but_keeps_what_was_learned():
    net = Network(seed=0)
    layer = NeuronGroup(net, 4, [PoissonInput(0.5), Axon()])
    syn = SynapseGroup(
        net,
        layer,
        layer,
        [
            ActiveSegments(segments=1, synapses=2, activation_threshold=2, plateau=3.0),
            SpikeGather(),
            SegmentLearning(cells_per_column=2, context=(1.0, 3.0), min_threshold=1),
        ],
        compartment="distal",
    )
    net.initialize()
    syn.segment_start.fill_(7)
    syn.last_spike.fill_(7)
    syn.last_win.fill_(7)
    syn.segment_used.fill_(7)
    syn.activation_synapses.fill_(True)
    syn.permanence.fill_(0.4)
    syn.presynaptic.fill_(1)
    net.reset_state()
    never = -(10**9)
    assert (syn.segment_start == never).all()
    assert (syn.last_spike == never).all()
    assert (syn.last_win == never).all()
    assert not syn.activation_synapses.any()
    assert (syn.segment_used == 7).all()
    assert (syn.permanence == 0.4).all()
    assert (syn.presynaptic == 1).all()


def test_learned_and_parameter_state_is_kept():
    net = Network(seed=0)
    src = NeuronGroup(net, 4, [SpikeInput(itertools.cycle(frames())), Axon()])
    dst = NeuronGroup(
        net,
        3,
        [
            ConductanceIntegration(),
            LIF(tau=10.0, threshold=-60.0, v_reset=-70.0, v_rest=-65.0),
            Fire(),
            AdaptiveThreshold(increment=0.5),
            ActivityHomeostasis(target_spikes=1, window=4, rate=0.1),
            Axon(),
        ],
    )
    syn = SynapseGroup(
        net,
        src,
        dst,
        [
            WeightInit(weights=torch.full((4, 3), 30.0)),
            DenseInput(),
            SpikeGather(),
            Traces(tau_pre=5.0),
            STDP(a_plus=0.05, a_minus=0.02),
        ],
    )
    net.initialize()
    net.run(SAMPLE)
    assert dst.theta.any()
    homeostasis = next(b for b in dst.behaviors if isinstance(b, ActivityHomeostasis))
    activity = homeostasis.activity.clone()
    weights, theta, threshold = syn.weights.clone(), dst.theta.clone(), dst.threshold.clone()
    net.reset_state()
    assert torch.equal(syn.weights, weights)
    assert torch.equal(dst.theta, theta)
    assert torch.equal(dst.threshold, threshold)
    assert torch.equal(homeostasis.activity, activity)
    assert net.iteration == SAMPLE


@pytest.mark.parametrize("build", BUILDERS)
def test_addresses_do_not_change(build):
    net, *_ = build()
    net.run(SAMPLE)
    before = {key: value.data_ptr() for key, value in snapshot(net).items()}
    net.reset_state()
    after = {key: value.data_ptr() for key, value in snapshot(net).items()}
    assert before == after


def test_a_spike_input_keeps_its_place_in_the_stream():
    net = Network()
    group = NeuronGroup(net, 4, [SpikeInput(iter(frames()))])
    net.run(2)
    net.reset_state()
    assert not group.spikes.any()
    net.step()
    assert group.spikes.tolist() == frames()[2].tolist()


def test_a_poisson_input_keeps_its_rates():
    net = Network()
    group = NeuronGroup(net, 3, [PoissonInput(torch.tensor([0.1, 0.2, 0.3]))])
    net.run(5)
    net.reset_state()
    assert not group.spikes.any()
    assert torch.allclose(group.rates, torch.tensor([0.1, 0.2, 0.3]))


def test_batched_state_is_reset_per_sample():
    net = Network(batch_size=2, seed=0)
    group = NeuronGroup(
        net,
        3,
        [
            PoissonInput(1.0),
            Axon(),
        ],
    )
    NeuronGroup(
        net,
        3,
        [
            LIF(
                tau=10.0,
                threshold=-55.0,
                v_reset=-70.0,
                v_rest=-65.0,
                v_init=torch.tensor([-60.0, -61.0, -62.0]),
            )
        ],
    )
    net.run(3)
    net.reset_state()
    assert not group.spikes.any()
    assert (net.groups[1].v == torch.tensor([-60.0, -61.0, -62.0])).all()


def test_the_recorder_keeps_its_recordings():
    net = Network()
    recorder = Recorder("v")
    NeuronGroup(net, 2, [LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0), recorder])
    net.run(3)
    net.reset_state()
    assert len(recorder.steps) == 3


def test_disabled_behaviors_are_reset_too():
    net = Network()
    group = NeuronGroup(
        net, 2, [lif := LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0)]
    )
    net.initialize()
    group.v.fill_(-1.0)
    lif.enabled = False
    net.reset_state()
    assert (group.v == -65.0).all()


def test_needs_an_initialized_network():
    with pytest.raises(RuntimeError, match="initialized"):
        Network().reset_state()


def test_behaviors_without_per_sample_state_do_nothing_by_default():
    from neurosush.core.behavior import Behavior

    assert Behavior().reset_state(object()) is None
