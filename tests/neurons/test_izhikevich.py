import shutil

import pytest
import torch

from neurosush.core.compiled import CompiledStepper
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.adaptation import SpikeTriggeredCurrent
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import (
    ConductanceIntegration,
    DendriteIntegration,
    DendriteStructure,
)
from neurosush.neurons.inputs import CorrelatedPoissonInput, PoissonDrive, PoissonInput
from neurosush.neurons.models import LIF, AdaptiveELIF, Fire, Izhikevich
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.traces import SpikeGather

RS = {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0}


def make(size=3, *, dt=1.0, **kwargs):
    net = Network(dt=dt, dtype=torch.float64)
    group = NeuronGroup(net, size, [Izhikevich(**{**RS, **kwargs}), Fire()])
    net.initialize()
    return net, group


class TestIzhikevich:
    def test_initial_state_follows_the_paper(self):
        _, group = make()
        assert group.v.tolist() == [-65.0] * 3
        assert group.u.tolist() == pytest.approx([-13.0] * 3)
        assert group.threshold.tolist() == [30.0] * 3
        assert group.v_reset == -65.0
        assert group.spikes.dtype == torch.bool
        assert group.model is group.behaviors[0]

    def test_v_init_and_u_init(self):
        _, group = make(2, v_init=torch.tensor([-70.0, -60.0]), u_init=1.5)
        assert group.v.tolist() == [-70.0, -60.0]
        assert group.u.tolist() == [1.5, 1.5]
        _, group = make(2, v_init=-60.0)
        assert group.u.tolist() == pytest.approx([-12.0, -12.0])

    def test_per_neuron_parameters(self):
        a = torch.tensor([0.02, 0.1, 0.02], dtype=torch.float64)
        c = torch.tensor([-65.0, -65.0, -50.0], dtype=torch.float64)
        d = torch.tensor([8.0, 2.0, 2.0], dtype=torch.float64)
        net, group = make(3, a=a, c=c, d=d)
        group.v = torch.full((3,), 31.0, dtype=torch.float64)
        u0 = group.u.clone()
        group.model.fire(group)
        assert group.v.tolist() == c.tolist()
        assert (group.u - u0).tolist() == d.tolist()
        net.step()
        # the three neurons evolve differently from here
        assert group.v[0] != group.v[2]

    def test_matches_a_neuron_run_alone(self):
        a = torch.tensor([0.02, 0.1], dtype=torch.float64)
        both, g_both = make(2, a=a, d=torch.tensor([8.0, 2.0], dtype=torch.float64), dt=0.5)
        g_both.I.fill_(10.0)
        both.run(300)
        for neuron, (ai, di) in enumerate([(0.02, 8.0), (0.1, 2.0)]):
            alone, g = make(1, a=ai, d=di, dt=0.5)
            g.I.fill_(10.0)
            alone.run(300)
            assert g_both.v[neuron].item() == pytest.approx(g.v[0].item(), abs=1e-12)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"a": 0.0}, "a must be positive, got 0.0"),
            ({"a": torch.tensor([0.1, -0.1])}, "a must be positive"),
            ({"c": 30.0}, r"c must be less than v_peak \(30.0\), got 30.0"),
            ({"c": torch.tensor([-65.0, 40.0])}, "c must be less than v_peak"),
            ({"substeps": 0}, "substeps must be a positive integer, got 0"),
            ({"substeps": 1.5}, "substeps must be a positive integer, got 1.5"),
        ],
    )
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            Izhikevich(**{**RS, **kwargs})

    def test_parameter_of_the_wrong_shape(self):
        net = Network()
        NeuronGroup(net, 3, [Izhikevich(**{**RS, "b": torch.ones(2)})])
        with pytest.raises(ValueError, match=r"b must have shape \(3,\), got \(2,\)"):
            net.initialize()
        net = Network()
        NeuronGroup(net, 3, [Izhikevich(**RS, v_init=torch.zeros(4))])
        with pytest.raises(ValueError, match="v_init must have shape"):
            net.initialize()

    def test_reset_state_restores_initial_values(self):
        net, group = make(2)
        group.I.fill_(15.0)
        net.run(80)
        assert group.v[0].item() != -65.0
        net.reset_state()
        assert group.v.tolist() == [-65.0, -65.0]
        assert group.u.tolist() == pytest.approx([-13.0, -13.0])
        assert group.I.tolist() == [0.0, 0.0]
        assert not group.spikes.any()

    def test_fire_needs_the_model_and_is_driven_by_dendrite_integration(self):
        net = Network(dt=0.5, dtype=torch.float64)
        src = NeuronGroup(net, 4, [PoissonInput(0.2), Axon()])
        dst = NeuronGroup(
            net, 3, [DendriteStructure(), DendriteIntegration(), Izhikevich(**RS), Fire()]
        )
        SynapseGroup(
            net,
            src,
            dst,
            [WeightInit(mode="ones", scale=6.0), DenseInput(), SpikeGather()],
        )
        net.run(200)
        assert dst.spikes.dtype == torch.bool
        assert dst.v.abs().max() < 100
        assert torch.isfinite(dst.u).all()

    def test_batched_and_independent_networks(self):
        for independent in (False, True):
            net = Network(dt=1.0, dtype=torch.float64, batch_size=3, independent=independent)
            a = torch.full((3, 2), 0.02, dtype=torch.float64) if independent else 0.02
            group = NeuronGroup(net, 2, [Izhikevich(**{**RS, "a": a}), Fire()])
            net.initialize()
            assert group.v.shape == (3, 2)
            group.I.copy_(torch.tensor([[0.0], [10.0], [20.0]]).expand(3, 2))
            net.run(5)
            assert group.v[0, 0] < group.v[1, 0] < group.v[2, 0]

    def test_independent_members_with_their_own_a(self):
        a = torch.tensor([[0.02, 0.02], [0.1, 0.1]], dtype=torch.float64)
        net = Network(dt=1.0, dtype=torch.float64, batch_size=2, independent=True)
        group = NeuronGroup(net, 2, [Izhikevich(**{**RS, "a": a}), Fire()])
        net.initialize()
        group.I.fill_(10.0)
        net.run(20)
        assert not torch.equal(group.u[0], group.u[1])

    def test_works_with_the_extra_behaviors(self):
        net = Network(dt=0.5, dtype=torch.float64, seed=1)
        group = NeuronGroup(
            net,
            4,
            [
                SpikeTriggeredCurrent(-1.0, 20.0),
                Izhikevich(**RS),
                PoissonDrive(10, 0.05, 0.5),
                Fire(),
            ],
        )
        net.initialize()
        group.I.fill_(8.0)  # persists: nothing rebuilds it, so only the first step is exact
        net.run(50)
        assert torch.isfinite(group.v).all()


@pytest.mark.skipif(shutil.which("g++") is None, reason="torch.compile on the CPU needs g++")
def test_compiled_stepper_matches_eager_with_the_new_behaviors():
    def build():
        net = Network(dt=0.5, dtype=torch.float32, seed=5)
        drive = NeuronGroup(net, 6, [CorrelatedPoissonInput(0.05, 0.5), Axon()])
        group = NeuronGroup(
            net,
            4,
            [
                DendriteStructure(),
                DendriteIntegration(),
                SpikeTriggeredCurrent(-1.0, 20.0),
                Izhikevich(**RS),
                PoissonDrive(10, 0.05, 0.5),
                Fire(),
                Axon(),
            ],
        )
        SynapseGroup(
            net,
            drive,
            group,
            [WeightInit(mode="ones", scale=5.0), DenseInput(), SpikeGather()],
        )
        net.initialize()
        return net, group

    (ref, g_ref), (got, g_got) = build(), build()
    ref.run(40)
    CompiledStepper(got, cuda_graph=False).run(40)
    assert torch.allclose(g_ref.v, g_got.v, atol=1e-3)
    assert torch.equal(ref.generator.get_state(), got.generator.get_state())


def test_per_neuron_adaptive_elif_and_conductance_arguments_are_validated_elementwise():
    elif_args = {
        "tau": 10.0,
        "threshold": -50.0,
        "v_reset": -70.0,
        "v_rest": -65.0,
        "delta": 2.0,
        "theta_rh": -50.0,
    }
    with pytest.raises(ValueError, match="tau_w must be positive"):
        AdaptiveELIF(alpha=0.0, beta=0.0, tau_w=torch.tensor([1.0, 0.0]), **elif_args)
    with pytest.raises(ValueError, match="tau_exc must be positive"):
        ConductanceIntegration(tau_exc=torch.tensor([1.0, -1.0]))
    net = Network(dt=2.0)
    NeuronGroup(
        net,
        2,
        [
            ConductanceIntegration(tau_exc=5.0, tau_inh=torch.tensor([5.0, 1.0])),
            LIF(tau=10.0, threshold=-50.0, v_reset=-70.0, v_rest=-65.0),
        ],
    )
    with pytest.raises(ValueError, match=r"tau_inh .* must be at least dt \(2.0\)"):
        net.initialize()
    net = Network(dt=1.0)
    NeuronGroup(
        net,
        2,
        [
            ConductanceIntegration(tau_exc=torch.ones(3)),
            LIF(tau=10.0, threshold=-50.0, v_reset=-70.0, v_rest=-65.0),
        ],
    )
    with pytest.raises(ValueError, match=r"tau_exc must have shape \(2,\), got \(3,\)"):
        net.initialize()
