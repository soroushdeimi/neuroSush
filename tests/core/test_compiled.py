"""CompiledStepper matches eager stepping to a tolerance, on CPU and CUDA."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from neurosush import checkpoint
from neurosush.core.behavior import Behavior
from neurosush.core.compiled import CompiledStepper
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import InherentNoise
from neurosush.neurons.dendrite import ConductanceIntegration
from neurosush.neurons.inputs import PoissonInput
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.traces import SpikeGather

from .test_graph import build as build_kwta

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "diehl_cook_mnist.py"
CUDA = pytest.param("cuda", marks=[pytest.mark.gpu])
DEVICES = ["cpu", CUDA]


def _dc():
    spec = importlib.util.spec_from_file_location("diehl_cook_mnist", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["diehl_cook_mnist"] = module
    spec.loader.exec_module(module)
    return module


def _skip_without_a_compiler(device):
    if device == "cpu":
        import shutil

        if shutil.which("g++") is None and shutil.which("c++") is None:
            pytest.skip("torch.compile on the CPU needs a C++ compiler")


def dc_model(device, *, rule="pair", members=None, seed=0):
    model = _dc().build_network(40, device, rule=rule, members=members, seed=seed)
    model.input.rates.fill_(0.06)
    return model


def trace(model, stepper, steps):
    """Spikes of every group at every step, then the final state."""
    groups = model.net.groups
    spikes = []
    for _ in range(steps):
        stepper.step()
        spikes.append(torch.cat([g.spikes.flatten() for g in groups]).clone())
    return torch.stack(spikes)


def assert_close_state(a, b, tol=1e-4):
    ea, eb = checkpoint.state_dict(a), checkpoint.state_dict(b)
    assert ea.keys() == eb.keys()
    for key in ea:
        x, y = ea[key], eb[key]
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            assert torch.allclose(x, y, rtol=tol, atol=tol), key


def compare(make, steps, device):
    _skip_without_a_compiler(device)
    ref, got = make(), make()
    expected = trace(ref, ref.net, steps)
    stepper = CompiledStepper(got.net)
    actual = trace(got, stepper, steps)
    assert expected.any(), "the network never spiked: a vacuous comparison"
    assert torch.equal(expected, actual)
    return ref, got, stepper


CASES = [
    pytest.param("cpu", 2, id="cpu-independent"),
    pytest.param("cuda", None, id="cuda", marks=pytest.mark.gpu),
    pytest.param("cuda", 2, id="cuda-independent", marks=pytest.mark.gpu),
]


@pytest.mark.parametrize(("device", "members"), CASES)
@pytest.mark.parametrize("rule", ["pair", "triplet"])
def test_diehl_cook_matches_eager(device, rule, members):
    steps = 300 if device == "cuda" else 120
    ref, got, _ = compare(lambda: dc_model(device, rule=rule, members=members), steps, device)
    assert torch.allclose(ref.exc.v, got.exc.v, atol=1e-4)
    assert torch.allclose(ref.syn.weights, got.syn.weights, atol=1e-4)
    assert torch.allclose(ref.exc.theta, got.exc.theta, atol=1e-4)


def test_cpu_refuses_the_event_driven_unbatched_stdp():
    with pytest.raises(ValueError, match=r"STDP on input->exc"):
        CompiledStepper(dc_model("cpu").net)


@pytest.mark.gpu
def test_cuda_captures_one_graph():
    _, _, stepper = compare(lambda: dc_model("cuda"), 40, "cuda")
    assert len(stepper._graphs) == 1


@pytest.mark.gpu
def test_graph_keys_capture_and_replay_correctly():
    # a 7-step homeostasis window: window-ending and ordinary steps are two graphs
    def make():
        return build_kwta("cuda", None, seed=2)

    ref, got = make(), make()
    ref.run(80)
    stepper = CompiledStepper(got)
    stepper.run(80)
    assert len(stepper._graphs) == 2
    assert_close_state(ref, got)
    # disabling a behavior is a new key: two more graphs
    for net in (ref, got):
        net.synapses[0].behaviors[-1].enabled = False
    ref.run(30)
    stepper.run(30)
    assert len(stepper._graphs) == 4  # window-ending or not, each
    assert_close_state(ref, got)


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [None, 4])
def test_kwta_homeostasis_network_matches_eager(batch_size):
    ref, got = build_kwta("cuda", batch_size, seed=1), build_kwta("cuda", batch_size, seed=1)
    spikes = []
    stepper = CompiledStepper(got)
    for _ in range(120):
        ref.step()
        stepper.step()
        spikes.append(torch.equal(ref.groups[1].spikes, got.groups[1].spikes))
    assert all(spikes)
    assert torch.allclose(ref.groups[1].v, got.groups[1].v, atol=1e-4)
    assert_close_state(ref, got)


@pytest.mark.parametrize("device", DEVICES)
def test_reset_state_between_samples_matches_eager(device):
    _skip_without_a_compiler(device)
    ref, got = dc_model(device, members=2, seed=3), dc_model(device, members=2, seed=3)
    stepper = CompiledStepper(got.net)
    for _ in range(4):
        ref.net.reset_state()
        got.net.reset_state()
        a, b = trace(ref, ref.net, 30), trace(got, stepper, 30)
        assert torch.equal(a, b)
    assert torch.allclose(ref.exc.v, got.exc.v, atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_random_stream_is_the_eager_stream(device):
    _skip_without_a_compiler(device)
    ref, got = dc_model(device, members=2, seed=4), dc_model(device, members=2, seed=4)
    CompiledStepper(got.net).run(10)
    ref.net.run(10)
    assert torch.equal(ref.net.generator.get_state(), got.net.generator.get_state())


def _noisy(device, max_delay=1):
    net = Network(device=device, seed=1)
    lif = {"tau": 10.0, "threshold": -55.0, "v_reset": -70.0, "v_rest": -65.0}
    src = NeuronGroup(net, 8, [PoissonInput(0.3), Axon(max_delay=max_delay)], name="src")
    dst = NeuronGroup(
        net,
        5,
        [
            ConductanceIntegration(),
            InherentNoise(scale=2.0, distribution="normal"),
            LIF(**lif),
            Fire(),
            Axon(),
        ],
        name="dst",
    )
    SynapseGroup(
        net, src, dst, [WeightInit(mode="uniform", scale=3.0), DenseInput(), SpikeGather()]
    )
    net.initialize()
    return net


@pytest.mark.parametrize("device", DEVICES)
def test_inherent_noise_uses_predrawn_samples(device):
    _skip_without_a_compiler(device)
    ref, got = _noisy(device), _noisy(device)
    ref.run(60)
    CompiledStepper(got).run(60)
    assert torch.allclose(ref.groups[1].v, got.groups[1].v, atol=1e-4)
    assert torch.equal(ref.groups[1].spikes, got.groups[1].spikes)
    assert torch.equal(ref.generator.get_state(), got.generator.get_state())


def test_refuses_delays_and_names_the_behaviors():
    net = _noisy("cpu", max_delay=3)
    with pytest.raises(ValueError, match=r"Axon on src"):
        CompiledStepper(net)


def test_refuses_behaviors_that_are_not_ready():
    class Syncs(Behavior):
        order = Order.NEURON_DYNAMICS

    net = _noisy("cpu")
    group = net.groups[1]
    behavior = Syncs()
    net._registrations.append((group, behavior))
    net.schedule = sorted(net._registrations, key=lambda pair: pair[1].order)
    with pytest.raises(ValueError, match=r"Syncs on dst"):
        CompiledStepper(net)


def test_eager_forward_without_drawn_numbers_is_unchanged():
    a, b = _noisy("cpu"), _noisy("cpu")
    a.run(20)
    b.run(20)
    assert torch.equal(a.groups[1].v, b.groups[1].v)
    assert PoissonInput().drawn == {}


@pytest.mark.parametrize("kind", ["src_delay", "dst_delay"])
def test_negative_delays_are_rejected_at_initialize(kind):
    net = Network(device="cpu", seed=1)
    src = NeuronGroup(net, 4, [PoissonInput(0.3), Axon()], name="src")
    dst = NeuronGroup(
        net,
        3,
        [
            ConductanceIntegration(),
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            Fire(),
            Axon(),
        ],
        name="dst",
    )
    syn = SynapseGroup(net, src, dst, [WeightInit(mode="uniform"), DenseInput(), SpikeGather()])
    delays = torch.zeros(4 if kind == "src_delay" else 3, dtype=torch.long)
    delays[0] = -1
    setattr(syn, kind, delays)
    with pytest.raises(ValueError, match=rf"{kind} must not be negative for"):
        net.initialize()


def test_attribute_created_after_construction_is_reported():
    class Lazy(Behavior):
        order = Order.NEURON_DYNAMICS
        graph_safe = True

        def forward(self, group):
            group.late = group.v + 1

    net = _noisy("cpu")
    group = net.groups[1]
    net._registrations.append((group, Lazy()))
    net.schedule = sorted(net._registrations, key=lambda pair: pair[1].order)
    with pytest.raises(RuntimeError, match=r"dst\.late"):
        CompiledStepper(net).step()
