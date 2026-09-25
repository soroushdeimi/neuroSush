import itertools
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from neurosush import checkpoint
from neurosush.core.behavior import Behavior
from neurosush.core.graph import GraphStepper
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order
from neurosush.modulation import Dopamine, Payoff
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import KWTA, InherentNoise
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.homeostasis import ActivityHomeostasis
from neurosush.neurons.inputs import SpikeInput
from neurosush.neurons.models import LIF, Fire
from neurosush.recording import Recorder
from neurosush.synapses.constraints import WeightNormalization
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces

pytestmark = pytest.mark.gpu


def _frames(shape):
    """A few fixed Poisson-like frames, independent of the network's own seed."""
    generator = torch.Generator().manual_seed(123)
    return [torch.rand(shape, generator=generator) < 0.3 for _ in range(4)]


def build(device, batch_size=None, seed=0, target_extra=()):
    """A dense, plastic, homeostatic network exercising every graph-ready behavior.

    A short 7-step homeostasis window with decay 0.9 makes several windows end over the
    length of a test run, so both of the network's two graphs (window-ending and not) get
    captured.
    """
    net = Network(device=device, seed=seed, batch_size=batch_size)
    shape = (10,) if batch_size is None else (batch_size, 10)
    src = NeuronGroup(net, 10, [SpikeInput(itertools.cycle(_frames(shape))), Axon()], name="src")
    dst = NeuronGroup(
        net,
        6,
        [
            DendriteStructure(),
            DendriteIntegration(),
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            KWTA(k=2),
            Fire(),
            ActivityHomeostasis(target_spikes=2, window=7, rate=0.5, decay=0.9),
            Axon(),
            Recorder("v"),
            *target_extra,
        ],
        name="dst",
    )
    SynapseGroup(
        net,
        src,
        dst,
        [
            WeightInit(mode="uniform"),
            DenseInput(coef=3.0),
            SpikeGather(),
            Traces(tau_pre=5.0),
            STDP(a_plus=0.02, a_minus=0.01, bound="soft"),
            WeightNormalization(norm=5.0),
        ],
        name="syn",
    )
    net.initialize()
    return net


def _equal(a, b):
    """Recursively ``torch.equal``, through the dicts a checkpoint stores buffers/behaviors as."""
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return a == b


class TestExactness:
    @pytest.mark.parametrize("batch_size", [None, 4])
    def test_matches_eager_bit_for_bit(self, batch_size):
        eager = build("cuda", batch_size, seed=1)
        eager.run(60)

        graphed = build("cuda", batch_size, seed=1)
        GraphStepper(graphed).run(60)

        expected, got = checkpoint.state_dict(eager), checkpoint.state_dict(graphed)
        assert expected.keys() == got.keys()
        for key in expected:
            assert _equal(expected[key], got[key]), key


class TestGraphCount:
    def test_captures_one_graph_per_key(self):
        # steps that end a 7-step homeostasis window, and steps that do not: two graphs
        stepper = GraphStepper(build("cuda", seed=2))
        stepper.run(60)
        assert len(stepper._graphs) == 2


class TestRecorder:
    def test_records_the_same_values_as_eager(self):
        eager = build("cuda", seed=3)
        eager.run(40)

        graphed = build("cuda", seed=3)
        GraphStepper(graphed).run(40)

        eager_v = eager.groups[1].behaviors[-1].get("v")
        graphed_v = graphed.groups[1].behaviors[-1].get("v")
        assert torch.equal(eager_v, graphed_v)


class TestDisabling:
    def test_disabling_a_behavior_captures_a_new_graph_and_matches_eager(self):
        eager, graphed = build("cuda", seed=4), build("cuda", seed=4)
        stepper = GraphStepper(graphed)

        eager.run(20)
        stepper.run(20)
        assert len(stepper._graphs) == 2

        eager.synapses[0].behaviors[-1].enabled = False  # WeightNormalization
        graphed.synapses[0].behaviors[-1].enabled = False

        eager.run(20)
        stepper.run(20)
        assert len(stepper._graphs) > 2

        assert _equal(checkpoint.state_dict(eager), checkpoint.state_dict(graphed))


class _RandomSpikes(Behavior):
    """Fires with probability ``p``, drawing from the network generator.

    Unlike :class:`~neurosush.neurons.inputs.SpikeInput`, this keeps no un-checkpointed
    Python state (an iterator position), so a network driven by it resumes exactly from a
    checkpoint; :class:`~neurosush.neurons.inputs.SpikeInput` itself is checkpoint-friendly
    only within a single continuous run, since its frame stream is not saved.
    """

    order = Order.FIRE
    graph_safe = True

    def __init__(self, p):
        self.p = p

    def initialize(self, group):
        group.spikes = group.state(False, dtype=torch.bool)

    def forward(self, group):
        group.spikes = group.rand() < self.p


def build_resumable(device, seed=0):
    """Like :func:`build`, but with an input whose whole state a checkpoint saves."""
    net = Network(device=device, seed=seed)
    src = NeuronGroup(net, 10, [_RandomSpikes(0.3), Axon()], name="src")
    dst = NeuronGroup(
        net,
        6,
        [
            DendriteStructure(),
            DendriteIntegration(),
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            KWTA(k=2),
            Fire(),
            ActivityHomeostasis(target_spikes=2, window=7, rate=0.5, decay=0.9),
            Axon(),
        ],
        name="dst",
    )
    SynapseGroup(
        net,
        src,
        dst,
        [
            WeightInit(mode="uniform"),
            DenseInput(coef=3.0),
            SpikeGather(),
            Traces(tau_pre=5.0),
            STDP(a_plus=0.02, a_minus=0.01, bound="soft"),
            WeightNormalization(norm=5.0),
        ],
        name="syn",
    )
    net.initialize()
    return net


class TestCheckpointContinuation:
    def test_resuming_after_graph_steps_continues_exactly(self, tmp_path):
        reference = build_resumable("cuda", seed=5)
        reference.run(60)

        first = build_resumable("cuda", seed=5)
        GraphStepper(first).run(25)
        checkpoint.save(first, tmp_path / "net.pt")

        resumed = build_resumable("cuda", seed=5)
        checkpoint.load(resumed, tmp_path / "net.pt")
        resumed.run(35)

        assert resumed.iteration == 60
        assert _equal(checkpoint.state_dict(resumed), checkpoint.state_dict(reference))


class TestErrors:
    def test_cpu_network_raises(self):
        net = Network()
        NeuronGroup(net, 3)
        with pytest.raises(ValueError, match="cuda"):
            GraphStepper(net)

    def test_payoff_and_dopamine_are_named(self):
        net = Network(device="cuda", behaviors=[Payoff(lambda n: 0.0), Dopamine(tau=5.0)])
        with pytest.raises(ValueError, match=r"Payoff on the network.*Dopamine on the network"):
            GraphStepper(net)

    def test_axon_with_a_delay_is_named_not_ready(self):
        net = Network(device="cuda")
        group = NeuronGroup(net, 3, [Axon(max_delay=2)])
        with pytest.raises(ValueError, match=rf"Axon on {group.name}"):
            GraphStepper(net)


class TestInherentNoise:
    """InherentNoise draws from net.generator every step; whether a replay reproduces the
    eager result depends on how the installed torch handles a graph-registered generator.
    """

    def _network(self, seed):
        net = Network(device="cuda", seed=seed)
        NeuronGroup(net, 8, [SpikeInput(itertools.cycle(_frames((8,)))), Axon()], name="src")
        NeuronGroup(
            net,
            5,
            [
                DendriteStructure(),
                DendriteIntegration(),
                LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
                InherentNoise(scale=1.5, distribution="normal"),
                Fire(),
                Axon(),
            ],
            name="dst",
        )
        net.initialize()
        return net

    def test_reproducible_and_matches_eager_when_ready(self):
        probe = self._network(seed=0)
        noise = probe.groups[1].behaviors[3]
        if not noise.graph_ready(probe.groups[1]):
            pytest.skip("InherentNoise is not graph-ready on this torch build")

        eager = self._network(seed=11)
        eager.run(50)

        graphed = self._network(seed=11)
        GraphStepper(graphed).run(50)
        # confirmed to match on the installed torch: register_generator_state (even as the
        # no-op it now is) leaves the generator's Philox offset advancing exactly as it would
        # for the same rand() calls run eagerly, so replays reproduce the eager sequence.
        assert torch.equal(eager.groups[1].v, graphed.groups[1].v)
        assert torch.equal(eager.groups[1].spikes, graphed.groups[1].spikes)

        second = self._network(seed=11)
        GraphStepper(second).run(50)
        assert torch.equal(graphed.groups[1].v, second.groups[1].v)


# a failed capture leaves PyTorch's CUDA allocator unusable in that process, so this check
# runs in a child process
SYNCING = """
import torch
from neurosush.core.behavior import Behavior
from neurosush.core.graph import GraphStepper
from neurosush.core.network import Network, NeuronGroup
from neurosush.core.order import Order
from neurosush.neurons.models import LIF, Fire


class Syncing(Behavior):
    order = Order.NEURON_DYNAMICS
    graph_safe = True  # wrongly: forward reads a value back to the host

    def forward(self, group):
        if group.v.sum().item() > 1e9:
            group.v = group.v * 0


net = Network(device="cuda")
lif = LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0)
NeuronGroup(net, 4, [lif, Syncing(), Fire()], name="g")
stepper = GraphStepper(net, warmup=1)
stepper.step()
try:
    stepper.step()
except RuntimeError as error:
    print(error, "| cause:", type(error.__cause__).__name__)
"""


def test_a_behavior_that_breaks_capture_is_named():
    src = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run(
        [sys.executable, "-c", SYNCING],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "Syncing on g failed while capturing a CUDA graph" in result.stdout
    assert "restart the process" in result.stdout
    assert "| cause: " in result.stdout
