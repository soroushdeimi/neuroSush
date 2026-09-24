import pytest
import torch

from neurosush import checkpoint
from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order
from neurosush.modulation import Dopamine, Payoff
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import InherentNoise
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.homeostasis import ActivityHomeostasis
from neurosush.neurons.models import LIF, Fire
from neurosush.recording import Recorder
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import DelayInit, WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces


class RandomSpikes(Behavior):
    """Fires with probability ``p``, drawing from the network generator."""

    order = Order.FIRE

    def __init__(self, p):
        self.p = p

    def initialize(self, group):
        group.spikes = group.state(False, dtype=torch.bool)

    def forward(self, group):
        group.spikes = group.rand() < self.p


def build(batch_size=None):
    """A network with every kind of state: randomness, delays, dendritic buffers,
    traces, plasticity, homeostasis and reward."""
    net = Network(
        seed=0,
        batch_size=batch_size,
        behaviors=[Payoff(lambda n: float(n.iteration % 3 == 0)), Dopamine(tau=5.0)],
    )
    src = NeuronGroup(net, 6, [RandomSpikes(0.3), Axon(max_delay=3)], name="src")
    dst = NeuronGroup(
        net,
        4,
        [
            DendriteStructure(distal_depth=3),
            DendriteIntegration(),
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            InherentNoise(scale=2.0),
            Fire(),
            ActivityHomeostasis(target_spikes=2, window=7, rate=0.5, decay=0.9),
            Axon(max_delay=3),
            Recorder("v"),
        ],
        name="dst",
    )
    SynapseGroup(
        net,
        src,
        dst,
        [
            WeightInit(mode="uniform", scale=0.5, offset=0.5),
            DelayInit(delays=torch.tensor([0, 1, 2, 0, 1, 2]), side="src"),
            DelayInit(delays=torch.tensor([0, 1, 2, 1]), side="dst"),
            DenseInput(coef=15.0),
            SpikeGather(),
            Traces(tau_pre=5.0),
            STDP(a_plus=0.05, a_minus=0.02, bound="soft"),
        ],
        compartment="distal",
        name="syn",
    )
    return net


def recorded(net):
    return net.groups[1].behaviors[-1].get("v")


@pytest.mark.parametrize("batch_size", [None, 3])
def test_resuming_continues_exactly(tmp_path, batch_size):
    reference = build(batch_size)
    reference.run(60)

    first = build(batch_size)
    first.run(25)
    checkpoint.save(first, tmp_path / "net.pt")
    resumed = build(batch_size)
    checkpoint.load(resumed, tmp_path / "net.pt")
    resumed.run(35)

    assert resumed.iteration == 60
    assert torch.equal(recorded(resumed), recorded(reference)[25:])
    for name in ("weights", "pre_trace", "post_trace"):
        assert torch.equal(getattr(resumed.synapses[0], name), getattr(reference.synapses[0], name))
    homeostasis = resumed.groups[1].behaviors[5]
    assert homeostasis.rate == reference.groups[1].behaviors[5].rate
    assert resumed.dopamine == reference.dopamine


def test_state_covers_buffers_behaviors_and_scalars():
    net = build()
    net.run(3)
    state = checkpoint.state_dict(net)
    assert state["iteration"] == 3
    assert set(state["groups.dst.dendrite"]) == {"proximal", "distal", "apical"}
    assert set(state["groups.src.spike_history"]) == {"storage", "head"}
    assert set(state["groups.dst.ActivityHomeostasis#5"]) == {"activity", "rate"}
    assert "net.dopamine" in state
    assert "synapses.syn.weights" in state
    # copies, not references
    state["synapses.syn.weights"].zero_()
    assert net.synapses[0].weights.abs().sum() > 0


def test_strict_loading_reports_mismatched_keys():
    net = build()
    net.run(1)
    state = checkpoint.state_dict(net)
    del state["synapses.syn.weights"]
    state["groups.extra.v"] = torch.zeros(1)
    with pytest.raises(
        KeyError, match=r"missing \['synapses.syn.weights'\], unexpected \['groups.extra.v'\]"
    ):
        checkpoint.load_state_dict(build(), state)
    checkpoint.load_state_dict(build(), state, strict=False)


def test_shapes_must_match():
    small, large = build(), build(batch_size=2)
    small.run(1)
    with pytest.raises(ValueError, match=r"^groups\.src\.\w+: .*shape"):
        checkpoint.load_state_dict(large, checkpoint.state_dict(small))


def test_saving_needs_an_initialized_network():
    with pytest.raises(RuntimeError, match="initialize"):
        checkpoint.state_dict(build())


def test_behavior_without_state_rejects_some():
    with pytest.raises(KeyError, match="keeps no state"):
        Fire().load_state_dict({"x": 1})
