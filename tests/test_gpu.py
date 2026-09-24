"""The same simulation on a GPU gives the CPU's results (skipped without CUDA)."""

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.htm.active_dendrites import ActiveDendrites
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.inputs import SpikeInput
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces

pytestmark = pytest.mark.gpu


def simulate(device, batch_size):
    """A deterministic plastic network: fixed input frames and fixed initial weights."""
    frames = (
        torch.rand(
            40,
            *(() if batch_size is None else (batch_size,)),
            30,
            generator=torch.Generator().manual_seed(0),
        )
        < 0.2
    )
    weights = torch.rand(30, 5, generator=torch.Generator().manual_seed(1))
    net = Network(device=device, batch_size=batch_size)
    src = NeuronGroup(net, 30, [SpikeInput(iter(frames.to(device))), Axon()])
    dst = NeuronGroup(
        net,
        5,
        [
            DendriteStructure(),
            DendriteIntegration(),
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            Fire(),
            Axon(),
        ],
    )
    syn = SynapseGroup(
        net,
        src,
        dst,
        [
            WeightInit(weights=weights.to(device)),
            DenseInput(coef=4.0),
            SpikeGather(),
            Traces(tau_pre=5.0),
            STDP(a_plus=0.02, a_minus=0.01, bound="soft"),
        ],
    )
    net.run(40)
    return dst.v.cpu(), syn.weights.cpu()


@pytest.mark.parametrize("batch_size", [None, 4])
def test_plastic_network_matches_the_cpu(batch_size):
    cpu_v, cpu_w = simulate("cpu", batch_size)
    gpu_v, gpu_w = simulate("cuda", batch_size)
    torch.testing.assert_close(gpu_v, cpu_v, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gpu_w, cpu_w, atol=1e-5, rtol=1e-4)


def test_active_dendrites_match_the_cpu():
    torch.manual_seed(0)
    layer = ActiveDendrites(8, 6, 3, segments=4, k=3)
    x, context = torch.randn(10, 8), torch.randn(10, 3)
    expected = layer(x, context)
    got = layer.to("cuda")(x.cuda(), context.cuda()).cpu()
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
