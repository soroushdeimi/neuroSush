"""Throughput of a 784 -> 400 network with dense STDP (MNIST-sized).

Poisson-like input, LIF outputs with k-winners-take-all and homeostasis, soft-bounded
STDP and weight normalization. Reports simulation steps per second and sample-steps per
second (steps times batch size).
Run: ``python benchmarks/dense_stdp.py --device cuda --batch 64``.
"""

from __future__ import annotations

import argparse
import itertools
import time

import torch

from neurosush.core.graph import GraphStepper
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import KWTA
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.homeostasis import ActivityHomeostasis
from neurosush.neurons.inputs import SpikeInput
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses.constraints import WeightNormalization
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces


def build(device: str, batch: int | None = None, inputs: int = 784, outputs: int = 400) -> Network:
    """The benchmark network, initialized."""
    net = Network(device=device, seed=0, batch_size=batch)
    shape = (inputs,) if batch is None else (batch, inputs)
    frame = torch.rand(shape, generator=torch.Generator().manual_seed(0)) < 0.05
    source = NeuronGroup(net, inputs, [SpikeInput(itertools.repeat(frame)), Axon()])
    target = NeuronGroup(
        net,
        outputs,
        [
            DendriteStructure(),
            DendriteIntegration(),
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            KWTA(k=5),
            Fire(),
            ActivityHomeostasis(target_spikes=5, window=100, rate=0.1),
            Axon(),
        ],
    )
    SynapseGroup(
        net,
        source,
        target,
        [
            WeightInit(mode="uniform"),
            DenseInput(coef=2.0),
            SpikeGather(),
            Traces(tau_pre=10.0),
            STDP(a_plus=0.01, a_minus=0.005, bound="soft"),
            WeightNormalization(norm=inputs / 10),
        ],
    )
    net.initialize()
    return net


def steps_per_second(
    device: str, steps: int, batch: int | None = None, *, graph: bool = False
) -> float:
    """Measured simulation speed after a short warm-up.

    Args:
        device: Device to build the network on.
        steps: Number of timed steps.
        batch: Samples simulated in parallel, or None for unbatched.
        graph: Replay a captured CUDA graph instead of stepping eagerly.
    """
    net = build(device, batch)
    stepper = GraphStepper(net) if graph else net
    stepper.run(20)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    stepper.run(steps)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return steps / (time.perf_counter() - start)


def main() -> None:
    """Print steps per second for the chosen device."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=None, help="samples in parallel")
    parser.add_argument(
        "--graph", action="store_true", help="replay a captured CUDA graph instead of stepping"
    )
    args = parser.parse_args()
    rate = steps_per_second(args.device, args.steps, args.batch, graph=args.graph)
    samples = rate * (args.batch or 1)
    print(f"{args.device} batch={args.batch}: {rate:.0f} steps/s, {samples:.0f} sample-steps/s")


if __name__ == "__main__":
    main()
