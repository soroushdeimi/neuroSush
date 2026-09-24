"""Two output neurons learn to tell two input patterns apart with STDP.

Twenty input neurons fire as Poisson trains: pattern A drives the first ten, pattern B the
last ten. Two LIF output neurons compete through k-winners-take-all, keep their firing rate
with activity homeostasis, and learn with soft-bounded STDP plus weight normalization, which
makes the inputs of each neuron compete. After training each output neuron fires for one
pattern only.

Run: ``python examples/two_patterns.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.data import spike_frames
from neurosush.encoding import rate_poisson
from neurosush.neurons.axon import Axon
from neurosush.neurons.competition import KWTA
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.homeostasis import ActivityHomeostasis
from neurosush.neurons.inputs import SpikeInput
from neurosush.neurons.models import LIF, Fire
from neurosush.synapses.constraints import WeightClip, WeightNormalization
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces

INPUTS = 20
PRESENT = 40  # steps per pattern presentation
SILENCE = 10  # silent steps after each presentation
RATE = 0.25  # spike probability per step of an active input


@dataclass
class Model:
    """The network and the objects the example inspects."""

    net: Network
    input: NeuronGroup
    output: NeuronGroup
    synapse: SynapseGroup
    steps: int


@dataclass
class Result:
    """Output spike counts per pattern, from the second half of training."""

    counts: dict[str, torch.Tensor] = field(default_factory=dict)

    def preferred(self) -> list[str]:
        """The pattern each output neuron fires for most."""
        a, b = self.counts["A"], self.counts["B"]
        return ["A" if a[j] > b[j] else "B" for j in range(len(a))]

    def selectivity(self) -> list[float]:
        """Fraction of each neuron's spikes that fall on its preferred pattern."""
        a, b = self.counts["A"], self.counts["B"]
        return [max(a[j], b[j]).item() / max(1.0, (a[j] + b[j]).item()) for j in range(len(a))]


def patterns() -> dict[str, torch.Tensor]:
    """Firing probability of every input neuron for each pattern."""
    a = torch.zeros(INPUTS)
    a[: INPUTS // 2] = RATE
    return {"A": a, "B": RATE - a}


def build(seed: int = 0, rounds: int = 40) -> Model:
    """Network with ``rounds`` presentations of A then B."""
    generator = torch.Generator().manual_seed(seed)
    samples = [
        (rate_poisson(rates, PRESENT, generator=generator), label)
        for _ in range(rounds)
        for label, rates in patterns().items()
    ]
    net = Network(seed=seed)
    source = NeuronGroup(
        net, INPUTS, [SpikeInput(spike_frames(samples, silence=SILENCE)), Axon()], name="input"
    )
    output = NeuronGroup(
        net,
        2,
        [
            DendriteStructure(),
            DendriteIntegration(),
            LIF(tau=10.0, threshold=-55.0, v_reset=-70.0, v_rest=-65.0),
            KWTA(k=1),
            Fire(),
            ActivityHomeostasis(target_spikes=6, window=200, rate=0.3),
            Axon(),
        ],
        name="output",
    )
    synapse = SynapseGroup(
        net,
        source,
        output,
        [
            WeightInit(mode="uniform", scale=0.4, offset=0.3),
            DenseInput(coef=8.0),
            SpikeGather(),
            Traces(tau_pre=10.0),
            STDP(a_plus=0.05, a_minus=0.01, bound="soft"),
            WeightNormalization(norm=INPUTS / 2),
            WeightClip(w_min=0.0, w_max=1.0),
        ],
    )
    return Model(net, source, output, synapse, rounds * 2 * (PRESENT + SILENCE))


def run(seed: int = 0, rounds: int = 40) -> Result:
    """Train, counting output spikes per pattern over the second half."""
    model = build(seed, rounds)
    result = Result({"A": torch.zeros(2), "B": torch.zeros(2)})
    for step in range(model.steps):
        model.net.step()
        label = model.input.label
        if step >= model.steps // 2 and label is not None:
            result.counts[label] += model.output.spikes.float()
    return result


def main() -> None:
    """Train once and print what each output neuron learned."""
    result = run()
    for j, (pattern, share) in enumerate(
        zip(result.preferred(), result.selectivity(), strict=True)
    ):
        print(f"output neuron {j}: fires for pattern {pattern} ({share:.0%} of its spikes)")


if __name__ == "__main__":
    main()
