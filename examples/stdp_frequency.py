"""Frequency dependence of STDP: pair rule versus triplet rule.

Sjostrom, Turrigiano and Nelson (2001, Neuron 32:1149) paired a presynaptic and a
postsynaptic spike 60 times at different repetition frequencies. With the pair order fixed
the sign of the change depends on the frequency: at low frequency pre-before-post
potentiates and post-before-pre depresses, but at high frequency both orders potentiate.
A pair-based rule cannot do that (its change is the sum of independent pair contributions,
so it hardly depends on the frequency). Pfister and Gerstner (2006, J. Neurosci.
26:9673) showed that adding triplet terms reproduces it. This example reproduces the
protocol with :class:`~neurosush.synapses.plasticity.STDP` and
:class:`~neurosush.synapses.triplet.TripletSTDP`.

Protocol: one presynaptic and one postsynaptic neuron driven by prescribed spike trains,
dt = 1 ms; 60 pairings at 0.1, 10, 20, 40 and 50 Hz, post spike 10 ms after (``+10``) or
before (``-10``) the pre spike; the weight change is the total over the 60 pairings.

Triplet parameters: the minimal all-to-all visual cortex model of Pfister and Gerstner
(2006), Table 3 ("Visual cortex data set", Min., All-to-All; the hippocampal data are
Table 4): A2+ = 0, A3+ = 6.5e-3, A2- = 7.1e-3, A3- = 0, tau_y = 114 ms; tau_x is blank in
the table (unused since A3- = 0; the example passes 101, the full-model value);
tau_plus = 16.8 ms and tau_minus = 33.7 ms are fixed in the paper (from Bi and Poo 2001).
For comparison the minimal nearest-spike model has A3+ = 5e-2, A2- = 8e-3, tau_y = 40 ms,
and the full all-to-all model A2+ = 5e-10, A3+ = 6.2e-3, A2- = 7e-3, A3- = 2.3e-4,
tau_x = 101, tau_y = 125 ms. Verified against the open-access full text, PMC6674434
(https://pmc.ncbi.nlm.nih.gov/articles/PMC6674434/). Convention: the paper's weight changes
are relative (to the initial weight) and the data are normalized; this example applies the
amplitudes as absolute changes of a weight of 0.5, so only the shape in frequency and lag
is comparable, not the magnitudes. The pair rule uses a_plus=5e-3, a_minus=7.1e-3 with the
same time constants: illustrative values of mine, not the paper's pair fit.

Deviations: the paper's data are weight changes of real synapses with noisy spike times;
here the spikes are exact. At 0.1 Hz the interval between pairings is capped at 1 s (the
traces have then decayed to under 0.1 %, so this does not change the result).

Run: ``python examples/stdp_frequency.py``. Takes a few seconds.

Measured (change of the weight over 60 pairings, initial weight 0.5, no bounds):

    freq (Hz)   pair +10   pair -10   triplet +10   triplet -10
          0.1    +0.1624    -0.3152       +0.0000       -0.3152
           10    +0.1335    -0.3300       +0.1162       -0.3301
           20    +0.0096    -0.3765       +0.2207       -0.3426
           40    -0.2902    -0.4378       +0.5037       +0.1482
           50    -0.4427    -0.4582       +0.7184       +0.7052

The triplet rule turns depression into potentiation at 40-50 Hz for both lags, and
potentiates at +10 ms increasingly with frequency, as in the data. The pair rule never
potentiates more at high frequency; it only changes with frequency through pair overlaps
(a spike pairs with its neighbours' traces, so at 40-50 Hz the depression of the
post-before-pre pairs between pairings dominates and +10 ms ends up depressing too). With
a2_plus = 0 the triplet rule's potentiation at 0.1 Hz is essentially zero, whereas the
data show a small positive change.
"""

from __future__ import annotations

import argparse

import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.neurons.inputs import SpikeInput
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces
from neurosush.synapses.triplet import TripletSTDP

FREQUENCIES = (0.1, 10.0, 20.0, 40.0, 50.0)
LAGS = (10, -10)
PAIRINGS = 60
MAX_PERIOD = 1000  # ms


def spike_trains(freq: float, lag: int, pairings: int = PAIRINGS) -> tuple[torch.Tensor, ...]:
    """Boolean ``(steps, 1)`` spike trains of the pre and post neuron, dt = 1 ms.

    Pairing ``k`` has its pre spike at ``start + k * period`` and its post spike ``lag`` ms
    later (earlier for a negative lag).
    """
    period = min(round(1000.0 / freq), MAX_PERIOD)
    start = max(0, -lag) + 1
    steps = start + pairings * period + abs(lag) + 200
    pre = torch.zeros(steps, 1, dtype=torch.bool)
    post = torch.zeros(steps, 1, dtype=torch.bool)
    for k in range(pairings):
        pre[start + k * period] = True
        post[start + k * period + lag] = True
    return pre, post


def rule_behaviors(rule: str) -> list:
    """The synapse behaviors of ``"pair"`` or ``"triplet"`` plasticity."""
    if rule == "pair":
        return [Traces(tau_pre=16.8, tau_post=33.7), STDP(a_plus=5e-3, a_minus=7.1e-3)]
    if rule == "triplet":
        return [
            TripletSTDP(
                a2_plus=0.0,
                a3_plus=6.5e-3,
                a2_minus=7.1e-3,
                a3_minus=0.0,
                tau_plus=16.8,
                tau_minus=33.7,
                tau_x=101.0,
                tau_y=114.0,
                interaction="all",
            )
        ]
    raise ValueError(f"rule must be 'pair' or 'triplet', got {rule!r}")


def weight_change(rule: str, freq: float, lag: int, pairings: int = PAIRINGS) -> float:
    """Total weight change of one synapse over ``pairings`` pre/post pairings."""
    pre, post = spike_trains(freq, lag, pairings)
    net = Network(seed=0)
    src = NeuronGroup(net, 1, [SpikeInput(pre), Axon()], name="pre")
    dst = NeuronGroup(net, 1, [SpikeInput(post), Axon()], name="post")
    syn = SynapseGroup(
        net,
        src,
        dst,
        [
            WeightInit(weights=torch.full((1, 1), 0.5)),
            DenseInput(),
            SpikeGather(),
            *rule_behaviors(rule),
        ],
    )
    net.initialize()
    net.run(len(pre))
    return float(syn.weights.item() - 0.5)


def run(frequencies=FREQUENCIES, lags=LAGS, pairings: int = PAIRINGS) -> dict:
    """Weight changes ``{(rule, freq, lag): dw}``."""
    return {
        (rule, f, lag): weight_change(rule, f, lag, pairings)
        for rule in ("pair", "triplet")
        for f in frequencies
        for lag in lags
    }


def main() -> None:
    """Print the weight change table."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pairings", type=int, default=PAIRINGS)
    args = p.parse_args()
    result = run(pairings=args.pairings)
    print(f"weight change over {args.pairings} pairings (lag in ms, pre before post is +)")
    header = ("freq (Hz)", "pair +10", "pair -10", "triplet +10", "triplet -10")
    print(" ".join(f"{h:>12}" for h in header))
    for f in FREQUENCIES:
        row = [f"{result[(r, f, lag)]:+.4f}" for r in ("pair", "triplet") for lag in LAGS]
        print(f"{f:>12g} " + " ".join(f"{v:>12}" for v in row))


if __name__ == "__main__":
    main()
