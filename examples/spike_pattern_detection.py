"""STDP finds the start of a repeating spike pattern in continuous noise.

Masquelier, Guyonneau and Thorpe (2008, PLoS ONE 3:e1377) showed that a neuron with
STDP, fed by many afferents firing at time-varying Poisson-like rates, becomes selective
to a 50 ms spike pattern that is copy-pasted at random times into the otherwise
unstructured input: the afferents that fire early in the pattern are potentiated, the
others depressed, and the neuron's response latency shrinks until it fires at the very
start of the pattern, while staying silent outside it. This example reproduces that with
a :class:`~neurosush.neurons.models.LIF` neuron and the library
:class:`~neurosush.synapses.plasticity.STDP`.

Setup (dt = 1 ms), following the paper: 2000 afferents, each with a rate that performs a
bounded random walk in 0-90 Hz (the rate changes with a speed ``s`` that itself does a
random walk, clipped to +-1800 Hz/s), plus 10 Hz spontaneous noise. A 50 ms pattern (the
spikes of 1000 afferents over one 50 ms stretch, drawn once) is pasted at random times
(gaps of 50 ms + exponential of mean 100 ms, so 25 % of the time, never twice in a row);
the other 1000 afferents just keep their noise during the pattern, and all afferents keep
their random-walk rate, so the population rate is the same in and out of the pattern.
Synapses are all excitatory with equal initial weights 0.475, hard bounds [0, 1].

Parameters checked in the paper (fetched text of the PLoS ONE article via WebFetch; one
automatic summary, not read line by line, so treat them as "as reported there"):
a+ = 0.03125 (2^-5), a- = 0.85 a+, tau+ = 16.8 ms, tau- = 33.7 ms, initial weight 0.475,
SRM neuron with tau_m = 10 ms, tau_s = 2.5 ms, threshold 500 (arbitrary units), 1 ms
refractory period, simulation 450 s, reported 96 % success over 100 runs, final latency
about 4 ms, hit rate 99.1 % with no false alarms. Not verified: the exact rate-change
procedure (I use my own random-walk speed) and the spike afterpotential constants
(K1 = 2, K2 = 4 in the summary).

Deviations: (1) the neuron is a LIF (tau = 10 ms, exponential synaptic current with tau = 2.5
ms through ``DendriteIntegration``) with its own threshold (calibrated by me, the paper's units
differ), not an SRM; its negative afterpotential is replaced by a 1 ms refractory period and a
slow after-spike current (see 6). (2) The paper uses the nearest-spike approximation: the
traces here are ``Traces(interaction="nearest")`` (reset to 1 at a spike, so a potentiation
uses the latest presynaptic spike only) and the library STDP uses ``pairing="nearest"``, so a
presynaptic spike depresses only if the neuron fired since that afferent's previous spike (an
afferent is depressed once per output spike). (3) With the default all-to-all pairing, every
presynaptic spike after a postsynaptic one depresses: the weights of 64 Hz afferents lose about
3.5 times more per output spike than they gain, and every setting I tried (thresholds 40-150,
a-/a+ 0.15-0.85, a+ 0.001-0.03) ended either silent or at 160 Hz with all weights at 1. Weights
are kept in [0, 1] by ``bound="hard"`` and ``WeightClip``. (4) The paper simulates 450 s; the
default here is 150 s (about 50 s of CPU). (5) The pattern jitter, the exact rate-change
process and the gap distribution are not those of the paper. (6) The threshold (70) and the
slow after-spike hyperpolarizing current (``Afterpotential``, amplitude 600, tau 20 ms, a local
behavior standing in for the paper's negative K2 afterpotential) are tuned by me; the paper's
units differ. Without the afterpotential (v_reset = -2 x threshold only, threshold 115) the
neuron became selective too but its latency grew from 6-9 ms to 19-20 ms over 450 s instead
of shrinking, and the threshold was a knife edge (105 or less ran away to 100+ Hz with all
weights at 1). With the afterpotential, thresholds 60-90 all learn (checked on seed 0, 150 s).

Run: ``python examples/spike_pattern_detection.py`` (``--seed``, ``--seconds``).

Measured (defaults: 150 s, threshold 70, seeds 0, 1, 2; the last third = 50 s, about 245
presentations; a hit is at least one output spike within 50 ms of the onset, a false alarm
a spike outside every presentation):

    seed   hit rate   false alarms   weights > 0.9 / < 0.1   latency first fifth -> last fifth
       0      93.4 %              1             348 / 1509                     7.5 -> 3.8 ms
       1      93.7 %              1             347 / 1505                    11.4 -> 3.5 ms
       2      85.9 %              1             350 / 1520                    11.5 -> 3.2 ms

By the example's criterion (hit rate above 90 %, under 0.5 false alarms/s) 2 of 3 seeds
are selective; seed 2 is at 86 %. The paper: 96 % of 100 runs, 383 weights at 1, hit rate
99.1 %, no false alarms, latency about 4 ms. The weights are bimodal (about 17 % near 1,
75 % near 0, 7 % in between) and every potentiated weight belongs to a pattern afferent;
97-100 % of the pattern afferents that spike in the 15 ms before the output spike end above
0.9, against about 20 % of the other pattern afferents. The latency to the pattern onset
falls from tens of ms (the first responses, 9-46 ms, are close to chance) to 3-4 ms, as in
the paper. In the first seconds the neuron fires nearly everywhere (about 60 false alarms
in the first 15 s). I did not run the 450 s of the paper with these final settings.

Diagnosis of the earlier latency growth (threshold 115 without the afterpotential): the
neuron fired at most once per presentation (1.0 spikes per presentation for most of the
run), so extra late spikes did not drag the potentiated window; the potentiated afferents
were early in the pattern (442 pattern spikes of the selected afferents in the first 10 ms
against 170-215 in each later 10 ms bin). The depression is nearest-pair (an afferent is
depressed once, by its first spike after an output spike, weighted by the decayed
postsynaptic trace), as in the paper. The neuron simply needed about 10 ms of integration of
the early burst to reach that high threshold; a lower threshold made possible by the slow
afterpotential fires earlier.
"""

from __future__ import annotations

import argparse

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.inputs import SpikeInput
from neurosush.neurons.models import LIF, Fire, Refractory
from neurosush.recording import Recorder
from neurosush.synapses.constraints import WeightClip
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces

PATTERN_MS = 50
MAX_HZ = 90.0
NOISE_HZ = 10.0
MAX_SPEED = 1.8  # Hz per ms, i.e. 1800 Hz/s
SPEED_STEP = 0.05  # Hz per ms per ms: std of the random change of the rate speed


class Afterpotential(Behavior):
    """Slow hyperpolarizing current after each output spike (the paper's negative K2 term).

    ``ahp`` jumps by ``amplitude`` at a spike, decays with ``tau`` and is subtracted from
    the membrane current.
    """

    order = Order.DENDRITE_INTEGRATION + 5
    graph_safe = True

    def __init__(self, amplitude: float, tau: float) -> None:
        self.amplitude, self.tau = amplitude, tau

    def initialize(self, group: NeuronGroup) -> None:
        """Allocate the current."""
        group.ahp = group.state()

    def reset_state(self, group: NeuronGroup) -> None:
        """Clear the current."""
        group.ahp.zero_()

    def forward(self, group: NeuronGroup) -> None:
        """Decay, add this step's spikes (of the previous step) and subtract."""
        group.ahp = group.ahp * (1 - group.net.dt / self.tau) + self.amplitude * group.spikes
        group.I = group.I - group.ahp


class PatternInput:
    """Endless afferent spike frames with a repeating pattern; ``onsets`` records its starts.

    Args:
        n: Number of afferents; the first ``n // 2`` take part in the pattern.
        seed: Seed of the private generator.
        mean_gap: Mean extra gap (ms) after the minimum gap of one pattern length.
    """

    def __init__(self, n: int = 2000, *, seed: int = 0, mean_gap: float = 100.0) -> None:
        self.n, self.n_pat, self.mean_gap = n, n // 2, mean_gap
        self.gen = torch.Generator().manual_seed(seed)
        self.rate = torch.rand(n, generator=self.gen) * MAX_HZ
        self.speed = (torch.rand(n, generator=self.gen) * 2 - 1) * MAX_SPEED
        self.onsets: list[int] = []
        self.t = 0
        self.pattern = torch.stack([self._poisson_step()[: self.n_pat] for _ in range(PATTERN_MS)])
        self._next_onset = self._draw_gap()

    def _draw_gap(self) -> int:
        u = torch.rand(1, generator=self.gen).item()
        return int(PATTERN_MS + (-self.mean_gap * torch.log(torch.tensor(1 - u))).item())

    def _poisson_step(self) -> torch.Tensor:
        """Advance the rates one ms; return the Poisson spikes at rate + noise."""
        self.speed = (self.speed + torch.randn(self.n, generator=self.gen) * SPEED_STEP).clamp(
            -MAX_SPEED, MAX_SPEED
        )
        self.rate = self.rate + self.speed
        over, under = self.rate > MAX_HZ, self.rate < 0
        self.rate = self.rate.clamp(0, MAX_HZ)
        self.speed = torch.where(over | under, -self.speed, self.speed)
        p = (self.rate + NOISE_HZ) * 1e-3
        return torch.rand(self.n, generator=self.gen) < p

    def __iter__(self):
        remaining = 0  # steps left in the current pattern presentation
        while True:
            spikes = self._poisson_step()
            if remaining == 0 and self.t >= self._next_onset:
                remaining = PATTERN_MS
                self.onsets.append(self.t)
                self._next_onset = self.t + PATTERN_MS + self._draw_gap()
            if remaining:
                k = PATTERN_MS - remaining
                spikes[: self.n_pat] = self.pattern[k] | (
                    torch.rand(self.n_pat, generator=self.gen) < NOISE_HZ * 1e-3
                )
                remaining -= 1
            self.t += 1
            yield spikes


def build(
    *,
    seed: int = 0,
    n: int = 2000,
    threshold: float = 70.0,
    w_init: float = 0.475,
    a_plus: float = 0.03125,
    a_minus_ratio: float = 0.85,
    reset_factor: float = 0.0,
    ahp_amplitude: float = 600.0,
    ahp_tau: float = 20.0,
) -> tuple[Network, SynapseGroup, PatternInput, Recorder]:
    """The network; returns it, the plastic synapses, the input source and the recorder."""
    source = PatternInput(n, seed=seed)
    net = Network(seed=seed)
    aff = NeuronGroup(net, n, [SpikeInput(iter(source)), Axon()], name="afferents")
    rec = Recorder("spikes")
    out = NeuronGroup(
        net,
        1,
        [
            DendriteStructure(),
            DendriteIntegration(tau_current=2.5),
            LIF(tau=10.0, threshold=threshold, v_reset=-reset_factor * threshold, v_rest=0.0),
            *([Afterpotential(ahp_amplitude, ahp_tau)] if ahp_amplitude else []),
            Refractory(1.0),
            Fire(),
            Axon(),
            rec,
        ],
        name="output",
    )
    syn = SynapseGroup(
        net,
        aff,
        out,
        [
            WeightInit(weights=torch.full((n, 1), w_init)),
            DenseInput(),
            SpikeGather(),
            Traces(tau_pre=16.8, tau_post=33.7, interaction="nearest"),
            STDP(a_plus=a_plus, a_minus=a_minus_ratio * a_plus, bound="hard", pairing="nearest"),
            WeightClip(),
        ],
        name="stdp",
    )
    net.initialize()
    return net, syn, source, rec


def run(seconds: float = 150.0, **kwargs) -> dict:
    """Simulate and return the output spike steps, pattern onsets and final weights."""
    net, syn, source, rec = build(**kwargs)
    net.run(int(seconds * 1000))
    return {
        "spikes": rec.get("spikes").squeeze(1).nonzero().squeeze(1),
        "onsets": torch.tensor(source.onsets[:-1]),  # the last presentation may be cut off
        "weights": syn.weights.detach().squeeze(1).clone(),
        "source": source,
        "steps": int(seconds * 1000),
    }


def analyse(result: dict, tail: float = 1 / 3) -> dict:
    """Hit rate, false alarms, latencies and weight statistics of a run.

    A hit is a presentation during which the neuron spikes at least once (within
    ``PATTERN_MS`` of the onset); a false alarm is a spike outside every presentation. Hit
    rate and false alarms use the last ``tail`` fraction of the run. The latency of a
    presentation is the time of the first spike after its onset (in ms, up to ~2 ms of
    propagation delay included).
    """
    spikes, onsets, w, source = (result[k] for k in ("spikes", "onsets", "weights", "source"))
    steps = result["steps"]
    in_window = torch.zeros(steps, dtype=torch.bool)
    first = []  # (onset, latency or None)
    for o in onsets.tolist():
        in_window[o : o + PATTERN_MS] = True
        mine = spikes[(spikes >= o) & (spikes < o + PATTERN_MS)]
        first.append((o, int(mine[0]) - o if len(mine) else None))
    start = int(steps * (1 - tail))
    late = [lat for o, lat in first if o >= start]
    late_spikes = spikes[spikes >= start]
    false_alarms = int((~in_window[late_spikes]).sum())
    latencies = [lat for _, lat in first if lat is not None]
    # which afferents precede the output spike: pattern afferents firing in the 15 ms before it
    pattern = source.pattern  # (PATTERN_MS, n_pat) bool
    final_lat = sorted(late_l for late_l in late if late_l is not None)
    median_lat = final_lat[len(final_lat) // 2] if final_lat else 0
    lo = max(0, median_lat - 15)
    early = pattern[lo : max(median_lat, 1)].any(0)
    selected = w > 0.9
    n_pat = source.n_pat
    return {
        "hit_rate": sum(lat is not None for lat in late) / max(len(late), 1),
        "false_alarms": false_alarms,
        "false_alarm_rate": false_alarms / (tail * steps / 1000),
        "presentations_tail": len(late),
        "latencies": latencies,
        "median_latency": median_lat,
        "high": int(selected.sum()),
        "low": int((w < 0.1).sum()),
        "middle": int(((w >= 0.1) & (w <= 0.9)).sum()),
        "selected_in_pattern": int(selected[:n_pat].sum()),
        "selected_outside": int(selected[n_pat:].sum()),
        "frac_selected_early": float(selected[:n_pat][early].float().mean()) if early.any() else 0,
        "frac_selected_other": float(selected[:n_pat][~early].float().mean()),
        "n_early": int(early.sum()),
    }


def report(a: dict, seed: int, seconds: float) -> bool:
    """Print the analysis; return whether the run counts as a success."""
    lat = a["latencies"]
    k = 5
    print(f"seed {seed}: {seconds:g} s, {a['presentations_tail']} presentations in the last third")
    print(
        f"  hit rate {a['hit_rate']:.1%}, false alarms {a['false_alarms']} "
        f"({a['false_alarm_rate']:.2f} /s)"
    )
    if lat:
        print(f"  latency (ms) of the first {k} responses: {lat[:k]}")
        print(f"  latency (ms) of the last {k} responses:  {lat[-k:]}")
        n = max(len(lat) // 5, 1)
        print(
            f"  mean latency, first fifth {sum(lat[:n]) / n:.1f} ms, "
            f"last fifth {sum(lat[-n:]) / n:.1f} ms (median at the end {a['median_latency']} ms)"
        )
    print(
        f"  weights: {a['high']} above 0.9 ({a['selected_in_pattern']} pattern, "
        f"{a['selected_outside']} others), {a['low']} below 0.1, {a['middle']} in between"
    )
    print(
        f"  selected among pattern afferents spiking in the 15 ms before the output spike: "
        f"{a['frac_selected_early']:.0%} (n={a['n_early']}), among the others: "
        f"{a['frac_selected_other']:.0%}"
    )
    ok = a["hit_rate"] > 0.9 and a["false_alarm_rate"] < 0.5
    print(f"  selective: {'yes' if ok else 'no'} (hit rate > 90 % and < 0.5 false alarms/s)")
    return ok


def main() -> None:
    """Run the seeds and print the analysis."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seconds", type=float, default=150.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--threshold", type=float, default=70.0)
    args = p.parse_args()
    ok = []
    for seed in args.seeds:
        result = run(args.seconds, seed=seed, threshold=args.threshold)
        ok.append(report(analyse(result), seed, args.seconds))
    print(f"{sum(ok)} of {len(ok)} seeds selective")


if __name__ == "__main__":
    main()
