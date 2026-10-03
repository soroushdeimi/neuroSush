"""Competitive Hebbian learning through STDP (Song, Miller and Abbott 2000).

Song, Miller and Abbott (2000, Nat. Neurosci. 3:919-926) showed that additive
spike-timing-dependent plasticity, with slightly more depression than potentiation, makes
the synapses onto one neuron compete: the weight distribution becomes bimodal (most weights
at 0 or at the maximum), the output rate becomes nearly independent of the input rate, and
inputs that are correlated end up strong. This example reproduces these three results with
:class:`~neurosush.synapses.plasticity.STDP` (``bound="none"``) and
:class:`~neurosush.synapses.constraints.WeightClip`, which together are additive STDP with
hard bounds.

Setup (conductances in units of the leak conductance, times in ms, dt = 1 ms): one
conductance-based integrate-and-fire neuron with tau_m 20, V_rest -70 mV, threshold -54 mV,
reset -60 mV, E_ex 0 mV, E_in -70 mV, tau_ex = tau_in = 5, driven by 1000 plastic excitatory
and 200 fixed inhibitory Poisson afferents (inhibitory weight 0.05, 10 Hz). Excitatory
weights start at g_max = 0.015; STDP has a_plus = 5 * 0.005 g_max (see the deviations),
a_minus = 1.05 a_plus, tau_plus = tau_minus = 20 ms, weights clipped to [0, g_max].

Values I could not verify: the paper's full text was not reachable when this was written,
so the parameters above are quoted from memory (g_max = 0.015, inhibitory conductance 0.05,
A+ = 0.005, A-/A+ = 1.05, tau = 20 ms are what I recall; the initial weight, which I take
as g_max by default (0.5 g_max gives only 1.4 Hz and almost no learning in 100 s; uniform
weights work in the long run), and the correlation
scheme below are my choices). Check them before relying on the numbers.

Deviations: dt = 1 ms (the paper used a smaller step); the quick defaults use an A+ 5
times the quoted value (SPEEDUP) and 100 s runs (``--speedup 1 --seconds 1000`` is the
paper-scale run above); the "10-40 Hz"
input rates are run here at 10 and 40 Hz (20 Hz with ``--rates``); the membrane is integrated
exactly per step; there is no refractory period. Part 3 (my own
correlation scheme): the first group of afferents shares a common Poisson event train; each
member copies a shared event with probability c = 0.2 and otherwise fires independently, so
every afferent keeps the same marginal rate. The paper's scheme (correlation time constant
of 20 ms) differs.

Run: ``python examples/competitive_stdp.py`` (see the measured time below).

Paper-scale run (the documented result): ``python examples/competitive_stdp.py --seconds
1000 --speedup 1 --init uniform --rates 10 40 --correlated 0`` with the paper's unscaled
A+ (about 7 min per input rate on CPU; seed 0, last 100 s). Weights drawn uniformly in
[0, g_max] or all at g_max (``--init 1``) give the same picture:

- Output rate nearly independent of the input rate: 9.3 Hz (10 Hz input) and 10.3 Hz
  (40 Hz input) from uniform initial weights; 11.8 and 10.3 Hz from all weights at g_max.
  The initial rates are very different (1.4 Hz and about 200 Hz at 10 and 40 Hz input from
  uniform weights; 154 and 348 Hz from g_max). The 10 Hz run is still creeping up (uniform
  start) or down (g_max start) slowly after 1000 s.
- Bimodal weights (fractions below 0.1 g_max / above 0.9 g_max): 10 Hz 26 % / 34 % (uniform
  start) or 20 % / 27 % (g_max start); 40 Hz 78 % / 8 % (both starts), the weights being
  pushed down so that a few strong synapses (about 8 %) control the neuron. The 10 Hz
  histograms have a clear peak at each end; the 40 Hz one is dominated by the zero end.

Quick defaults (100 s per run, A+ 5 times larger, all weights at g_max, seed 0, about
2.5 min on CPU) are far from equilibrium and show the trend only:

- Weights (10 Hz input): 11 % below 0.1 g_max and 18 % above 0.9 g_max (mean 0.55 g_max),
  the histogram rising towards both ends.
- Output rate in the last 30 s: 11.6 Hz (10 Hz input), 19.1 Hz (20 Hz), 38.5 Hz (40 Hz).
  The rate is NOT yet independent of the input rate; the 300 s runs at this speedup gave
  7, 21 and 37 Hz, i.e. 5 times the learning rate does not reach the equilibrium by itself
  (it is noisier), so use the paper-scale run above for this result.
- Correlation (500 of 1000 afferents with c = 0.2, 10 Hz): the correlated group ends at a
  mean weight of 0.95 g_max, the uncorrelated afferents at 0.36 g_max; output 39.8 Hz.
"""

from __future__ import annotations

import argparse

import torch

from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import ConductanceIntegration
from neurosush.neurons.inputs import PoissonInput
from neurosush.neurons.models import LIF, Fire
from neurosush.recording import SpikeCounter
from neurosush.synapses.constraints import WeightClip
from neurosush.synapses.currents import DenseInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces

G_MAX = 0.015
G_INH = 0.05
SPEEDUP = 5.0
TAU_STDP = 20.0
CORRELATION = 0.2
INIT = 1.0


class CorrelatedPoissonInput(PoissonInput):
    """Poisson spikes in which the first ``group_size`` neurons are correlated.

    A shared event train with the common rate is drawn each step; a member of the group
    copies a shared event with probability ``correlation`` and otherwise fires on its own
    with rate ``(1 - correlation) * rate``, so its marginal rate stays ``rate``. Later
    neurons are plain independent Poisson neurons. (Local to this example.)
    """

    def __init__(self, rate: float, group_size: int, correlation: float) -> None:
        super().__init__(rate)
        self.group_size, self.correlation = group_size, correlation

    def forward(self, group: NeuronGroup) -> None:
        """Draw this step's spikes."""
        p = group.rates * group.net.dt
        k, c = self.group_size, self.correlation
        independent = group.rand() < p
        if k:
            shared = group.rand()[0] < p[0]
            copy = group.rand()[:k] < c
            own = group.rand()[:k] < p[:k] * (1 - c)
            independent = independent.clone()
            independent[:k] = (shared & copy) | own
        group.spikes = independent


def build(
    rate: float = 10.0,
    n_exc: int = 1000,
    n_inh: int = 200,
    *,
    correlated: int = 0,
    speedup: float = SPEEDUP,
    init: float | None = INIT,
    seed: int = 0,
) -> tuple[Network, NeuronGroup, SynapseGroup]:
    """The network: returns it, the output neuron and the plastic excitatory synapses.

    ``rate`` is the rate of the excitatory afferents in Hz; the first ``correlated`` of them
    form a correlated group.
    """
    net = Network(seed=seed)
    a_plus = speedup * 0.005 * G_MAX
    if init is None:  # uniform in [0, g_max]
        initial = torch.rand(n_exc, 1, generator=torch.Generator().manual_seed(seed)) * G_MAX
    else:
        initial = torch.full((n_exc, 1), init * G_MAX)
    exc_in = NeuronGroup(
        net, n_exc, [CorrelatedPoissonInput(rate / 1000.0, correlated, CORRELATION), Axon()]
    )
    inh_in = NeuronGroup(net, n_inh, [PoissonInput(0.010), Axon()], inhibitory=True)
    cell = NeuronGroup(
        net,
        1,
        [
            ConductanceIntegration(e_exc=0.0, e_inh=-70.0, tau_exc=5.0, tau_inh=5.0),
            LIF(tau=20.0, threshold=-54.0, v_reset=-60.0, v_rest=-70.0),
            Fire(),
            SpikeCounter(),
            Axon(),
        ],
        name="cell",
    )
    plastic = SynapseGroup(
        net,
        exc_in,
        cell,
        [
            WeightInit(weights=initial),
            DenseInput(),
            SpikeGather(),
            Traces(tau_pre=TAU_STDP, tau_post=TAU_STDP),
            STDP(a_plus=a_plus, a_minus=1.05 * a_plus, w_min=0.0, w_max=G_MAX, bound="none"),
            WeightClip(w_min=0.0, w_max=G_MAX),
        ],
        name="exc",
    )
    SynapseGroup(
        net,
        inh_in,
        cell,
        [WeightInit(weights=torch.full((n_inh, 1), G_INH)), DenseInput(), SpikeGather()],
        name="inh",
    )
    net.initialize()
    return net, cell, plastic


def run(
    rate: float = 10.0,
    seconds: int = 100,
    window: int = 10,
    *,
    n_exc: int = 1000,
    n_inh: int = 200,
    correlated: int = 0,
    speedup: float = SPEEDUP,
    init: float | None = INIT,
    seed: int = 0,
) -> dict:
    """Simulate ``seconds`` of learning; returns the rate per window and the final weights."""
    net, cell, plastic = build(
        rate, n_exc, n_inh, correlated=correlated, speedup=speedup, init=init, seed=seed
    )
    rates = []
    for _ in range(max(seconds // window, 1)):
        before = cell.spike_count.clone()
        net.run(window * 1000)
        rates.append(float((cell.spike_count - before).sum()) / window)
    return {"rates": rates, "weights": plastic.weights.detach().flatten().clone() / G_MAX}


def histogram(weights: torch.Tensor, bins: int = 10, width: int = 50) -> str:
    """A text histogram of weights in units of g_max."""
    counts = torch.histc(weights, bins=bins, min=0.0, max=1.0)
    top = float(counts.max()) or 1.0
    lines = []
    for i, count in enumerate(counts.tolist()):
        bar = "#" * round(width * count / top)
        lines.append(f"  {i / bins:.1f}-{(i + 1) / bins:.1f} g_max {int(count):>5}  {bar}")
    return "\n".join(lines)


def bimodality(weights: torch.Tensor, edge: float = 0.1) -> tuple[float, float]:
    """Fractions of weights below ``edge`` g_max and above ``1 - edge`` g_max."""
    return float((weights < edge).float().mean()), float((weights > 1 - edge).float().mean())


def final_rate(rates: list[float], window: int = 10) -> tuple[float, int]:
    """Mean rate over the last 100 s (30 s of a run shorter than 300 s), and that duration."""
    n = 10 if len(rates) >= 30 else min(3, len(rates))
    return sum(rates[-n:]) / n, n * window


def main() -> None:
    """Print the weight distribution, the rate independence and the correlated group."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seconds", type=int, default=100)
    p.add_argument("--rates", type=float, nargs="+", default=[10.0, 40.0])
    p.add_argument("--correlated", type=int, default=500, help="group size in part 3")
    p.add_argument("--speedup", type=float, default=SPEEDUP, help="A+ in units of 0.005 g_max")
    p.add_argument("--init", default=str(INIT), help="initial weight in g_max, or 'uniform'")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    init = None if args.init == "uniform" else float(args.init)
    kw = {"speedup": args.speedup, "init": init, "seed": args.seed}

    print("1+2. uncorrelated inputs: output rate and weights after learning")
    results = {}
    for rate in args.rates:
        r = run(rate, args.seconds, **kw)
        results[rate] = r
        low, high = bimodality(r["weights"])
        last, span = final_rate(r["rates"])
        print(
            f"  input {rate:g} Hz: output {r['rates'][0]:.1f} Hz (first 10 s) -> "
            f"{last:.1f} Hz (last {span} s); weights < 0.1 g_max: "
            f"{low:.0%}, > 0.9 g_max: {high:.0%}, mean {r['weights'].mean():.2f} g_max"
        )
    first = args.rates[0]
    print(f"\nweight histogram at {first:g} Hz ({len(results[first]['weights'])} synapses)")
    print(histogram(results[first]["weights"]))

    if not args.correlated:
        return
    print(f"\n3. {args.correlated} of 1000 afferents correlated (c = {CORRELATION}), {first:g} Hz")
    r = run(first, args.seconds, correlated=args.correlated, **kw)
    w = r["weights"]
    print(f"  mean weight, correlated group: {w[: args.correlated].mean():.2f} g_max")
    print(f"  mean weight, uncorrelated:     {w[args.correlated :].mean():.2f} g_max")
    last, span = final_rate(r["rates"])
    print(f"  output rate {last:.1f} Hz (last {span} s)")


if __name__ == "__main__":
    main()
