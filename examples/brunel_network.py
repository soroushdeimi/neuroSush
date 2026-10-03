"""Sparse excitatory/inhibitory LIF network with four dynamical regimes (Brunel 2000, model A).

Brunel (2000, J. Comput. Neurosci. 8:183-208) studied N_E = 4 N_I leaky integrate-and-fire
neurons with delta-current synapses, random connectivity (epsilon = 0.1), a synaptic delay D
and an external Poisson drive of rate ``nu_ext = eta * nu_thr``, where
``nu_thr = theta / (J C_E tau)`` is the rate at which the external input alone brings the
mean potential to threshold. The ratio ``g`` of inhibitory to excitatory weight and ``eta``
select the regime (Fig. 8 of the paper shows these four points; parameters below are those
of the paper's Fig. 8 as listed in the Brian 2 reproduction, I did not have the paper's own
text, so values were not checked against it):

* A, synchronous regular (SR):        g = 3,   eta = 2
* C, asynchronous irregular (AI):     g = 5,   eta = 2
* B, synchronous irregular, fast (SI fast): g = 6, eta = 4
* D, synchronous irregular, slow (SI slow): g = 4.5, eta = 0.9

Fixed parameters: tau = 20 ms, theta = 20 mV, V_reset = 10 mV, tau_rp = 2 ms, J = 0.1 mV,
D = 1.5 ms, epsilon = 0.1 (the paper uses N_E = 10000, N_I = 2500).

How it maps to neuroSush: potentials are measured from rest (``v_rest = 0``). The LIF step is
``dv = dt / tau * (R I - v)``, so a delta synapse that jumps ``v`` by ``J`` needs a one-step
current ``I = tau / dt * J``; this is the ``coef`` of the :class:`SparseInput` synapses and
of the external drive. External input to each neuron is the sum of ``C_E`` Poisson trains,
i.e. a Poisson count per step with mean ``C_E nu_ext dt`` (drawn by the local
``PoissonDrive`` behavior; the library's ``PoissonInput`` emits at most one spike per step
and cannot carry this count). Inhibition has weight ``g J`` through ``inhibitory=True``.

Deviations from the paper:

* Network size: N_E = 2000, N_I = 500, so C_E = epsilon N_E = 200 (paper: 1000). To keep
  the mean-field feedback ``C_E J`` (and ``nu_thr = theta / (J C_E tau) = 40 Hz``) as in the
  paper, J is scaled to 0.1 mV * 10000 / N_E = 0.5 mV. Mean input is then as in the paper
  but fluctuations are larger (they scale with ``J sqrt(C_E)``, here sqrt(5) times more).
* Time step dt = 0.5 ms instead of 0.1 ms (forward Euler, spikes
  seen at step ends). The delay D = 1.5 ms is 3 steps: the library's one-step transmission
  provides one and ``src_delay = 2`` the rest. The refractory period is 4 steps.
* Every neuron receives exactly epsilon N_E excitatory and epsilon N_I inhibitory
  connections (fixed in-degree, as in the paper; a local ``FixedInDegree`` behavior, since
  ``WeightInit`` samples binomial in-degrees, which with J = 0.5 mV leaves some neurons
  silent and others saturated). Inhibitory neurons get the same external drive.
* Statistics use 2 s after a 0.5 s transient (the paper's figures are much longer).

Run: ``python examples/brunel_network.py`` (about a minute on CPU; ``--regime AI`` runs one).
Statistics: mean rate and mean CV of inter-spike intervals over excitatory neurons with at
least 3 spikes, the synchrony chi (Golomb-Hansel: std of the population activity over the
mean single-neuron std, 2 ms bins; about 1/sqrt(N) = 0.02 for independent neurons, 1 for
full synchrony), and the frequency of the peak of the population-rate spectrum.

Measured with the defaults (seed 0, excitatory neurons; a loaded CPU took 4 minutes, an idle
one about 80 s):

    regime      g  eta  rate Hz     CV    chi  peak Hz
    SR          3    2    328.2   0.07   0.44      166
    AI          5    2     43.4   1.10   0.14      0.5
    SI fast     6    4     60.6   1.45   0.14      170
    SI slow   4.5  0.9     16.4   0.85   0.20       42

Reading: SR is clearly synchronous (chi 0.44, every neuron firing in lockstep, CV 0.07) and
regular. AI is irregular (CV 1.1) with a flat spectrum (largest bin at the lowest
frequencies, no peak); its chi of 0.14 is well above the 0.02 of independent neurons, so
with this small network and J = 0.5 mV it is only weakly asynchronous. SI fast fires
irregularly (CV 1.45) with a population oscillation near 170 Hz, a fast oscillation of the kind the
paper describes (set by the synaptic delay; I did not check its frequency); chi alone does not
separate it from AI here, the spectrum does. SI slow has a low rate (16 Hz), CV 0.85 and a
42 Hz peak, i.e. a slow oscillation, but a weak one (chi 0.2) that I did not check against
the paper's frequency. Not matched: the SR single-neuron rate (328 Hz, two spikes per
population cycle at 166 Hz) is not compared with the paper, and all numbers come from one
seed, 2 s and 2000 neurons, so they are qualitative.
"""

from __future__ import annotations

import argparse

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.core.order import Order
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import DendriteIntegration, DendriteStructure
from neurosush.neurons.models import LIF, Fire, Refractory
from neurosush.recording import Recorder
from neurosush.synapses.currents import SparseInput
from neurosush.synapses.init import DelayInit
from neurosush.synapses.traces import SpikeGather

TAU, THETA, V_RESET, TAU_RP = 20.0, 20.0, 10.0, 2.0  # ms, mV, mV, ms
J_PAPER, EPSILON, DELAY = 0.1, 0.1, 1.5  # mV (at C_E = 1000), connection probability, ms
N_PAPER = 10000

# name -> (g, eta), the four panels of the paper's Fig. 8
REGIMES = {
    "SR": (3.0, 2.0),
    "AI": (5.0, 2.0),
    "SI fast": (6.0, 4.0),
    "SI slow": (4.5, 0.9),
}


class PoissonDrive(Behavior):
    """Add ``coef`` times a Poisson count (mean ``mean``) to the membrane current each step.

    Stands for many independent Poisson inputs onto every neuron, each jumping ``v`` by J.
    """

    order = Order.DENDRITE_INTEGRATION + 10  # after the synaptic currents are summed

    def __init__(self, mean: float, coef: float) -> None:
        self.mean, self.coef = mean, coef

    def forward(self, group: NeuronGroup) -> None:
        """Draw the counts and add the scaled current."""
        counts = torch.poisson(torch.full_like(group.I, self.mean), generator=group.net.generator)
        group.I = group.I + self.coef * counts


class FixedInDegree(Behavior):
    """Connect every destination neuron to exactly ``k`` random distinct sources."""

    order = Order.INITIALIZATION

    def __init__(self, k: int, weight: float) -> None:
        self.k, self.weight = k, weight

    def initialize(self, syn: SynapseGroup) -> None:
        """Draw the sources of each destination and set the edge weights."""
        net = syn.net
        draw = torch.rand(syn.src.size, syn.dst.size, generator=net.generator, device=net.device)
        src = draw.argsort(0)[: self.k]  # (k, n_dst)
        syn.src_idx = src.flatten()
        syn.dst_idx = torch.arange(syn.dst.size, device=net.device).repeat(self.k)
        syn.weights = torch.full((self.k * syn.dst.size,), self.weight, dtype=net.dtype)


def build(
    n_exc: int = 2000, g: float = 5.0, eta: float = 2.0, dt: float = 0.5, seed: int = 0
) -> tuple[Network, NeuronGroup, NeuronGroup]:
    """The network; returns it, the excitatory and the inhibitory population."""
    n_inh = n_exc // 4
    c_exc = EPSILON * n_exc
    jump = J_PAPER * N_PAPER / n_exc  # keeps C_E J (the mean feedback) at its paper value
    nu_thr = THETA / (jump * c_exc * TAU)  # spikes per ms
    coef = TAU / dt * jump  # one-step current that makes v jump by J
    delay_steps = round(DELAY / dt)
    net = Network(dt=dt, seed=seed)

    def population(name: str, size: int, inhibitory: bool) -> NeuronGroup:
        return NeuronGroup(
            net,
            size,
            [
                DendriteStructure(),
                DendriteIntegration(),
                PoissonDrive(c_exc * eta * nu_thr * dt, coef),
                LIF(tau=TAU, threshold=THETA, v_reset=V_RESET, v_rest=0.0),
                Refractory(TAU_RP),
                Fire(),
                Axon(max_delay=delay_steps),
                Recorder("spikes"),
            ],
            inhibitory=inhibitory,
            name=name,
        )

    exc, inh = population("exc", n_exc, False), population("inh", n_inh, True)
    for src in (exc, inh):
        for dst in (exc, inh):
            SynapseGroup(
                net,
                src,
                dst,
                [
                    FixedInDegree(round(EPSILON * src.size), g if src.inhibitory else 1.0),
                    DelayInit(delays=delay_steps - 1),  # plus one step of transmission
                    SparseInput(coef=coef),
                    SpikeGather(),
                ],
                name=f"{src.name}_{dst.name}",
            )
    net.initialize()
    return net, exc, inh


def _recorder(group: NeuronGroup) -> Recorder:
    return next(b for b in group.behaviors if isinstance(b, Recorder))


def statistics(spikes: torch.Tensor, dt: float, bin_ms: float = 2.0) -> dict:
    """Rate, ISI CV, synchrony chi and spectral peak of a ``(steps, neurons)`` spike raster."""
    steps, n = spikes.shape
    seconds = steps * dt / 1000.0
    counts = spikes.sum(0)
    rate = float(counts.float().mean()) / seconds
    cvs = []
    for i in torch.nonzero(counts >= 3).flatten().tolist():
        isi = torch.diff(torch.nonzero(spikes[:, i]).flatten().float())
        cvs.append(float(isi.std(unbiased=False) / isi.mean()))
    cv = sum(cvs) / len(cvs) if cvs else float("nan")
    k = max(1, round(bin_ms / dt))
    binned = spikes[: steps // k * k].reshape(-1, k, n).sum(1).float()
    pop = binned.mean(1)
    single = binned.var(0, unbiased=False).mean()
    chi = float(pop.var(unbiased=False).sqrt() / single.sqrt()) if single > 0 else float("nan")
    spectrum = torch.fft.rfft(pop - pop.mean()).abs()
    freqs = torch.fft.rfftfreq(pop.numel(), d=bin_ms / 1000.0)
    peak = int(spectrum[1:].argmax()) + 1
    return {"rate": rate, "cv": cv, "chi": chi, "peak_hz": float(freqs[peak]), "pop": pop}


def run(
    regime: str = "AI",
    n_exc: int = 2000,
    seconds: float = 2.0,
    transient: float = 0.5,
    dt: float = 0.5,
    seed: int = 0,
) -> dict:
    """Simulate one regime; returns the statistics of the excitatory population."""
    g, eta = REGIMES[regime]
    net, exc, _ = build(n_exc, g, eta, dt, seed)
    net.run(round((transient + seconds) * 1000 / dt))
    raster = _recorder(exc).get("spikes")[round(transient * 1000 / dt) :]
    return statistics(raster, dt) | {"raster": raster, "g": g, "eta": eta}


def text_raster(raster: torch.Tensor, dt: float, neurons: int = 8, ms: float = 400.0) -> str:
    """The first ``ms`` of ``neurons`` rows as text, one character per 4 ms."""
    k, width = round(4.0 / dt), round(ms / 4.0)
    rows = []
    for i in range(neurons):
        col = raster[: k * width, i].reshape(width, k).any(1)
        rows.append("".join("|" if s else "." for s in col.tolist()))
    return "\n".join(rows)


def main() -> None:
    """Print the statistics of the four regimes."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--regime", choices=list(REGIMES), default=None, help="one regime only")
    p.add_argument("--n-exc", type=int, default=2000)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raster", action="store_true", help="print a text raster per regime")
    args = p.parse_args()
    print(f"{'regime':8} {'g':>4} {'eta':>4} {'rate Hz':>8} {'CV':>6} {'chi':>6} {'peak Hz':>8}")
    for name in [args.regime] if args.regime else REGIMES:
        r = run(name, args.n_exc, args.seconds, dt=args.dt, seed=args.seed)
        print(
            f"{name:8} {r['g']:4g} {r['eta']:4g} {r['rate']:8.1f} {r['cv']:6.2f} "
            f"{r['chi']:6.2f} {r['peak_hz']:8.0f}"
        )
        if args.raster:
            print(text_raster(r["raster"], args.dt))


if __name__ == "__main__":
    main()
