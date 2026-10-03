"""Inhibitory plasticity balances excitation (reduced version of Vogels et al. 2011).

Vogels, Sprekeler, Zenke, Clopath and Gerstner (2011, Science 334:1569) showed that a
symmetric STDP rule on inhibitory synapses (:class:`~neurosush.synapses.plasticity.ISTDP`)
drives a postsynaptic neuron to a target firing rate ``rho`` whatever the strength of its
excitatory drive: pre/post coincidences potentiate the inhibition, every presynaptic spike
depresses it by ``alpha = 2 * rho * tau``, so inhibition grows while the neuron fires above
``rho`` and shrinks below it.

Here 20 excitatory conductance-based LIF neurons receive 200 Poisson excitatory inputs at
10 Hz and 50 Poisson inhibitory inputs at 10 Hz through plastic weights that start weak,
so the neurons initially fire far above the 5 Hz target (``rho = 0.005`` per ms, dt = 1 ms).
The firing rate over time and the final inhibitory weights are printed.

Deviations from the paper (a much reduced network): the paper simulates recurrent networks
of 8000 excitatory and 2000 inhibitory neurons with sparse connectivity, where the
inhibitory neurons are driven by the excitatory ones; here both input populations are
independent Poisson sources (the inhibitory population does not react to the excitatory
cells), the connectivity is all-to-all and the neuron model is a conductance LIF with
parameters of my choosing. Only the convergence of a postsynaptic rate to ``rho`` is shown,
not the detailed balance of excitation and inhibition or the response to stimuli.

Run: ``python examples/balanced_network.py`` (about 15 s on CPU).

Measured with the defaults (60 s, 5 s windows, seed 0): the mean excitatory rate falls
from 44 Hz in the first window through 16, 9.8 and 7.6 Hz to between 4.4 and 7.8 Hz
(mean about 5.7 Hz) from 25 s on, around the 5 Hz target; the inhibitory weights grow
from 0.02 to a mean of 0.224 (std 0.027). The rate fluctuates around the target because
of the Poisson input and the noisy plasticity.
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
from neurosush.synapses.plasticity import ISTDP
from neurosush.synapses.traces import SpikeGather, Traces

TARGET_HZ = 5.0


def build(
    neurons: int = 20,
    n_exc: int = 200,
    n_inh: int = 50,
    *,
    exc_weight: float = 0.06,
    inh_weight: float = 0.02,
    lr: float = 2e-3,
    target_hz: float = TARGET_HZ,
    seed: int = 0,
) -> tuple[Network, NeuronGroup, SynapseGroup]:
    """The network; returns it, the excitatory population and the plastic inhibitory synapses."""
    net = Network(seed=seed)
    exc_in = NeuronGroup(net, n_exc, [PoissonInput(0.010), Axon()], name="exc_input")
    inh_in = NeuronGroup(
        net, n_inh, [PoissonInput(0.010), Axon()], inhibitory=True, name="inh_input"
    )
    cells = NeuronGroup(
        net,
        neurons,
        [
            ConductanceIntegration(e_exc=0.0, e_inh=-80.0, tau_exc=5.0, tau_inh=10.0),
            LIF(tau=20.0, threshold=-50.0, v_reset=-60.0, v_rest=-60.0),
            Fire(),
            SpikeCounter(),
            Axon(),
        ],
        name="cells",
    )
    SynapseGroup(
        net,
        exc_in,
        cells,
        [WeightInit(weights=torch.full((n_exc, neurons), exc_weight)), DenseInput(), SpikeGather()],
        name="exc",
    )
    inhibition = SynapseGroup(
        net,
        inh_in,
        cells,
        [
            WeightInit(weights=torch.full((n_inh, neurons), inh_weight)),
            DenseInput(),
            SpikeGather(),
            Traces(tau_pre=20.0),
            ISTDP(lr=lr, rho=target_hz / 1000.0),
            WeightClip(w_min=0.0, w_max=5.0),
        ],
        name="inh",
    )
    net.initialize()
    return net, cells, inhibition


def run(seconds: int = 60, window: int = 5, **kwargs: float) -> dict:
    """Simulate ``seconds`` of activity; returns the rate per window and the final weights."""
    net, cells, inhibition = build(**kwargs)
    rates = []
    for _ in range(seconds // window):
        before = cells.spike_count.clone()
        net.run(window * 1000)
        rates.append(float((cells.spike_count - before).float().mean()) / window)
    return {"rates": rates, "window": window, "weights": inhibition.weights.detach().clone()}


def main() -> None:
    """Print the firing rate over time and the final inhibitory weights."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--window", type=int, default=5, help="seconds per rate estimate")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    result = run(args.seconds, args.window, lr=args.lr, seed=args.seed)
    print(f"mean excitatory rate (target {TARGET_HZ:g} Hz)")
    for i, rate in enumerate(result["rates"]):
        print(f"  {i * args.window:>4}-{(i + 1) * args.window:<4} s  {rate:6.2f} Hz")
    w = result["weights"]
    print(f"final inhibitory weights: mean {w.mean():.3f}, std {w.std():.3f}, max {w.max():.3f}")


if __name__ == "__main__":
    main()
