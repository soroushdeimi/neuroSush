"""Diehl and Cook (2015): unsupervised MNIST digit recognition with STDP.

Reproduces "Unsupervised learning of digit recognition using spike-timing-dependent
plasticity" (Front. Comput. Neurosci. 9:99). 784 Poisson input neurons drive N excitatory
LIF neurons through plastic conductance synapses (STDP with traces, weights clipped to
[0, 1], each neuron's input weights normalized to a sum of 78 before every sample). Every
excitatory neuron excites one inhibitory neuron, which inhibits all the other excitatory
neurons: a soft winner-take-all. An adaptive threshold (theta) keeps any neuron from
dominating. No labels are used while learning. Afterwards the weights and thetas are frozen,
every neuron is assigned the digit it responds to most on training images, and a test image
is classified by the digit whose neurons fire most.

Learning is online, one sample at a time, as in the paper. Each sample runs for 350 ms
(``--presentation``, ``round(350 / dt)`` steps); the 150 ms rest between samples is replaced
by resetting the state in place (which also keeps CUDA graphs valid). The rest is not
simulated except for the relaxation of theta (``--rest``, default 150 ms, applied after
every presentation including repeats, as the paper's rest follows each one); voltages,
conductances and traces are reset instead, which is what 150 ms with tau_m = 100 ms nearly
does. Without it theta's equilibrium is too high (about 71 mV instead of 50 mV at five
spikes per sample) and neurons stop firing.

A sample that makes the excitatory layer fire fewer
than 5 spikes is shown again with a higher input intensity.

Learning rule (``--rule``): ``pair`` (default) potentiates on a postsynaptic spike by the
presynaptic trace and depresses on a presynaptic spike, with traces of 20 ms. ``triplet`` is
the rule of the paper's code: potentiation on a postsynaptic spike scales with a slower
postsynaptic trace (40 ms) and traces are nearest-spike (set to one at a spike), implemented
with ``TripletSTDP``.

``--dt`` (ms, default 1.0) sets the simulation step; every duration stays in ms and is
converted to steps with ``round(ms / dt)`` (presentation, rest). The paper's Brian code uses
dt = 0.5 ms; with it the one-step synaptic transmission makes exc -> inh -> exc inhibition
arrive after 1 ms instead of 2 ms. Time constants and rates are in ms (Hz / 1000), so only the
step counts depend on dt; the conductance taus (1 and 2 ms) must be at least dt.

During training the winner diagnostics are recorded for the last presentation of every sample:
the number of distinct excitatory neurons that fired and the share of the sample's spikes
produced by its most active neuron (JSON ``winner_diagnostics``: means and means per block of
10000 samples, per member with ``--members``).

Run: ``python examples/diehl_cook_mnist.py --data path/to/MNIST/raw`` (the folder with the
IDX files; without it the files are downloaded). The full paper setting (100 neurons, 60000
samples, one epoch) takes about an hour on a GPU. Quick check:
``--train-samples 500 --label-samples 1000 --test-samples 1000``.
The quick check runs on CPU at about 2 samples/s; accuracy is low for the first few thousand
training samples, so it only shows that the code runs. Full-epoch results on an RTX 3090
will be reported. Use ``--checkpoint file.pt``
(and ``--resume``) to survive interruptions and ``--out results.json`` to keep the results.

``--members B`` (B >= 2) trains B independent copies of the network at once, in one batch of an
``independent=True`` network: every member has its own random initial weights, Poisson input,
weights and thetas, and learns online exactly as the single network does (the same sample
sequence is shown to all members, and a member that needed a repeat does not move the others'
weights). Members differ by their random draws (all taken from the ``--seed`` generator). On a
GPU this costs little more than one network. Evaluation is done per member, and the JSON holds
``accuracies`` (one per member), ``accuracy`` (their mean) and ``accuracy_std``. The default
``--members 1`` is the plain single network.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from neurosush.core.behavior import Behavior
from neurosush.core.compiled import CompiledStepper
from neurosush.core.graph import GraphStepper
from neurosush.core.network import Network, NeuronGroup, SynapseGroup
from neurosush.data import load_mnist
from neurosush.neurons.axon import Axon
from neurosush.neurons.dendrite import ConductanceIntegration
from neurosush.neurons.homeostasis import AdaptiveThreshold
from neurosush.neurons.inputs import PoissonInput
from neurosush.neurons.models import LIF, Fire, Refractory
from neurosush.readout import accuracy, assign_labels, classify
from neurosush.recording import SpikeCounter
from neurosush.synapses.constraints import WeightClip
from neurosush.synapses.currents import DenseInput, OneToOneInput
from neurosush.synapses.init import WeightInit
from neurosush.synapses.plasticity import STDP
from neurosush.synapses.traces import SpikeGather, Traces
from neurosush.synapses.triplet import TripletSTDP

MAX_RATE = 63.75  # Hz of a white pixel at intensity 2
WEIGHT_SUM = 78.0
MIN_SPIKES = 5
RULES = ("pair", "triplet")
THETA_TAU = 1e7
REST = 150  # ms of rest after every training presentation in the paper
MAX_INTENSITY = 10.0


@dataclass
class Model:
    """The network, the groups the example touches and the object that steps it."""

    net: Network
    input: NeuronGroup
    exc: NeuronGroup
    inh: NeuronGroup
    syn: SynapseGroup
    stepper: Network | GraphStepper | CompiledStepper
    learn: bool


def build_network(
    neurons: int = 100,
    device: str | torch.device = "cpu",
    *,
    batch: int | None = None,
    learn: bool = True,
    graph: bool | Literal["graph", "compiled"] = False,
    seed: int = 0,
    rule: str = "pair",
    members: int | None = None,
    dt: float = 1.0,
) -> Model:
    """Build the Diehl and Cook network; ``learn=False`` leaves out plasticity and theta.

    ``rule`` is ``"pair"`` (traces and pair STDP, the default) or ``"triplet"`` (the rule of
    the paper's code: nearest-spike triplet STDP, see the module docstring). ``members`` builds
    that many independent copies in one batch (states ``(members, size)``, weights
    ``(members, 784, N)``); it cannot be combined with ``batch``. ``dt`` is the step in ms.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if members is not None and batch is not None:
        raise ValueError("members and batch cannot be combined")
    n = neurons
    net = Network(
        device=device,
        seed=seed,
        dt=dt,
        batch_size=members if members is not None else batch,
        independent=members is not None,
    )
    inp = NeuronGroup(net, 784, [PoissonInput(), Axon()], name="input")
    exc_behaviors: list[Behavior] = [
        ConductanceIntegration(e_exc=0.0, e_inh=-100.0, tau_exc=1.0, tau_inh=2.0),
        LIF(tau=100.0, threshold=-52.0, v_reset=-65.0, v_rest=-65.0),
        Refractory(5.0),
        Fire(),
        SpikeCounter(),
        Axon(),
    ]
    if learn:
        exc_behaviors.insert(4, AdaptiveThreshold(increment=0.05, tau=THETA_TAU))
    exc = NeuronGroup(net, n, exc_behaviors, name="exc")
    inh = NeuronGroup(
        net,
        n,
        [
            ConductanceIntegration(e_exc=0.0, e_inh=-85.0, tau_exc=1.0, tau_inh=2.0),
            LIF(tau=10.0, threshold=-40.0, v_reset=-45.0, v_rest=-60.0),
            Refractory(2.0),
            Fire(),
            Axon(),
        ],
        inhibitory=True,
        name="inh",
    )
    syn_behaviors: list[Behavior] = [
        WeightInit(mode="uniform", scale=0.3),
        DenseInput(),
        SpikeGather(),
    ]
    if learn:
        if rule == "pair":
            syn_behaviors += [
                Traces(tau_pre=20.0, tau_post=20.0),
                STDP(a_plus=0.01, a_minus=0.0001),
            ]
        else:
            syn_behaviors.append(
                TripletSTDP(
                    a2_plus=0.0,
                    a3_plus=0.01,
                    a2_minus=0.0001,
                    a3_minus=0.0,
                    tau_plus=20.0,
                    tau_minus=20.0,
                    tau_x=20.0,  # unused: a3_minus is 0
                    tau_y=40.0,
                    interaction="nearest",
                )
            )
        syn_behaviors.append(WeightClip(w_min=0.0, w_max=1.0))
    syn = SynapseGroup(net, inp, exc, syn_behaviors, name="input->exc")
    SynapseGroup(
        net,
        exc,
        inh,
        [WeightInit(weights=torch.full((n,), 10.4), shape=(n,)), OneToOneInput(), SpikeGather()],
        name="exc->inh",
    )
    SynapseGroup(
        net,
        inh,
        exc,
        [WeightInit(weights=17.0 * (1 - torch.eye(n))), DenseInput(), SpikeGather()],
        name="inh->exc",
    )
    net.initialize()
    stepper: Network | GraphStepper | CompiledStepper = net
    if graph == "compiled":
        stepper = CompiledStepper(net)
    elif graph:
        stepper = GraphStepper(net)
    return Model(net, inp, exc, inh, syn, stepper, learn)


def normalize_weights(model: Model) -> None:
    """Scale every excitatory neuron's input weights to sum to 78, in place."""
    w = model.syn.weights
    w.mul_(WEIGHT_SUM / w.sum(-2, keepdim=True).clamp(min=1e-12))


def relax_theta(model: Model, steps: int) -> None:
    """Relax theta as ``steps`` silent steps would: ``theta *= (1 - dt / tau) ** steps``."""
    exc = model.exc
    exc.theta.mul_((1 - model.net.dt / THETA_TAU) ** steps)
    exc.threshold.copy_(exc.base_threshold + exc.theta)


def present(
    model: Model,
    images: torch.Tensor,
    steps: int = 350,
    intensity: float = 2.0,
    *,
    min_spikes: int = MIN_SPIKES,
    max_intensity: float = MAX_INTENSITY,
    rest: int = 0,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Show images until each makes the excitatory layer fire ``min_spikes`` spikes.

    ``images`` is ``(784,)`` uint8 (unbatched network; with independent members every member
    sees it) or ``(batch, 784)``. A sample with too
    few spikes is shown again with the intensity raised by one, up to ``max_intensity``;
    each sample keeps the counts of its last presentation. With independent members, the
    weights and thetas of a member whose sample was already good are restored after a repeat
    that other members needed, so each member learns as if it were alone. Returns the spike
    counts (shape of ``exc.spike_count``) and
    ``{"repeats": samples shown again, "sim_steps": steps run}``, with a batch counted as one
    run per step. With ``rest`` > 0 (training) every presentation, re-presentations included,
    is followed by ``rest`` steps of rest, of which only the relaxation of theta is applied
    (see :func:`relax_theta`).
    """
    net = model.net
    rates = images.to(net.device, net.dtype) / 255.0 * MAX_RATE / 1000.0  # spikes per ms
    result: torch.Tensor | None = None
    pending = torch.zeros((), dtype=torch.bool, device=net.device)
    repeats = sim_steps = 0
    while True:
        saved = None
        if result is not None and model.learn and net.independent:
            saved = (model.syn.weights.clone(), model.exc.theta.clone())
        net.reset_state()
        model.input.rates.copy_(rates * intensity / 2.0)
        model.stepper.run(steps)
        sim_steps += steps
        if rest and model.learn:
            relax_theta(model, rest)
        if saved is not None:  # members that did not need this repeat keep what they had
            done = ~pending
            model.syn.weights.copy_(torch.where(done[:, None, None], saved[0], model.syn.weights))
            model.exc.theta.copy_(torch.where(done[:, None], saved[1], model.exc.theta))
            model.exc.threshold.copy_(model.exc.base_threshold + model.exc.theta)
        counts = model.exc.spike_count.clone()
        if result is None:
            result = counts
            pending = counts.sum(-1) < min_spikes
        else:
            result = torch.where(pending.unsqueeze(-1), counts, result)
            repeats += int(pending.sum().item())
            pending = pending & (counts.sum(-1) < min_spikes)
        if intensity >= max_intensity or not bool(pending.any().item()):
            return result, {"repeats": repeats, "sim_steps": sim_steps}
        intensity += 1.0


def _save_checkpoint(path: Path, model: Model, state: dict[str, Any]) -> None:
    payload = {
        "weights": model.syn.weights.cpu(),
        "theta": model.exc.theta.cpu(),
        "rng": model.net.generator.get_state(),
        **state,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(
    model: Model,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    samples: int,
    epochs: int = 1,
    steps: int = 350,
    log_every: int = 1000,
    checkpoint: str | Path | None = None,
    checkpoint_every: int = 5000,
    resume: bool = False,
    window: int = 10000,
    rest: int = REST,
    block: int = 10000,
) -> dict[str, Any]:
    """Train online, one sample at a time, for ``epochs * samples`` presentations.

    ``steps`` and ``rest`` are in simulation steps (``main`` converts from ms). Winner
    diagnostics (distinct firing neurons, top neuron's share of spikes) of every sample are
    kept on the device and reduced only at log time and at the end (``winner_diagnostics``,
    with means per ``block`` samples).

    Samples are taken in order from ``images`` (wrapping around). Returns timing and counts;
    the running accuracy follows the paper: labels are assigned from the previous ``window``
    samples' responses and tested on the latest ``log_every`` ones.
    """
    total = samples * epochs
    start = 0
    stats = {"repeats": 0, "sim_steps": 0, "seconds": 0.0}
    recent: list[tuple[torch.Tensor, int]] = []
    ckpt = Path(checkpoint) if checkpoint else None
    if resume and ckpt is not None and ckpt.exists():
        saved = torch.load(ckpt, map_location="cpu")
        model.syn.weights.copy_(saved["weights"])
        model.exc.theta.copy_(saved["theta"])
        model.exc.threshold.copy_(model.exc.base_threshold + model.exc.theta)
        model.net.generator.set_state(saved["rng"])
        start, stats = int(saved["index"]), dict(saved["stats"])
        recent = list(zip(saved["recent_counts"], saved["recent_labels"], strict=True))
        print(f"resumed from {ckpt} at sample {start}", flush=True)
    t0 = time.perf_counter() - stats["seconds"]
    t_log, steps_log, done_log = time.perf_counter(), stats["sim_steps"], start
    accuracies: list[float] = []  # one list per log point with independent members
    diag = torch.zeros(
        (total - start, 2, *model.exc.spike_count.shape[:-1]), device=model.net.device
    )
    diag_log = 0
    for i in range(start, total):
        normalize_weights(model)
        counts, info = present(model, images[i % len(images)], steps, rest=rest)
        stats["repeats"] += info["repeats"]
        stats["sim_steps"] += info["sim_steps"]
        label = int(labels[i % len(labels)])
        fired = counts.sum(-1)
        diag[i - start, 0] = (counts > 0).sum(-1)
        diag[i - start, 1] = counts.amax(-1) / fired.clamp(min=1)
        recent.append((counts.cpu(), label))
        del recent[: max(0, len(recent) - window - log_every)]
        done = i + 1
        if done % log_every == 0 or done == total:
            now = time.perf_counter()
            line = (
                f"{done}/{total} samples, {now - t0:.0f}s, "
                f"{(done - done_log) / (now - t_log):.1f} samples/s, "
                f"{(stats['sim_steps'] - steps_log) / (now - t_log):.0f} steps/s, "
                f"mean theta {model.exc.theta.mean().item():.3f}"
            )
            window_diag = diag[diag_log : done - start].mean(0).reshape(2, -1).mean(1)
            line += (
                f", winners {window_diag[0].item():.1f} distinct neurons, "
                f"top share {window_diag[1].item():.2f}"
            )
            diag_log = done - start
            if len(recent) > log_every:
                labelled, latest = recent[:-log_every], recent[-log_every:]
                old = torch.stack([c for c, _ in labelled])
                new = torch.stack([c for c, _ in latest])
                truth = torch.tensor([y for _, y in latest])
                targets = torch.tensor([y for _, y in labelled])
                if model.net.independent:  # counts are (samples, members, N)
                    per_member = [
                        accuracy(
                            classify(new[:, b], assign_labels(old[:, b], targets)),
                            truth,
                        )
                        for b in range(old.shape[1])
                    ]
                    accuracies.append(per_member)  # type: ignore[arg-type]
                    line += f", running accuracy {sum(per_member) / len(per_member):.3f} (mean)"
                else:
                    accuracies.append(accuracy(classify(new, assign_labels(old, targets)), truth))
                    line += f", running accuracy {accuracies[-1]:.3f}"
            print(line, flush=True)
            t_log, steps_log, done_log = now, stats["sim_steps"], done
        if ckpt is not None and (done % checkpoint_every == 0 or done == total):
            stats["seconds"] = time.perf_counter() - t0
            _save_checkpoint(
                ckpt,
                model,
                {
                    "index": done,
                    "stats": stats,
                    "recent_counts": [c for c, _ in recent],
                    "recent_labels": [y for _, y in recent],
                },
            )
    stats["seconds"] = time.perf_counter() - t0
    stats["samples"] = total
    stats["running_accuracy"] = accuracies  # type: ignore[assignment]
    stats["winner_diagnostics"] = _winner_summary(diag, start, block)
    normalize_weights(model)
    return stats


def _winner_summary(diag: torch.Tensor, start: int, block: int) -> dict[str, Any]:
    """Means of the per-sample diagnostics (``(samples, 2, *members)``), overall and per block."""
    if len(diag) == 0:
        return {"samples": 0}

    def means(x: torch.Tensor) -> dict[str, Any]:
        m = x.mean(0).cpu()
        return {"distinct_neurons_mean": m[0].tolist(), "top_share_mean": m[1].tolist()}

    blocks = []
    first = start
    while first < start + len(diag):  # blocks are aligned to multiples of ``block``
        last = min((first // block + 1) * block, start + len(diag))
        blocks.append({"start": first, "end": last, **means(diag[first - start : last - start])})
        first = last
    return {"samples": len(diag), **means(diag), "blocks": blocks}


def respond(
    weights: torch.Tensor,
    theta: torch.Tensor,
    images: torch.Tensor,
    *,
    device: str | torch.device,
    batch: int,
    steps: int,
    graph: bool | Literal["graph", "compiled"],
    seed: int,
    dt: float = 1.0,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Spike counts of the frozen network on ``images``, batched; returns ``(samples, N)``."""
    neurons = weights.shape[1]
    models: dict[int, Model] = {}
    out: list[torch.Tensor] = []
    total = {"repeats": 0, "sim_steps": 0, "sample_steps": 0}
    for start in range(0, len(images), batch):
        chunk = images[start : start + batch]
        size = len(chunk)
        if size not in models:
            m = build_network(
                neurons, device, batch=size, learn=False, graph=graph, seed=seed, dt=dt
            )
            m.syn.weights.copy_(weights)
            m.exc.threshold.copy_(m.exc.threshold + theta.to(m.net.device))
            models[size] = m
        counts, info = present(models[size], chunk, steps)
        out.append(counts.cpu())
        total["repeats"] += info["repeats"]
        total["sim_steps"] += info["sim_steps"]
        total["sample_steps"] += info["sim_steps"] * size
    return torch.cat(out), total


def evaluate(
    model: Model,
    label_images: torch.Tensor,
    label_targets: torch.Tensor,
    test_images: torch.Tensor,
    test_targets: torch.Tensor,
    *,
    batch: int = 1000,
    steps: int = 350,
    graph: bool | Literal["graph", "compiled"] = False,
    seed: int = 0,
    dt: float = 1.0,
) -> dict[str, Any]:
    """Assign neurons to digits on ``label_images``, then classify ``test_images``.

    With independent members every member is evaluated on its own (its weights and thetas in
    a frozen shared-batch network); ``accuracy`` is then the mean over members, with
    ``accuracies``, ``accuracy_std`` and one entry per member in ``member_results``.
    """
    if model.net.independent:
        parts = [
            _evaluate_one(
                model.syn.weights[b].detach().clone(),
                model.exc.theta[b].detach().clone(),
                model.net.device,
                label_images,
                label_targets,
                test_images,
                test_targets,
                batch=batch,
                steps=steps,
                graph=graph,
                seed=seed,
                dt=dt,
            )
            for b in range(model.net.batch_size or 1)
        ]
        accuracies = torch.tensor([part["accuracy"] for part in parts])
        seconds = sum(part["evaluation_seconds"] for part in parts)
        return {
            "accuracy": float(accuracies.mean()),
            "accuracy_std": float(accuracies.std(unbiased=False)),
            "accuracies": accuracies.tolist(),
            "assignment_histogram": [part["assignment_histogram"] for part in parts],
            "evaluation_seconds": seconds,
            "evaluation_sample_steps_per_second": sum(
                part["evaluation_sample_steps"] for part in parts
            )
            / seconds,
            "evaluation_repeats": sum(part["evaluation_repeats"] for part in parts),
        }
    result = _evaluate_one(
        model.syn.weights.detach().clone(),
        model.exc.theta.detach().clone(),
        model.net.device,
        label_images,
        label_targets,
        test_images,
        test_targets,
        batch=batch,
        steps=steps,
        graph=graph,
        seed=seed,
        dt=dt,
    )
    del result["evaluation_sample_steps"]
    return result


def _evaluate_one(
    weights: torch.Tensor,
    theta: torch.Tensor,
    device: torch.device,
    label_images: torch.Tensor,
    label_targets: torch.Tensor,
    test_images: torch.Tensor,
    test_targets: torch.Tensor,
    *,
    batch: int,
    steps: int,
    graph: bool | Literal["graph", "compiled"],
    seed: int,
    dt: float = 1.0,
) -> dict[str, Any]:
    """Evaluate one set of weights and thetas; also reports the sample steps it simulated."""
    kwargs: dict[str, Any] = {
        "device": device,
        "batch": batch,
        "steps": steps,
        "graph": graph,
        "seed": seed,
        "dt": dt,
    }
    t0 = time.perf_counter()
    label_counts, info1 = respond(weights, theta, label_images, **kwargs)
    assignment = assign_labels(label_counts, label_targets.cpu())
    test_counts, info2 = respond(weights, theta, test_images, **kwargs)
    predicted = classify(test_counts, assignment)
    seconds = time.perf_counter() - t0
    sample_steps = info1["sample_steps"] + info2["sample_steps"]
    return {
        "accuracy": accuracy(predicted, test_targets.cpu()),
        "assignment_histogram": torch.bincount(assignment[assignment >= 0], minlength=10).tolist(),
        "evaluation_seconds": seconds,
        "evaluation_sample_steps_per_second": sample_steps / seconds,
        "evaluation_repeats": info1["repeats"] + info2["repeats"],
        "evaluation_sample_steps": sample_steps,
    }


def machine_info(device: torch.device) -> dict[str, str]:
    """Software and hardware the run used."""
    return {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def main() -> None:
    """Train, evaluate, print and optionally save the results."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--neurons", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train-samples", type=int, default=60000, help="per epoch")
    p.add_argument("--label-samples", type=int, default=10000)
    p.add_argument("--test-samples", type=int, default=10000)
    p.add_argument("--eval-batch", type=int, default=1000)
    p.add_argument("--dt", type=float, default=1.0, help="simulation step in ms")
    p.add_argument(
        "--presentation",
        "--steps",
        dest="presentation",
        type=float,
        default=350.0,
        help="ms each sample is shown (--steps is an old alias, now also in ms)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-graph", action="store_true", help="step eagerly (no CUDA graphs)")
    p.add_argument(
        "--stepper",
        choices=("graph", "compiled"),
        default="graph",
        help="CUDA only: replay the step as a CUDA graph, or torch.compile it first "
        "(CompiledStepper: matches eager to a tolerance, not bit for bit)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--members",
        type=int,
        default=1,
        help="train this many independent networks at once (different random draws)",
    )
    p.add_argument("--rest", type=float, default=REST, help="ms of rest after each training sample")
    p.add_argument("--rule", choices=RULES, default="pair", help="synaptic learning rule")
    p.add_argument("--data", default="data/MNIST/raw", help="folder of MNIST IDX files")
    p.add_argument("--out", help="write the results as JSON")
    p.add_argument("--checkpoint", help="file to save progress to")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--resume", action="store_true", help="continue from --checkpoint")
    args = p.parse_args()
    if args.dt <= 0:
        p.error("--dt must be positive")
    steps = max(1, round(args.presentation / args.dt))
    rest_steps = round(args.rest / args.dt)

    device = torch.device(args.device)
    graph = args.stepper if device.type == "cuda" and not args.no_graph else False
    xtr, ytr, xte, yte = load_mnist(args.data, download=True)
    xtr, xte = xtr.reshape(len(xtr), -1), xte.reshape(len(xte), -1)
    torch.manual_seed(args.seed)
    if args.members < 1:
        p.error("--members must be at least 1")
    model = build_network(
        args.neurons,
        device,
        graph=graph,
        seed=args.seed,
        rule=args.rule,
        members=args.members if args.members > 1 else None,
        dt=args.dt,
    )
    stats = train(
        model,
        xtr,
        ytr,
        samples=args.train_samples,
        epochs=args.epochs,
        steps=steps,
        log_every=args.log_every,
        checkpoint=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        rest=rest_steps,
    )
    n_label = min(args.label_samples, len(xtr))
    n_test = min(args.test_samples, len(xte))
    result = evaluate(
        model,
        xtr[:n_label],
        ytr[:n_label],
        xte[:n_test],
        yte[:n_test],
        batch=args.eval_batch,
        steps=steps,
        graph=graph,
        seed=args.seed,
        dt=args.dt,
    )
    result.update(
        rule=args.rule,
        rest=args.rest,
        dt=args.dt,
        presentation=args.presentation,
        steps_per_sample=steps,
        rest_steps=rest_steps,
        winner_diagnostics=stats["winner_diagnostics"],
        training_seconds=stats["seconds"],
        training_samples=stats["samples"],
        training_samples_per_second=stats["samples"] / stats["seconds"],
        training_steps_per_second=stats["sim_steps"] / stats["seconds"],
        training_repeats=stats["repeats"],
        running_accuracy=stats["running_accuracy"],
        parameters={**vars(args), "graph": graph},
        machine=machine_info(device),
    )
    if args.members > 1:
        result.update(
            members=args.members,
            training_member_samples_per_second=args.members * stats["samples"] / stats["seconds"],
        )
        print(
            f"accuracy {result['accuracy']:.4f} +- {result['accuracy_std']:.4f} "
            f"(mean +- std of {args.members} members) on {n_test} test images; "
            f"per member {[round(a, 4) for a in result['accuracies']]}"
        )
    else:
        print(f"accuracy {result['accuracy']:.4f} on {n_test} test images")
    print(
        f"training {stats['seconds']:.0f}s, evaluation {result['evaluation_seconds']:.0f}s",
        flush=True,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
