"""Batched TEM optimization and evaluation on freshly labelled worlds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import monotonic
from typing import TypedDict, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from neurosush.spatial import GraphWorld, Walk
from neurosush.spatial.batch import BatchedWalk, batched_random_walks, make_worlds
from neurosush.spatial.controls import seen_targets, sensory_count_predictions
from neurosush.spatial.probe import position_probe

from .config import TEMConfig
from .diagnostics import saturation
from .loss import loss_weights, total_loss
from .model import TEM, TEMState


@dataclass(frozen=True)
class TrainResult:
    """Trained model, per-update raw losses and schedules, and elapsed seconds."""

    model: TEM
    history: list[dict[str, float]]
    config: TEMConfig
    seconds: float
    state_dict_path: str | None = None


class Evaluation(TypedDict):
    """Accuracies share the novel-edge-to-seen mask; empty bins are NaN."""

    accuracy: float
    generative_ce: float
    sensory_only: float
    frequency: float
    conditional: float
    node_conditional: float
    events: int
    position_r2: float
    shift_null_p95: float
    by_fraction: dict[str, dict[str, float]]


def _worlds(batch: int, config: TEMConfig, seed: int) -> list[GraphWorld]:
    if min(config.widths) >= 5:
        return make_worlds(batch, widths=config.widths, sensory=config.sensory, seed=seed)
    # The batch helper does not expose GraphWorld's small-world diagnostic flag.
    return [
        GraphWorld(
            config.widths[i % len(config.widths)],
            sensory=config.sensory,
            seed=seed * 100003 + i,
            diagnostic_small=True,
        )
        for i in range(batch)
    ]


def _row(walk: BatchedWalk, row: int) -> Walk:
    return Walk(
        walk.nodes[row],
        walk.observations[row],
        walk.actions[row],
        walk.novel_edge_to_seen[row],
        walk.fraction_seen[row],
        walk.executed_actions[row],
    )


def _oracle(nodes: Tensor, config: TEMConfig) -> tuple[Tensor, ...]:
    if len(set(config.widths)) != 1:
        raise ValueError(f"oracle needs a single width, got {config.widths}")
    width = config.widths[0]
    side = width + 2
    if any(n < side * side for n in config.modules):
        raise ValueError(
            f"oracle modules={config.modules} must each fit lattice size={side * side}"
        )
    index = (nodes // width + 1) * side + nodes % width + 1
    return tuple(F.one_hot(index, n).float() for n in config.modules)


def _stage_walk(model: TEM, walk: BatchedWalk) -> BatchedWalk:
    model._validate_ranges(walk.observations, walk.executed_actions)
    device = model.alpha.device
    return replace(
        walk,
        observations=walk.observations.to(device),
        executed_actions=walk.executed_actions.to(device),
        nodes=walk.nodes.to(device),
    )


def _step(
    model: TEM,
    walk: BatchedWalk,
    t: int,
    state: TEMState,
    *,
    decay: float,
    rate: float,
    oracle: bool,
) -> tuple[Tensor, Tensor, TEMState]:
    observation = walk.observations[:, t]
    action = walk.executed_actions[:, t - 1] if t else torch.full_like(observation, 4)
    override = _oracle(walk.nodes[:, t], model.config) if oracle else None
    output, state = model._forward(
        observation,
        action,
        state,
        decay=decay,
        rate=rate,
        prior_override=override,
    )
    return output.logits_gen, output.g_inf, state


def train(
    config: TEMConfig,
    *,
    steps: int,
    seed: int,
    device: str = "cpu",
    log_every: int = 0,
    oracle: bool = False,
    diagnostics: bool = False,
    log_grads: bool = False,
    grad_every: int = 100,
    on_update: Callable[[list[dict[str, float]]], None] | None = None,
) -> TrainResult:
    """Fit slow weights with Adam and truncated BPTT through fast memory.

    ``steps`` counts transitions per batch row, rather than optimizer updates.
    Walks reset every ``walk_len`` transitions. Only previously seen targets
    contribute loss. World seeds and torch randomness are explicit and local.
    ``oracle`` replaces g with exact one-hot positions for the memory diagnostic;
    its lattice must be larger than the world and fit each configured module.
    ``diagnostics`` records per-parameter gradients before clipping and latent
    saturation fractions without changing optimization. Per-term gradient norms
    are sampled every grad_every updates (including zero) when log_grads is set.
    """
    if grad_every < 1:
        raise ValueError(f"grad_every must be positive, got {grad_every}")
    if steps < 1 or log_every < 0:
        raise ValueError(f"need steps >= 1 and log_every >= 0, got {steps}, {log_every}")
    start = monotonic()
    generator = torch.Generator(device=device).manual_seed(seed)
    model = TEM(config, generator=generator).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_max)
    history: list[dict[str, float]] = []
    completed, walk_number = 0, 0
    while completed < steps:
        length = min(config.walk_len, steps - completed)
        worlds = _worlds(config.batch, config, seed + walk_number)
        walk = batched_random_walks(worlds, length, stay=False, valid_only=True)
        mask = torch.stack([seen_targets(_row(walk, b)) for b in range(config.batch)]).to(device)
        walk = _stage_walk(model, walk)
        state = model.reset(config.batch)
        for begin in range(1, length + 1, config.bptt):
            iteration = len(history)
            lr, decay, rate, _ = config.schedules(iteration)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if begin == 1:
                _, _, state = _step(
                    model,
                    walk,
                    0,
                    state,
                    decay=decay,
                    rate=rate,
                    oracle=oracle,
                )
            measure_grads = log_grads and iteration % grad_every == 0
            losses: list[Tensor] = []
            terms: dict[str, list[Tensor]] = {}
            grad_terms: dict[str, list[Tensor]] = {}
            for t in range(begin, min(begin + config.bptt, length + 1)):
                observation = walk.observations[:, t]
                override = _oracle(walk.nodes[:, t], config) if oracle else None
                output, state = model._forward(
                    observation,
                    walk.executed_actions[:, t - 1],
                    state,
                    decay=decay,
                    rate=rate,
                    prior_override=override,
                )
                loss, values = total_loss(
                    output, observation, config, iteration=iteration, mask=mask[:, t - 1]
                )
                losses.append(loss)
                if measure_grads:
                    for name, value in values.items():
                        grad_terms.setdefault(name, []).append(value)
                if diagnostics:
                    values.update(saturation(output))
                for name, value in values.items():
                    terms.setdefault(name, []).append(value.detach())
            optimizer.zero_grad()
            loss = torch.stack(losses).mean()
            term_norms: dict[str, Tensor] = {}
            if measure_grads:
                weights = loss_weights(config, iteration)
                parameters = tuple(p for p in model.parameters() if p.requires_grad)
                for name, term_values in grad_terms.items():
                    term = torch.stack(term_values).mean() * weights[name]
                    grads = torch.autograd.grad(
                        term, parameters, retain_graph=True, allow_unused=True
                    )
                    term_norms[f"grad_term/{name}"] = (
                        torch.stack([g.detach().square().sum() for g in grads if g is not None])
                        .sum()
                        .sqrt()
                    )
            torch.autograd.backward(loss)
            gradients = (
                {
                    f"grad/{name}": parameter.grad.detach().norm()
                    if parameter.grad is not None
                    else loss.new_zeros(())
                    for name, parameter in model.named_parameters()
                }
                if diagnostics
                else {}
            )
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()
            state = state.detach()
            metrics = {name: torch.stack(values).mean() for name, values in terms.items()}
            metrics.update(loss=loss.detach(), grad_norm=norm)
            metrics.update(gradients)
            metrics.update(term_norms)
            values_cpu = torch.stack(list(metrics.values())).cpu().tolist()
            transferred = dict(zip(metrics, values_cpu, strict=True))
            record = {name: transferred[name] for name in terms}
            record.update(
                loss=transferred["loss"],
                lr=lr,
                memory_decay=decay,
                memory_rate=rate,
                grad_norm=transferred["grad_norm"],
            )
            record.update({name: transferred[name] for name in gradients})
            record.update({name: transferred[name] for name in term_norms})
            history.append(record)
            if on_update is not None:
                on_update(history)
            if log_every and len(history) % log_every == 0:
                print(f"update={len(history)} loss={record['loss']:.5f}", flush=True)
        completed += length
        walk_number += 1
    return TrainResult(model, history, config, monotonic() - start)


def _additional_controls(walk: Walk, sensory: int, size: int) -> dict[str, Tensor]:
    # Online source-label and source-node controls see only completed transitions.
    label_counts = torch.zeros(sensory, sensory)
    node_counts = torch.zeros(size, sensory)
    frequency = sensory_count_predictions(walk, sensory)["frequency"]
    labels, nodes = [], []
    for t in range(len(walk.actions)):
        if t:
            source, target = walk.observations[t - 1], walk.observations[t]
            label_counts[source, target] += 1
            node_counts[walk.nodes[t - 1], target] += 1
        label_row = label_counts[walk.observations[t]]
        node_row = node_counts[walk.nodes[t]]
        labels.append(label_row.argmax() if label_row.sum() else frequency[t])
        nodes.append(node_row.argmax() if node_row.sum() else frequency[t])
    return {"sensory_only": torch.stack(labels), "node_conditional": torch.stack(nodes)}


def _accuracy(correct: Tensor, mask: Tensor) -> float:
    return float(correct[mask].float().mean()) if bool(mask.any()) else float("nan")


def evaluate(
    model: TEM,
    config: TEMConfig,
    *,
    environments: int,
    steps: int,
    seed: int,
    oracle: bool = False,
) -> Evaluation:
    """Evaluate frozen slow weights in newly sampled environments.

    Each environment starts with empty memory. Novel directed edges into seen
    nodes define the prediction score. Sensory-only is an online source-label
    predictor; conditional additionally knows the action. Node-conditional knows
    the source node, but no action. All count controls fall back to frequency.
    Fraction bins use this same event mask, with coverage measured before moving.
    The position probe is fitted separately within each environment on inferred g;
    R2 and each environment's shift-null p95 are averaged across environments.
    Choose evaluation seeds disjoint from all training world seeds.
    """
    if environments < 1 or steps < 10:
        raise ValueError(f"need environments >= 1 and steps >= 10, got {environments}, {steps}")
    if config != model.config:
        raise ValueError(f"config={config} differs from model.config={model.config}")
    worlds = _worlds(environments, config, seed)
    walk = batched_random_walks(worlds, steps, stay=False, valid_only=True)
    staged_walk = _stage_walk(model, walk)
    was_training = model.training
    model.eval()
    predictions, activities = [], []
    cross_entropies: list[Tensor] = []
    try:
        with torch.no_grad():
            state = model.reset(environments)
            _, _, state = _step(
                model,
                staged_walk,
                0,
                state,
                decay=config.memory_decay,
                rate=config.memory_rate,
                oracle=oracle,
            )
            for t in range(1, steps + 1):
                logits, g, state = _step(
                    model,
                    staged_walk,
                    t,
                    state,
                    decay=config.memory_decay,
                    rate=config.memory_rate,
                    oracle=oracle,
                )
                cross_entropies.append(
                    F.cross_entropy(logits, staged_walk.observations[:, t], reduction="none").cpu()
                )
                predictions.append(logits.argmax(-1).cpu())
                activities.append(g.cpu())
    finally:
        model.train(was_training)
    prediction = torch.stack(predictions, 1)
    activity = torch.stack(activities, 1)
    target = walk.observations[:, 1:]
    correct = {"accuracy": prediction == target}
    controls: dict[str, list[Tensor]] = {
        name: [] for name in ("sensory_only", "frequency", "conditional", "node_conditional")
    }
    probes = []
    for b, world in enumerate(worlds):
        row = _row(walk, b)
        counts = sensory_count_predictions(row, config.sensory)
        counts.update(_additional_controls(row, config.sensory, world.size))
        for name, values in counts.items():
            controls[name].append(values)
        probes.append(position_probe(activity[b], row.nodes[1:], world.width))
    for name, series in controls.items():
        correct[name] = torch.stack(series) == target
    event = walk.novel_edge_to_seen
    scores = {name: _accuracy(values, event) for name, values in correct.items()}
    by_fraction: dict[str, dict[str, float]] = {}
    for low, high in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        mask = event & (walk.fraction_seen >= low) & (walk.fraction_seen < high)
        bin_scores = {name: _accuracy(value, mask) for name, value in correct.items()}
        bin_scores["events"] = float(mask.sum())
        by_fraction[f"{low:.2f}-{min(high, 1.0):.2f}"] = bin_scores
    return Evaluation(
        accuracy=scores["accuracy"],
        generative_ce=float(torch.stack(cross_entropies, 1)[event].mean())
        if bool(event.any())
        else float("nan"),
        sensory_only=scores["sensory_only"],
        frequency=scores["frequency"],
        conditional=scores["conditional"],
        node_conditional=scores["node_conditional"],
        events=int(event.sum()),
        position_r2=sum(cast(float, p["r2"]) for p in probes) / environments,
        shift_null_p95=sum(cast(float, p["shift_null_p95"]) for p in probes) / environments,
        by_fraction=by_fraction,
    )
