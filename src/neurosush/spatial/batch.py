"""Batched random walks across multiple graph worlds."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neurosush.spatial.graph_world import GraphWorld


@dataclass(frozen=True)
class BatchedWalk:
    """Batched walk across independent graph world environments.

    Each batch row is an independent environment; node IDs are only comparable
    within a row.
    """

    nodes: torch.Tensor
    observations: torch.Tensor
    actions: torch.Tensor
    executed_actions: torch.Tensor
    novel_edge_to_seen: torch.Tensor
    fraction_seen: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Return the batch size B."""
        return int(self.nodes.shape[0])

    @property
    def steps(self) -> int:
        """Return the number of steps T."""
        return int(self.actions.shape[1])


def make_worlds(
    batch: int,
    *,
    widths: tuple[int, ...] = (5, 6, 7, 8),
    sensory: int = 20,
    seed: int = 0,
    track: bool = False,
) -> list[GraphWorld]:
    """Create a list of GraphWorld instances with cycling widths and derived seeds."""
    if batch < 1:
        raise ValueError(f"batch must be at least 1, got {batch}")
    worlds: list[GraphWorld] = []
    for i in range(batch):
        width = 40 if track else widths[i % len(widths)]
        sub_seed = seed * 100003 + i
        worlds.append(GraphWorld(width, track=track, sensory=sensory, seed=sub_seed))
    return worlds


def batched_random_walks(
    worlds: list[GraphWorld],
    steps: int,
    *,
    stay: bool = False,
    valid_only: bool = True,
) -> BatchedWalk:
    """Generate batched random walks from a list of GraphWorld instances."""
    if not worlds:
        raise ValueError(f"worlds list cannot be empty, got {worlds}")
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")

    walks = [w.random_walk(steps, stay=stay, valid_only=valid_only) for w in worlds]

    executed_list = []
    for w in walks:
        exec_act = w.executed_actions
        if exec_act is None:
            exec_act = w.actions
        executed_list.append(exec_act)

    return BatchedWalk(
        nodes=torch.stack([w.nodes for w in walks]),
        observations=torch.stack([w.observations for w in walks]),
        actions=torch.stack([w.actions for w in walks]),
        executed_actions=torch.stack(executed_list),
        novel_edge_to_seen=torch.stack([w.novel_edge_to_seen for w in walks]),
        fraction_seen=torch.stack([w.fraction_seen for w in walks]),
    )
