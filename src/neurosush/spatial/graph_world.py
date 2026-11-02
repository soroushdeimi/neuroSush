"""Discrete worlds for spatial representation and navigation.

Deviations:
    Square widths 5--8 (2--3 only with diagnostic_small=True) reduce the
    original TEM's 8--11 (2020 PDF p.18).
    The spiking demo uses width 8 and a 40-position track (2025 PDF pp.10,16).
    Boundaries reject moves; no wrapping, hex graphs, or continuous locomotion.
    Observations are sampled with replacement and may alias different nodes.
    Position/node IDs are evaluation metadata, never network inputs.
    Executed actions expose boundary rejection as stay without coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Walk:
    """Positions/observations have T+1 entries; actions have T entries."""

    nodes: torch.Tensor
    observations: torch.Tensor
    actions: torch.Tensor
    novel_edge_to_seen: torch.Tensor
    fraction_seen: torch.Tensor
    executed_actions: torch.Tensor | None = None


class GraphWorld:
    """Square graph or linear track, with an explicit local CPU random generator.

    Actions are right, left, up, down, stay for squares, and right, left,
    stay for tracks. Invalid boundary actions leave the agent in place.
    """

    def __init__(
        self,
        width: int = 8,
        *,
        track: bool = False,
        sensory: int = 20,
        seed: int = 0,
        diagnostic_small: bool = False,
    ) -> None:
        if (track and width != 40) or (
            not track and not (2 if diagnostic_small else 5) <= width <= 16
        ):
            raise ValueError(
                f"invalid width={width}, track={track}, diagnostic_small={diagnostic_small}"
            )
        if sensory < 1:
            raise ValueError(f"sensory must be positive, got {sensory}")
        self.width: int = width
        self.track: bool = track
        self.sensory: int = sensory
        self.size: int = width if track else width * width
        self.generator: torch.Generator = torch.Generator().manual_seed(seed)
        self.labels: torch.Tensor = torch.randint(sensory, (self.size,), generator=self.generator)
        self.deltas: tuple[tuple[int, int], ...] = (
            ((1, 0), (-1, 0), (0, 0)) if track else ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))
        )

    def step(self, node: int, action: int) -> int:
        """Return the destination for one action; reject out-of-range values."""
        if not 0 <= node < self.size or not 0 <= action < len(self.deltas):
            raise ValueError(f"invalid node={node} or action={action}")
        x, y = node % self.width, node // self.width
        dx, dy = self.deltas[action]
        nx, ny = x + dx, y + dy
        if not 0 <= nx < self.width or not 0 <= ny < (1 if self.track else self.width):
            return node
        return ny * self.width + nx

    def random_walk(
        self,
        steps: int,
        *,
        start: int | None = None,
        stay: bool = True,
        valid_only: bool = False,
    ) -> Walk:
        """Sample uniformly over actions, including rejected boundary moves.

        A novel edge is directed and must change position. Its destination must
        have been visited before this transition; the fraction is also measured
        before the transition, avoiding future information in evaluation bins.
        valid_only samples legal moves, a diagnostic policy that avoids hidden
        boundary rejections without giving coordinates to the model.
        """
        if steps < 0:
            raise ValueError(f"steps must be nonnegative, got {steps}")
        node = (
            int(torch.randint(self.size, (), generator=self.generator)) if start is None else start
        )
        if not 0 <= node < self.size:
            raise ValueError(f"invalid start={node}")
        actions = torch.randint(len(self.deltas) - (not stay), (steps,), generator=self.generator)
        nodes: list[int] = [node]
        novel: list[bool] = []
        fractions: list[float] = []
        visited: set[int] = {node}
        edges: set[tuple[int, int]] = set()
        for t, action_tensor in enumerate(actions):
            action = int(action_tensor.item())
            if valid_only:
                choices = [
                    a
                    for a in range(len(self.deltas) - (not stay))
                    if self.step(node, a) != node or self.deltas[a] == (0, 0)
                ]
                action = choices[int(torch.randint(len(choices), (), generator=self.generator))]
                actions[t] = action
            destination = self.step(node, action)
            novel.append(
                destination != node and destination in visited and (node, destination) not in edges
            )
            fractions.append(len(visited) / self.size)
            edges.add((node, destination))
            visited.add(destination)
            node = destination
            nodes.append(node)
        ids = torch.tensor(nodes, dtype=torch.long)
        return Walk(
            nodes=ids,
            observations=self.labels[ids],
            actions=actions,
            novel_edge_to_seen=torch.tensor(novel, dtype=torch.bool),
            fraction_seen=torch.tensor(fractions, dtype=torch.float32),
            executed_actions=torch.where(
                ids[1:] == ids[:-1],
                torch.full_like(actions, 4 if not self.track else 2),
                actions,
            ),
        )

    def shuttle(self, laps: int) -> torch.Tensor:
        """Return alternating complete track traversals, without duplicate endpoints."""
        if not self.track or laps < 1:
            raise ValueError(
                f"shuttle needs a track and positive laps, got track={self.track}, laps={laps}"
            )
        forward = torch.arange(self.width)
        return torch.cat(
            [
                forward if i == 0 else (forward.flip(0) if i % 2 else forward)[1:]
                for i in range(laps)
            ]
        )
