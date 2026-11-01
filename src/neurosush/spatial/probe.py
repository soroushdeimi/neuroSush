"""Spatial coordinate decoding probes and circular time-shift nulls."""

from __future__ import annotations

import torch


def coordinates(nodes: torch.Tensor, width: int) -> torch.Tensor:
    """Compute normalized (x, y) coordinates in [0, 1] from node indices."""
    if width < 2:
        raise ValueError(f"width must be >= 2, got {width}")
    x = nodes % width
    y = nodes // width
    return torch.stack((x, y), dim=1).double() / (width - 1)


def position_probe(
    g: torch.Tensor,
    nodes: torch.Tensor,
    width: int,
    *,
    shifts: int = 16,
    ridge: float = 1.0,
    train_fraction: float = 0.4,
    gap_fraction: float = 0.2,
) -> dict[str, float | list[float]]:
    """Time-separated ridge coordinate R2 and circular-shift position null."""
    if width < 2:
        raise ValueError(f"width must be >= 2, got {width}")
    if shifts < 1:
        raise ValueError(f"shifts must be >= 1, got {shifts}")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not (0.0 <= gap_fraction < 1.0):
        raise ValueError(f"gap_fraction must be in [0, 1), got {gap_fraction}")
    if train_fraction + gap_fraction >= 1.0:
        raise ValueError(
            f"train_fraction + gap_fraction must be < 1, got {train_fraction + gap_fraction}"
        )
    if g.shape[0] != nodes.shape[0]:
        raise ValueError(f"g length {g.shape[0]} does not match nodes length {nodes.shape[0]}")

    n = len(g)
    stop, start = int(n * train_fraction), int(n * (train_fraction + gap_fraction))
    mean, std = g[:stop].mean(0), g[:stop].std(0).clamp_min(1e-4)
    design = torch.cat(((g - mean) / std, torch.ones(n, 1)), dim=1).double()
    train, test = design[:stop], design[start:]
    ridge_mat = torch.eye(design.shape[1], dtype=torch.double) * ridge
    ridge_mat[-1, -1] = 0
    projector = torch.linalg.solve(train.T @ train + ridge_mat, train.T)
    coords = coordinates(nodes, width)

    def score(y: torch.Tensor) -> float:
        prediction = test @ projector @ y[:stop]
        denominator = (y[start:] - y[:stop].mean(0)).square().sum()
        return float(1 - (prediction - y[start:]).square().sum() / denominator.clamp_min(1e-12))

    null = [score(coords.roll(int(n * k / (shifts + 1)), dims=0)) for k in range(1, shifts + 1)]
    actual = score(coords)
    return {
        "r2": actual,
        "shift_null_mean": sum(null) / len(null),
        "shift_null_p95": float(torch.tensor(null).quantile(0.95)),
        "shift_null": null,
    }
