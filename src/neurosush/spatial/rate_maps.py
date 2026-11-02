"""Rate maps, spatial information, and position-shuffling control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from neurosush.spatial.gridness import gridness, spatial_autocorrelogram


def rate_maps(
    activity: torch.Tensor, nodes: torch.Tensor, shape: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit-first occupancy means and counts; unseen bins remain NaN.

    Activity is (samples, units); each sample has equal duration. For spikes,
    divide by sample duration in seconds first to obtain Hz.
    """
    activity, nodes = _prepare_samples(activity, nodes, shape)
    counts = torch.bincount(nodes, minlength=math.prod(shape))
    return _accumulate_maps(activity, nodes, shape, counts), counts.reshape(shape)


def _prepare_samples(
    activity: torch.Tensor, nodes: torch.Tensor, shape: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    activity = activity.detach().to(device="cpu", dtype=torch.float64)
    nodes = nodes.detach().to(device="cpu", dtype=torch.long)
    size = math.prod(shape)
    if not shape or min(shape) < 1 or activity.ndim != 2 or nodes.shape != activity.shape[:1]:
        raise ValueError("expected positive shape, activity (samples, units), nodes (samples,)")
    if not torch.isfinite(activity).all() or (nodes < 0).any() or (nodes >= size).any():
        raise ValueError("activity must be finite and nodes must lie within shape")
    return activity, nodes


def _accumulate_maps(
    activity: torch.Tensor, nodes: torch.Tensor, shape: tuple[int, ...], counts: torch.Tensor
) -> torch.Tensor:
    total = torch.zeros(counts.numel(), activity.shape[1], dtype=activity.dtype)
    total.index_add_(0, nodes, activity)
    means = total / counts[:, None].clamp_min(1)
    means[counts == 0] = float("nan")
    return means.T.reshape(activity.shape[1], *shape)


def spatial_information(rate: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
    """Skaggs information in bits/spike for nonnegative unit-first rate maps.

    Unoccupied bins are excluded; silent cells return zero. Finite rates are
    required at occupied bins. This is not a bias-corrected significance test.
    """
    counts = occupancy.detach().double().cpu().flatten()
    if (counts < 0).any() or not torch.isfinite(counts).all() or counts.sum() <= 0:
        raise ValueError("occupancy must be finite, nonnegative and nonempty")
    if rate.ndim != occupancy.ndim + 1 or rate.shape[1:] != occupancy.shape:
        raise ValueError("rate must have shape (units, *occupancy.shape)")
    r = rate.detach().double().cpu().reshape(-1, occupancy.numel())
    r = r[:, counts > 0]
    if not torch.isfinite(r).all() or (r < 0).any():
        raise ValueError("occupied rates must be finite and nonnegative")
    p = counts[counts > 0] / counts.sum()
    mean = (r * p).sum(-1, keepdim=True)
    ratio = r / mean.clamp_min(torch.finfo(r.dtype).tiny)
    terms = torch.where(ratio > 0, ratio * ratio.clamp_min(1e-300).log2(), 0)
    return (p * terms).sum(-1)


@dataclass(frozen=True)
class ShuffleNull:
    """Position-shuffle null arrays of shape (repeats, units); grid scores NaN in 1D."""

    hexagonal: torch.Tensor
    square: torch.Tensor
    information: torch.Tensor


def shuffled_position_control(
    activity: torch.Tensor,
    nodes: torch.Tensor,
    shape: tuple[int, ...],
    *,
    repeats: int = 32,
    seed: int = 0,
    annulus: tuple[float, float] | None = None,
) -> ShuffleNull:
    """Shuffle sample-wise node labels, then recompute maps and both spatial nulls.

    This preserves occupancy and per-unit activity distributions. Permuting a
    finished rate map would preserve spatial information at uniform occupancy
    and is NOT this control. IID label shuffling destroys temporal correlations;
    it is a simple diagnostic, not a substitute for lap/block-preserving controls.
    """
    if repeats < 1 or len(shape) not in (1, 2):
        raise ValueError(f"need positive repeats and 1D/2D shape, got {repeats}, {shape}")
    activity, labels = _prepare_samples(activity, nodes, shape)
    counts = torch.bincount(labels, minlength=math.prod(shape))
    occupancy = counts.reshape(shape)
    generator = torch.Generator().manual_seed(seed)
    hex_scores: list[list[float]] = []
    square_scores: list[list[float]] = []
    information: list[torch.Tensor] = []
    for _ in range(repeats):
        shuffled = labels[torch.randperm(len(labels), generator=generator)]
        maps = _accumulate_maps(activity, shuffled, shape, counts)
        information.append(spatial_information(maps, occupancy))
        scores = [gridness(m, annulus=annulus) for m in maps] if len(shape) == 2 else []
        hex_scores.append([s.hexagonal for s in scores] if scores else [math.nan] * len(maps))
        square_scores.append([s.square for s in scores] if scores else [math.nan] * len(maps))
    return ShuffleNull(
        torch.tensor(hex_scores, dtype=torch.float64),
        torch.tensor(square_scores, dtype=torch.float64),
        torch.stack(information),
    )


def _to_json_ready(obj: Any, decimals: int | None = 3) -> Any:
    """Recursively convert tensors, NaNs, and floats into JSON-safe types."""
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().tolist()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, decimals) if decimals is not None else obj
    if isinstance(obj, list):
        return [_to_json_ready(item, decimals=decimals) for item in obj]
    if isinstance(obj, tuple):
        return [_to_json_ready(item, decimals=decimals) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _to_json_ready(v, decimals=decimals) for k, v in obj.items()}
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return obj


def rate_map_export(
    activity: torch.Tensor,
    nodes: torch.Tensor,
    shape: tuple[int, ...],
    *,
    annulus: tuple[float, float] | None = None,
    decimals: int | None = 3,
) -> dict[str, Any]:
    """Turn per-node activity into JSON-ready spatial maps and autocorrelograms."""
    activity, nodes = _prepare_samples(activity, nodes, shape)
    maps, counts = rate_maps(activity, nodes, shape)
    occupancy = _to_json_ready(counts.tolist(), decimals=decimals)

    units_export = []
    si = spatial_information(maps, counts.double())
    for u in range(activity.shape[1]):
        rate = maps[u]
        rate_list = _to_json_ready(rate.tolist(), decimals=decimals)

        sac_list = None
        grid_dict = None
        if len(shape) == 2 and not rate.isnan().all():
            try:
                sac = spatial_autocorrelogram(rate)
                sac_list = _to_json_ready(sac.tolist(), decimals=decimals)
                g = gridness(rate, annulus=annulus)
                grid_dict = {
                    "hexagonal": _to_json_ready(g.hexagonal, decimals=decimals),
                    "square": _to_json_ready(g.square, decimals=decimals),
                    "resolvable": g.resolvable,
                    "annulus": _to_json_ready(list(g.annulus), decimals=decimals),
                    "correlations": _to_json_ready(g.correlations, decimals=decimals),
                }
            except Exception:
                pass

        units_export.append(
            {
                "unit": u,
                "rate_map": rate_list,
                "spatial_information": _to_json_ready(si[u].item(), decimals=decimals)
                if u < len(si)
                else None,
                "autocorrelogram": sac_list,
                "gridness": grid_dict,
            }
        )

    return {
        "shape": list(shape),
        "occupancy": occupancy,
        "units": units_export,
    }
