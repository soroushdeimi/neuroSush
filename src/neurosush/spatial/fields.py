"""Evaluation helpers for spatial and grid field metrics."""

from __future__ import annotations

import math

import torch

from neurosush.spatial import (
    circular_linear_regression,
    gridness,
    phase_slope_class,
    rate_maps,
    shuffled_position_control,
    spatial_information,
)


def grid_fraction(
    activity: torch.Tensor,
    nodes: torch.Tensor,
    shape: tuple[int, ...],
    *,
    threshold: float = 0.8,
) -> dict[str, float | int | str | bool]:
    """Compute fraction of units exceeding hexagonal or square grid score thresholds."""
    maps, _occupancy = rate_maps(activity, nodes, shape)
    n_units = maps.shape[0]
    hex_scores = []
    square_scores = []
    resolvable_flags = []
    undefined_count = 0

    for i in range(n_units):
        rate = maps[i]
        if not torch.isfinite(rate).all() or rate.isnan().all():
            undefined_count += 1
            continue
        g = gridness(rate)
        if math.isnan(g.hexagonal) or math.isnan(g.square):
            undefined_count += 1
            continue
        hex_scores.append(g.hexagonal)
        square_scores.append(g.square)
        resolvable_flags.append(g.resolvable)

    valid_units = len(hex_scores)
    if valid_units == 0:
        hex_fraction = 0.0
        square_fraction = 0.0
        resolvable = False
    else:
        hex_fraction = sum(1 for s in hex_scores if s >= threshold) / valid_units
        square_fraction = sum(1 for s in square_scores if s >= threshold) / valid_units
        resolvable = all(resolvable_flags)

    label = "square/diamond periodicity" if square_fraction > hex_fraction else "hexagonal-score"
    result: dict[str, float | int | str | bool] = {
        "hex_fraction": hex_fraction,
        "square_fraction": square_fraction,
        "n_units": n_units,
        "undefined_count": undefined_count,
        "resolvable": resolvable,
        "label": label,
    }
    if not resolvable:
        result["caveat"] = (
            "Maps are below minimum size for reliable grid geometry resolution (resolvable=False)."
        )
    return result


def spatial_information_vs_shuffle(
    activity: torch.Tensor,
    nodes: torch.Tensor,
    shape: tuple[int, ...],
    *,
    n_shuffles: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    """Compare unit spatial information against time-shuffled position nulls."""
    maps, occupancy = rate_maps(activity, nodes, shape)
    info = spatial_information(maps, occupancy)
    mean_info = float(info.mean().item())

    null = shuffled_position_control(activity, nodes, shape, repeats=n_shuffles, seed=seed)
    shuffle_infos = null.information
    shuffle_mean = float(shuffle_infos.mean().item())
    p95_per_unit = torch.quantile(shuffle_infos, 0.95, dim=0)
    shuffle_p95 = float(p95_per_unit.mean().item())
    fraction_above_p95 = float((info > p95_per_unit).float().mean().item())

    return {
        "mean_info": mean_info,
        "shuffle_mean": shuffle_mean,
        "shuffle_p95": shuffle_p95,
        "fraction_above_p95": fraction_above_p95,
    }


def track_phase_slopes(
    spike_positions: torch.Tensor,
    spike_phases: torch.Tensor,
    spike_directions: torch.Tensor,
    directions: list[int] | torch.Tensor,
    *,
    field_edges: list[tuple[float, float]],
) -> list[dict[str, float | int | str | tuple[float, float] | None]]:
    """Fit circular-linear regression on spike phases vs positions.

    For field segments and directions.
    """
    records: list[dict[str, float | int | str | tuple[float, float] | None]] = []
    pos = spike_positions.detach().cpu()
    phases = spike_phases.detach().cpu()
    dirs = spike_directions.detach().cpu()

    for lo, hi in field_edges:
        for direction in directions:
            mask = (pos >= lo) & (pos <= hi) & (dirs == direction)
            p_seg = pos[mask]
            ph_seg = phases[mask]
            n_spikes = int(mask.sum().item())

            if n_spikes < 5:
                rec: dict[str, float | int | str | tuple[float, float] | None] = {
                    "field": (lo, hi),
                    "direction": int(direction),
                    "slope": float("nan"),
                    "concentration": float("nan"),
                    "class": None,
                    "n_spikes": n_spikes,
                }
                records.append(rec)
            else:
                reg = circular_linear_regression(p_seg, ph_seg, direction=int(direction))
                cls = phase_slope_class(reg.slope)
                rec = {
                    "field": (lo, hi),
                    "direction": int(direction),
                    "slope": reg.slope,
                    "concentration": reg.concentration,
                    "class": cls,
                    "n_spikes": n_spikes,
                }
                records.append(rec)
    return records


def summarize_classes(
    records: list[dict[str, float | int | str | tuple[float, float] | None]],
) -> dict[str, dict[str, int] | dict[str, float]]:
    """Summarize counts and fractions of phase classification records."""
    categories = ["locked", "precession", "independent", "None"]
    counts = dict.fromkeys(categories, 0)
    total = len(records)

    for rec in records:
        c = rec.get("class")
        if c is None:
            counts["None"] += 1
        elif isinstance(c, str) and c in counts:
            counts[c] += 1
        else:
            counts["None"] += 1

    fractions = {cat: (counts[cat] / total if total > 0 else 0.0) for cat in categories}
    return {"counts": counts, "fractions": fractions}
