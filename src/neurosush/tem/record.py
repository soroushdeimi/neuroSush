"""Recorded-run exporter for TEM matching the shared viewer contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from neurosush.spatial import GraphWorld, rate_maps
from neurosush.tem.checkpoint import load_checkpoint
from neurosush.tem.config import TEMConfig
from neurosush.tem.model import TEM


def _maps(a: Tensor, pos: Tensor, size: int) -> tuple[Tensor, Tensor]:
    """Return node-first means and occupancy."""
    res = rate_maps(a, pos, (size, size))
    assert res is not None
    m_raw, c_raw = res
    assert m_raw is not None
    assert c_raw is not None
    m_tensor: Tensor = m_raw
    c_tensor: Tensor = c_raw
    return m_tensor.reshape(a.shape[1], size * size).T, c_tensor.flatten()


def _information(a: Tensor, pos: Tensor, size: int) -> Tensor:
    """Compute spatial information for units given activity and node positions."""
    m, c = _maps(a, pos, size)
    p = c[c > 0].double() / c.sum()
    r = m[c > 0]
    r = torch.where(torch.isfinite(r), r, torch.tensor(0.0, dtype=r.dtype))
    mean_rate = (r * p[:, None]).sum(0)
    ratio = r / mean_rate.clamp_min(1e-12)
    terms = torch.where(ratio > 0, ratio * ratio.clamp_min(1e-300).log2(), 0)
    return (p[:, None] * terms).sum(0)


def record_run(
    checkpoint_path: str | Path | None = None,
    *,
    worlds: int = 3,
    size: int = 8,
    steps: int = 400,
    units: int = 16,
    seed: int = 42,
    model: TEM | None = None,
    train_seed: int = 0,
    updates: int = 0,
    label: str = "learned structure",
    note: str = "one seed; no claim of grid cells",
    probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record random walks, predictions, and top-unit rate maps across worlds."""
    if worlds < 1 or size < 2 or steps < 1 or units < 1:
        raise ValueError("worlds, size, steps, and units must be positive")

    variant = "v2"
    if model is None:
        if checkpoint_path is not None:
            path_str = str(checkpoint_path)
            model, _train_seed, updates = load_checkpoint(checkpoint_path)
            if "v1" in path_str:
                variant = "v1"
            elif "v3" in path_str:
                variant = "v3"
        else:
            config = TEMConfig(widths=(5, 6, 7, 8))
            model = TEM(config, generator=torch.Generator().manual_seed(seed))

    model.eval()
    worlds_list: list[dict[str, Any]] = []

    with torch.no_grad():
        for w in range(worlds):
            world = GraphWorld(size, sensory=model.config.sensory, seed=seed * 100003 + w)
            walk = world.random_walk(steps, stay=False, valid_only=True)
            state = model.reset(1)

            walk_coords: list[list[int]] = []
            pred_correct_list: list[int] = []
            first_visit_list: list[int] = []
            g_acts: list[Tensor] = []
            p_acts: list[Tensor] = []

            for t in range(steps + 1):
                obs = walk.observations[t : t + 1]
                if t == 0:
                    action = torch.tensor([4])
                else:
                    exec_actions = walk.executed_actions
                    actions = walk.actions
                    action = (
                        exec_actions[t - 1 : t]
                        if exec_actions is not None
                        else actions[t - 1 : t]
                    )
                out, state = model(obs, action, state)

                if t >= 1:
                    node = int(walk.nodes[t].item())
                    walk_coords.append([node % size, node // size])
                    pred = int(out.logits_gen.argmax(-1).item())
                    target = int(walk.observations[t].item())
                    pred_correct_list.append(1 if pred == target else 0)
                    first_visit_list.append(1 if walk.novel_edge_to_seen[t - 1].item() else 0)
                    g_acts.append(out.g_inf.squeeze(0).cpu())
                    p_acts.append(out.p_inf.squeeze(0).cpu())

            g_tensor = torch.stack(g_acts)
            p_tensor = torch.stack(p_acts)
            nodes_tensor = walk.nodes[1:]

            rate_dicts: dict[str, list[list[float]]] = {}
            for name, tensor in (("g", g_tensor), ("p", p_tensor)):
                res = rate_maps(tensor, nodes_tensor, (size, size))
                assert res is not None
                m_raw, _ = res
                assert m_raw is not None
                m_tensor: Tensor = m_raw
                si = _information(tensor, nodes_tensor, size)
                si = torch.where(torch.isfinite(si), si, -float("inf"))
                top_u = min(units, tensor.shape[1])
                top_indices = si.argsort(descending=True)[:top_u]

                maps_list = []
                for u in top_indices.tolist():
                    rm = m_tensor[u].flatten()
                    rm = torch.where(torch.isnan(rm), torch.tensor(0.0, dtype=rm.dtype), rm)
                    maps_list.append([round(float(v), 3) for v in rm.tolist()])
                rate_dicts[name] = maps_list

            worlds_list.append(
                {
                    "id": w,
                    "size": size,
                    "sensory": [int(s) for s in world.labels.tolist()],
                    "walk": walk_coords,
                    "pred_correct": pred_correct_list,
                    "first_visit_new_edge": first_visit_list,
                    "rate_maps": rate_dicts,
                }
            )

    return {
        "version": 1,
        "meta": {
            "variant": variant,
            "seed": seed,
            "updates": updates,
            "label": label,
            "note": note,
        },
        "worlds": worlds_list,
        "probe": probe,
    }


def write_run(path: str | Path, run: dict[str, Any]) -> None:
    """Atomically write run dict as compact JSON under 300 KB."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(run, allow_nan=False)
    if len(content.encode("utf-8")) >= 300 * 1024:
        raise ValueError("run json payload exceeds 300 KB limit")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(content + "\n")
    temporary.replace(path)
