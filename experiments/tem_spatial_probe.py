"""Frozen TEM spatial measurements matching the reference analysis definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from neurosush.spatial import GraphWorld
from neurosush.spatial.rate_maps import rate_maps
from neurosush.tem import TEM, TEMConfig
from neurosush.tem.checkpoint import load_checkpoint


def maps(a: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
    """Return node-first 8x8 means and occupancy."""
    m, c = rate_maps(a, pos, (8, 8))
    return m.reshape(a.shape[1], 64).T, c.flatten()


def information(a: Tensor, pos: Tensor) -> Tensor:
    """Match reference SI, including its signed-activation convention."""
    m, c = maps(a, pos)
    p = c[c > 0].double() / c.sum()
    r = m[c > 0]
    ratio = r / (r * p[:, None]).sum(0).clamp_min(1e-12)
    return (p[:, None] * torch.where(ratio > 0, ratio * ratio.clamp_min(1e-300).log2(), 0)).sum(0)


def corr(a: Tensor, b: Tensor) -> float:
    """Pearson correlation on shared finite entries, undefined for constants."""
    ok = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[ok].double(), b[ok].double()
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack((a, b)))[0, 1])


def mean(values: list[float]) -> float:
    """Average defined measurements."""
    return float(torch.tensor(values, dtype=torch.float64).nanmean())


def shifted(pos: Tensor, rng: torch.Generator) -> Tensor:
    """Circular shift by at least 100 samples from either endpoint."""
    if len(pos) <= 200:
        raise ValueError("shift null requires more than 200 retained samples")
    return pos.roll(int(torch.randint(100, len(pos) - 100, (), generator=rng)))


def decoder(a: Tensor, pos: Tensor, rng: torch.Generator, repeats: int) -> tuple[float, float]:
    """Five contiguous test folds, 25-sample exclusion gap, fixed-prediction null."""
    edges = torch.linspace(0, len(pos), 6).long()
    pred = torch.full_like(pos, -1)
    for start, end in zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True):
        train_mask = torch.ones(len(pos), dtype=torch.bool)
        train_mask[max(0, start - 25) : min(len(pos), end + 25)] = False
        template, _ = maps(a[train_mask], pos[train_mask])
        # Reference nansum assigns zero distance to unvisited templates.
        distance = (a[start:end, None] - template[None]).square().nansum(-1)
        pred[start:end] = distance.argmin(-1)
    null = torch.tensor(
        [float((pred == shifted(pos, rng)).double().mean()) for _ in range(repeats)],
        dtype=torch.float64,
    )
    return float((pred == pos).double().mean()), float(null.quantile(0.95))


def measure(
    worlds: list[tuple[Tensor, Tensor, Tensor]],
    *,
    seed: int = 42,
    si_repeats: int = 32,
    repeats: int = 100,
) -> dict[str, float]:
    """Average within-world scores and pairwise cross-world geometry."""
    if len(worlds) < 2 or min(si_repeats, repeats) < 1:
        raise ValueError("need at least two worlds and positive null repeats")
    rng = torch.Generator().manual_seed(seed)
    rows: dict[str, list[float]] = {
        k: []
        for k in (
            "si",
            "perm_null",
            "circ_null",
            "frac_perm",
            "frac_circ",
            "splithalf",
            "dec",
            "dec_null",
            "map_corr",
            "rsa",
            "rsa_null_p95",
            "rsa_grid",
            "rsa_sensory",
        )
    }
    templates = []
    iu = torch.triu_indices(64, 64, 1)
    nodes = torch.arange(64)
    grid = (
        (nodes[:, None] // 8 - nodes[None] // 8).abs()
        + (nodes[:, None] % 8 - nodes[None] % 8).abs()
    )[iu[0], iu[1]]

    def dm(t: Tensor) -> Tensor:
        return (t[:, None] - t[None]).square().nansum(-1).sqrt()[iu[0], iu[1]]

    for a, pos, sensory in worlds:
        a = a.double()
        active = a.std(0) > 1e-9
        si = information(a, pos)
        perm = torch.stack(
            [
                information(a, pos[torch.randperm(len(pos), generator=rng)])
                for _ in range(si_repeats)
            ]
        )
        circ = torch.stack([information(a, shifted(pos, rng)) for _ in range(si_repeats)])
        for key, value in (
            ("si", si[active].mean()),
            ("perm_null", perm[:, active].mean()),
            ("circ_null", circ[:, active].mean()),
            ("frac_perm", (si > perm.quantile(0.95, dim=0))[active].double().mean()),
            ("frac_circ", (si > circ.quantile(0.95, dim=0))[active].double().mean()),
        ):
            rows[key].append(float(value))
        h = len(pos) // 2
        m1, c1 = maps(a[:h], pos[:h])
        m2, c2 = maps(a[h:], pos[h:])
        ok = (c1 > 0) & (c2 > 0)
        rows["splithalf"].append(
            mean([corr(m1[ok, u], m2[ok, u]) for u in active.nonzero().flatten().tolist()])
        )
        acc, null = decoder(a, pos, rng, repeats)
        rows["dec"].append(acc)
        rows["dec_null"].append(null)
        template, counts = maps(a, pos)
        templates.append((template, counts))
        sx = torch.tensor(
            [
                int(sensory[pos == q].bincount().argmax()) if (pos == q).any() else 0
                for q in range(64)
            ]
        )
        rows["rsa_grid"].append(corr(dm(template), grid))
        rows["rsa_sensory"].append(corr(dm(template), (sx[:, None] != sx[None])[iu[0], iu[1]]))
    null_rsa = []
    for i, (left, lc) in enumerate(templates):
        for right, rc in templates[i + 1 :]:
            ok = (lc > 0) & (rc > 0)
            uc = [
                corr(left[ok, u], right[ok, u])
                for u in range(left.shape[1])
                if left[ok, u].std() > 1e-6 and right[ok, u].std() > 1e-6
            ]
            rows["map_corr"].append(mean(uc))
            rows["rsa"].append(corr(dm(left), dm(right)))
            null_rsa.extend(
                corr(dm(left), dm(right[torch.randperm(64, generator=rng)])) for _ in range(repeats)
            )
    finite_null = torch.tensor(
        [v for v in null_rsa if torch.isfinite(torch.tensor(v))], dtype=torch.float64
    )
    rows["rsa_null_p95"] = [float(finite_null.quantile(0.95)) if len(finite_null) else float("nan")]
    within = {
        "si",
        "perm_null",
        "circ_null",
        "frac_perm",
        "frac_circ",
        "splithalf",
        "dec",
        "dec_null",
    }
    return {
        key: float(torch.tensor(values, dtype=torch.float64).mean())
        if key in within
        else mean(values)
        for key, values in rows.items()
    }


def extract(
    model: TEM, *, worlds: int, steps: int, seed: int, burn: int = 300
) -> dict[str, list[tuple[Tensor, Tensor, Tensor]]]:
    """Extract aligned inferred g, prior g, and bound p from fresh 8x8 worlds."""
    if worlds < 2 or steps - burn <= 200 or burn < 0:
        raise ValueError("need two worlds and more than 200 post-burn samples")
    streams: dict[str, list[tuple[Tensor, Tensor, Tensor]]] = {k: [] for k in ("g", "g_gen", "p")}
    model.eval()
    with torch.no_grad():
        for w in range(worlds):
            world = GraphWorld(8, sensory=model.config.sensory, seed=seed + w)
            walk = world.random_walk(steps, stay=False, valid_only=True)
            state = model.reset(1)
            acts: dict[str, list[Tensor]] = {k: [] for k in streams}
            for t in range(steps + 1):
                action = torch.tensor([4]) if t == 0 else walk.executed_actions[t - 1 : t]
                out, state = model(walk.observations[t : t + 1], action, state)
                if t >= burn:
                    for key, value in (("g", out.g_inf), ("g_gen", out.g_prior), ("p", out.p_inf)):
                        acts[key].append(value[0].cpu())
            for key in streams:
                streams[key].append(
                    (torch.stack(acts[key]), walk.nodes[burn:], walk.observations[burn:])
                )
    return streams


def table(results: dict[str, dict[str, float]]) -> str:
    """Render the reference table layout."""
    lines = [
        "| Stream | Spatial Info (Null) | Frac > p95 (Perm / Circ) | Split-Half r | "
        "Decoder Acc (Shift Null p95) | Cross-World Map Corr | Cross-World RSA (Null p95) | "
        "RSA vs Grid | RSA vs Sensory |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, r in results.items():
        lines.append(
            f"| {key} | {r['si']:.3f} ({r['perm_null']:.3f}/{r['circ_null']:.3f}) | "
            f"{r['frac_perm']:.2f} / {r['frac_circ']:.2f} | {r['splithalf']:.2f} | "
            f"{r['dec']:.3f} ({r['dec_null']:.3f}) | {r['map_corr']:.3f} | "
            f"{r['rsa']:.3f} ({r['rsa_null_p95']:.3f}) | {r['rsa_grid']:.3f} | "
            f"{r['rsa_sensory']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Probe a saved model or an explicitly labelled random-init control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1000000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.checkpoint:
        model, train_seed, updates = load_checkpoint(args.checkpoint)
    else:
        train_seed, updates = 0, 0
        model = TEM(TEMConfig(widths=(5, 6, 7, 8)), generator=torch.Generator().manual_seed(0))
    streams = extract(model, worlds=args.worlds, steps=args.steps, seed=args.seed)
    results = {key: measure(value) for key, value in streams.items()}
    diagnostics = {
        key: [
            {
                "negative_fraction": float((a < 0).double().mean()),
                "active_units": int((a.double().std(0) > 1e-9).sum()),
                "visited_nodes": int(pos.unique().numel()),
                "retained_samples": len(pos),
            }
            for a, pos, _ in value
        ]
        for key, value in streams.items()
    }
    clean = {
        key: {k: v if torch.isfinite(torch.tensor(v)) else None for k, v in row.items()}
        for key, row in results.items()
    }
    payload = {
        "checkpoint": str(args.checkpoint),
        "training_seed": train_seed,
        "updates": updates,
        "probe_seed": args.seed,
        "worlds": args.worlds,
        "steps": args.steps,
        "burn": 300,
        "gridness": {"resolvable": False, "reason": "8x8 is too small"},
        "results": clean,
        "diagnostics": diagnostics,
        "config": vars(model.config),
        "si_repeats": 32,
        "decoder_rsa_repeats": 100,
        "si_signed_warning": "Signed activations invalidate bits/spike interpretation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    args.out.with_suffix(".md").write_text(table(results))


if __name__ == "__main__":
    main()
