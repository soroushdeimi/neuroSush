"""Frozen TEM spatial measurements matching the reference analysis definitions."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from time import monotonic

import torch
from torch import Tensor

from neurosush.spatial import GraphWorld
from neurosush.spatial.rate_maps import rate_maps
from neurosush.tem import TEM, TEMConfig
from neurosush.tem.checkpoint import load_checkpoint
from neurosush.tem.model import MLP, TEMState


def maps(a: Tensor, pos: Tensor, size: int = 8) -> tuple[Tensor, Tensor]:
    """Return node-first size x size means and occupancy."""
    m, c = rate_maps(a, pos, (size, size))
    return m.reshape(a.shape[1], size * size).T, c.flatten()


def information(a: Tensor, pos: Tensor, size: int = 8) -> Tensor:
    """Match reference SI, including its signed-activation convention.

    Warning: Signed activations invalidate bits/spike interpretation.
    """
    m, c = maps(a, pos, size)
    p = c[c > 0].double() / c.sum()
    r = m[c > 0]
    ratio = r / (r * p[:, None]).sum(0).clamp_min(1e-12)
    return (p[:, None] * torch.where(ratio > 0, ratio * ratio.clamp_min(1e-300).log2(), 0)).sum(0)


def eta2(a: Tensor, pos: Tensor, size: int = 8) -> Tensor:
    """Share of activity variance explained by position (eta-squared), in [0, 1]."""
    a = a.detach().double()
    pos = pos.detach().long()
    num_nodes = size * size
    ok = torch.isfinite(a).all(dim=-1) & (pos >= 0) & (pos < num_nodes)
    a = a[ok]
    pos = pos[ok]
    if len(a) == 0:
        return torch.full((a.shape[1],), float("nan"))
    mean_all = a.mean(dim=0)
    tss = (a - mean_all).square().sum(dim=0)
    bss = torch.zeros_like(tss)
    for q in range(num_nodes):
        mask = pos == q
        n_q = mask.sum()
        if n_q > 0:
            mean_q = a[mask].mean(dim=0)
            bss += n_q.double() * (mean_q - mean_all).square()
    return torch.where(tss > 1e-12, bss / tss, torch.zeros_like(tss)).clamp(0.0, 1.0)


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


def decoder(
    a: Tensor, pos: Tensor, rng: torch.Generator, repeats: int, size: int = 8
) -> tuple[float, float]:
    """Five contiguous test folds, 25-sample exclusion gap, fixed-prediction null."""
    edges = torch.linspace(0, len(pos), 6).long()
    pred = torch.full_like(pos, -1)
    for start, end in zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True):
        train_mask = torch.ones(len(pos), dtype=torch.bool)
        train_mask[max(0, start - 25) : min(len(pos), end + 25)] = False
        template, counts = maps(a[train_mask], pos[train_mask], size)
        distance = (a[start:end, None] - template[None]).square().nansum(-1)
        if (counts == 0).any():
            distance[:, counts == 0] = float("inf")
        pred[start:end] = distance.argmin(-1)
    null = torch.tensor(
        [float((pred == shifted(pos, rng)).double().mean()) for _ in range(repeats)],
        dtype=torch.float64,
    )
    return float((pred == pos).double().mean()), float(null.quantile(0.95))


def compute_disp_r2(template: Tensor, counts: Tensor, size: int) -> float:
    """Share of within-world distance variance explained by displacement (i-j)."""
    valid = counts > 0
    indices = valid.nonzero().flatten().tolist()
    if len(indices) < 3:
        return float("nan")
    dists = []
    disps = []
    for i_idx, i in enumerate(indices):
        ri, ci = i // size, i % size
        for j in indices[i_idx + 1 :]:
            rj, cj = j // size, j % size
            d = (template[i] - template[j]).square().sum().sqrt().item()
            dr, dc = ri - rj, ci - cj
            dists.append(d)
            disps.append((dr, dc))
    if not dists:
        return float("nan")
    dists_t = torch.tensor(dists, dtype=torch.float64)
    mean_d = dists_t.mean()
    sstot = (dists_t - mean_d).square().sum()
    if sstot < 1e-12:
        return 0.0
    disp_groups: dict[tuple[int, int], list[float]] = {}
    for d, disp in zip(dists, disps, strict=True):
        disp_groups.setdefault(disp, []).append(d)
    pred_dists = torch.tensor(
        [torch.tensor(disp_groups[disp], dtype=torch.float64).mean() for disp in disps],
        dtype=torch.float64,
    )
    ssreg = (pred_dists - mean_d).square().sum()
    r2 = ssreg / sstot
    return float(r2.clamp(0.0, 1.0))


def compute_abs_map_corr(left: Tensor, lc: Tensor, right: Tensor, rc: Tensor) -> float:
    """Per-unit map correlation ignoring sign."""
    ok = (lc > 0) & (rc > 0)
    uc = [
        abs(corr(left[ok, u], right[ok, u]))
        for u in range(left.shape[1])
        if left[ok, u].std() > 1e-6 and right[ok, u].std() > 1e-6
    ]
    return mean(uc) if uc else float("nan")


def compute_best_shift_map_corr(
    left: Tensor, lc: Tensor, right: Tensor, rc: Tensor, size: int
) -> float:
    """Max over integer 2D shifts of world B's map (overlap >= 25 nodes)."""
    ma_all = left.T.reshape(-1, size, size)
    mb_all = right.T.reshape(-1, size, size)
    ca = lc.reshape(size, size)
    cb = rc.reshape(size, size)

    unit_corrs = []
    for u in range(left.shape[1]):
        ma = ma_all[u]
        mb = mb_all[u]
        best_c = float("-inf")
        found = False
        for dr in range(-size + 1, size):
            for dc in range(-size + 1, size):
                r_start_a = max(0, dr)
                r_end_a = min(size, size + dr)
                r_start_b = max(0, -dr)
                r_end_b = min(size, size - dr)

                c_start_a = max(0, dc)
                c_end_a = min(size, size + dc)
                c_start_b = max(0, -dc)
                c_end_b = min(size, size - dc)

                sub_a = ma[r_start_a:r_end_a, c_start_a:c_end_a]
                sub_b = mb[r_start_b:r_end_b, c_start_b:c_end_b]
                sub_ca = ca[r_start_a:r_end_a, c_start_a:c_end_a]
                sub_cb = cb[r_start_b:r_end_b, c_start_b:c_end_b]

                overlap = (
                    (sub_ca > 0) & (sub_cb > 0) & torch.isfinite(sub_a) & torch.isfinite(sub_b)
                )
                if overlap.sum() >= 25:
                    va = sub_a[overlap].double()
                    vb = sub_b[overlap].double()
                    if va.std() > 1e-6 and vb.std() > 1e-6:
                        c = float(torch.corrcoef(torch.stack((va, vb)))[0, 1])
                        if torch.isfinite(torch.tensor(c)) and c > best_c:
                            best_c = c
                            found = True
        if found:
            unit_corrs.append(best_c)
    return (
        float(torch.tensor(unit_corrs, dtype=torch.float64).mean()) if unit_corrs else float("nan")
    )


def measure(
    worlds: list[tuple[Tensor, Tensor, Tensor]],
    *,
    seed: int = 42,
    si_repeats: int = 32,
    repeats: int = 100,
    size: int = 8,
    stream: str = "g",
    shuffle_labels: bool = False,
    return_rsa_pairs: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[float]]:
    """Average within-world scores and pairwise cross-world geometry."""
    if len(worlds) < 2 or min(si_repeats, repeats) < 1:
        raise ValueError("need at least two worlds and positive null repeats")
    rng = torch.Generator().manual_seed(seed)
    num_nodes = size * size
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
            "abs_map_corr",
            "best_shift_map_corr",
            "disp_r2",
            "rsa",
            "rsa_null_p95",
            "rsa_grid",
            "rsa_sensory",
        )
    }
    templates = []
    iu = torch.triu_indices(num_nodes, num_nodes, 1)
    nodes = torch.arange(num_nodes)
    grid = (
        (nodes[:, None] // size - nodes[None] // size).abs()
        + (nodes[:, None] % size - nodes[None] % size).abs()
    )[iu[0], iu[1]]

    def dm(t: Tensor, counts: Tensor) -> Tensor:
        valid = counts > 0
        diff = t[:, None, :] - t[None, :, :]
        dist_sq = diff.square().nansum(-1).sqrt()
        invalid = (~valid[:, None]) | (~valid[None, :])
        dist_sq[invalid] = float("nan")
        return dist_sq[iu[0], iu[1]]

    for a, pos, sensory in worlds:
        a = a.double()
        if shuffle_labels:
            pos = pos[torch.randperm(len(pos), generator=rng)]
        active = a.std(0) > 1e-9

        def score_fn(act: Tensor, p: Tensor) -> Tensor:
            if stream == "p":
                return information(act, p, size)
            return eta2(act, p, size)

        si = score_fn(a, pos)
        perm = torch.stack(
            [score_fn(a, pos[torch.randperm(len(pos), generator=rng)]) for _ in range(si_repeats)]
        )
        circ = torch.stack([score_fn(a, shifted(pos, rng)) for _ in range(si_repeats)])
        for key, value in (
            ("si", si[active].mean()),
            ("perm_null", perm[:, active].mean()),
            ("circ_null", circ[:, active].mean()),
            (
                "frac_perm",
                (si > perm.quantile(0.95, dim=0))[active].double().mean(),
            ),
            (
                "frac_circ",
                (si > circ.quantile(0.95, dim=0))[active].double().mean(),
            ),
        ):
            rows[key].append(float(value))
        h = len(pos) // 2
        m1, c1 = maps(a[:h], pos[:h], size)
        m2, c2 = maps(a[h:], pos[h:], size)
        ok = (c1 > 0) & (c2 > 0)
        rows["splithalf"].append(
            mean([corr(m1[ok, u], m2[ok, u]) for u in active.nonzero().flatten().tolist()])
        )
        acc, null = decoder(a, pos, rng, repeats, size)
        rows["dec"].append(acc)
        rows["dec_null"].append(null)
        template, counts = maps(a, pos, size)
        templates.append((template, counts))
        rows["disp_r2"].append(compute_disp_r2(template, counts, size))
        sx = torch.tensor(
            [
                int(sensory[pos == q].bincount().argmax()) if (pos == q).any() else 0
                for q in range(num_nodes)
            ]
        )
        rows["rsa_grid"].append(corr(dm(template, counts), grid))
        rows["rsa_sensory"].append(
            corr(dm(template, counts), (sx[:, None] != sx[None])[iu[0], iu[1]])
        )

    null_rsa = []
    rsa_pairs = []
    for i, (left, lc) in enumerate(templates):
        for right, rc in templates[i + 1 :]:
            ok = (lc > 0) & (rc > 0)
            uc = [
                corr(left[ok, u], right[ok, u])
                for u in range(left.shape[1])
                if left[ok, u].std() > 1e-6 and right[ok, u].std() > 1e-6
            ]
            rows["map_corr"].append(mean(uc))
            rows["abs_map_corr"].append(compute_abs_map_corr(left, lc, right, rc))
            rows["best_shift_map_corr"].append(
                compute_best_shift_map_corr(left, lc, right, rc, size)
            )
            val_rsa = corr(dm(left, lc), dm(right, rc))
            rows["rsa"].append(val_rsa)
            rsa_pairs.append(val_rsa)
            null_rsa.extend(
                [
                    corr(
                        dm(left, lc),
                        dm(
                            right[torch.randperm(num_nodes, generator=rng)],
                            rc,
                        ),
                    )
                    for _ in range(repeats)
                ]
            )
    finite_null = torch.tensor(
        [v for v in null_rsa if torch.isfinite(torch.tensor(v))],
        dtype=torch.float64,
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
        "disp_r2",
    }
    metrics = {
        key: float(torch.tensor(values, dtype=torch.float64).mean())
        if key in within
        else mean(values)
        for key, values in rows.items()
    }
    if return_rsa_pairs:
        return metrics, rsa_pairs
    return metrics


def extract(
    model: TEM,
    *,
    worlds: int,
    steps: int,
    seed: int,
    burn: int = 300,
    size: int = 8,
    memory_enabled: bool = True,
    random_init: bool = False,
    probe_generator: torch.Generator | None = None,
) -> dict[str, list[tuple[Tensor, Tensor, Tensor]]]:
    """Extract aligned inferred g, prior g, and bound p from fresh worlds."""
    if worlds < 2 or steps - burn <= 200 or burn < 0:
        raise ValueError("need two worlds and more than 200 post-burn samples")
    streams: dict[str, list[tuple[Tensor, Tensor, Tensor]]] = {k: [] for k in ("g", "g_gen", "p")}
    model.eval()
    with torch.no_grad():
        for w in range(worlds):
            world = GraphWorld(size, sensory=model.config.sensory, seed=seed + w)
            walk = world.random_walk(steps, stay=False, valid_only=True)
            state = model.reset(1)
            if random_init and probe_generator is not None:
                new_gs = tuple(
                    torch.randn(1, n, generator=probe_generator) * model.config.initial_std
                    for n in model.config.modules
                )
                state = TEMState(new_gs, state.x, state.gen, state.inf)
            acts: dict[str, list[Tensor]] = {k: [] for k in streams}
            for t in range(steps + 1):
                action = torch.tensor([4]) if t == 0 else walk.executed_actions[t - 1 : t]
                rate_val = 0.5 if memory_enabled else 0.0
                decay_val = 0.9999 if memory_enabled else 1.0
                out, state = model(
                    walk.observations[t : t + 1],
                    action,
                    state,
                    rate=rate_val,
                    decay=decay_val,
                )
                if t >= burn:
                    for key, value in (
                        ("g", out.g_inf),
                        ("g_gen", out.g_prior),
                        ("p", out.p_inf),
                    ):
                        acts[key].append(value[0].cpu())
            for key in streams:
                streams[key].append(
                    (
                        torch.stack(acts[key]),
                        walk.nodes[burn:],
                        walk.observations[burn:],
                    )
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
    parser.add_argument("--probe-seeds", type=str, default="1000000,2000000,3000000")
    parser.add_argument("--size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)

    probe_seeds = [int(s.strip()) for s in args.probe_seeds.split(",")]
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    if args.checkpoint:
        base_model, train_seed, updates = load_checkpoint(args.checkpoint)
    else:
        train_seed, updates = 0, 0
        base_model = TEM(
            TEMConfig(widths=(5, 6, 7, 8) if args.size <= 8 else (args.size,)),
            generator=torch.Generator().manual_seed(0),
        )

    start_time = monotonic()

    def run_probe_for_stream_or_control(
        model: TEM,
        stream_name: str,
        memory_enabled: bool,
        random_init: bool,
        shuffle_labels: bool = False,
        custom_transitions_scale: float | None = None,
    ) -> tuple[
        dict[str, float],
        dict[str, dict[str, float]],
        list[float],
        list[dict],
    ]:
        seed_results = {}
        seed_rsa_pairs = {}
        seed_diagnostics = []
        for seed in probe_seeds:
            probe_rng = torch.Generator().manual_seed(seed)
            eval_model = model
            if custom_transitions_scale is not None:
                eval_model = TEM(
                    model.config,
                    generator=torch.Generator().manual_seed(seed),
                )
                eval_model.load_state_dict(model.state_dict())
                for transition in eval_model.transitions:
                    if isinstance(transition, MLP):
                        inp = transition.first.weight.shape[1]
                        hid = transition.first.weight.shape[0]
                        s_in = custom_transitions_scale / math.sqrt(inp)
                        s_hi = custom_transitions_scale / math.sqrt(hid)
                        with torch.no_grad():
                            transition.first.weight.uniform_(-s_in, s_in, generator=probe_rng)
                            transition.last.weight.uniform_(-s_hi, s_hi, generator=probe_rng)

            streams = extract(
                eval_model,
                worlds=args.worlds,
                steps=args.steps,
                seed=seed,
                size=args.size,
                memory_enabled=memory_enabled,
                random_init=random_init,
                probe_generator=probe_rng,
            )
            target_stream_data = streams[stream_name]
            metrics, rsa_pairs = measure(
                target_stream_data,
                seed=seed,
                size=args.size,
                stream=stream_name,
                shuffle_labels=shuffle_labels,
                return_rsa_pairs=True,
            )
            seed_results[str(seed)] = metrics
            seed_rsa_pairs[str(seed)] = rsa_pairs

            diag = [
                {
                    "negative_fraction": float((a < 0).double().mean()),
                    "active_units": int((a.double().std(0) > 1e-9).sum()),
                    "visited_nodes": int(pos.unique().numel()),
                    "retained_samples": len(pos),
                }
                for a, pos, _ in target_stream_data
            ]
            seed_diagnostics.append(diag)

        avg_res = {}
        first_metrics = next(iter(seed_results.values()))
        for k in first_metrics:
            vals = [
                seed_results[str(s)][k]
                for s in probe_seeds
                if k in seed_results[str(s)] and not math.isnan(seed_results[str(s)][k])
            ]
            avg_res[k] = (
                float(torch.tensor(vals, dtype=torch.float64).mean()) if vals else float("nan")
            )

        all_rsa_pairs = [p for pairs in seed_rsa_pairs.values() for p in pairs]
        return avg_res, seed_results, all_rsa_pairs, seed_diagnostics[0]

    print("Running probe and controls across probe seeds...")
    results_dict = {}
    per_seed_dict = {}
    rsa_pairs_dict = {}
    diagnostics_dict = {}

    for s_name in ("g", "g_gen", "p"):
        avg, per_seed, rsa_pairs, diag = run_probe_for_stream_or_control(
            base_model, s_name, memory_enabled=True, random_init=False
        )
        results_dict[s_name] = avg
        per_seed_dict[s_name] = per_seed
        rsa_pairs_dict[s_name] = rsa_pairs
        diagnostics_dict[s_name] = diag

    pi_shared_model = TEM(base_model.config, generator=torch.Generator().manual_seed(0))
    pi_shared_model.load_state_dict(base_model.state_dict())

    control_specs = [
        ("pi_shared", pi_shared_model, False, False, False, None),
        ("pi_random_init", pi_shared_model, False, True, False, None),
        ("pi_random_transitions_0.1", pi_shared_model, False, False, False, 0.1),
        ("pi_random_transitions_0.3", pi_shared_model, False, False, False, 0.3),
        ("pi_random_transitions_1.0", pi_shared_model, False, False, False, 1.0),
        ("full_random_init", base_model, True, True, False, None),
        ("label_shuffle", base_model, True, False, True, None),
    ]

    untrained_model = TEM(base_model.config, generator=torch.Generator().manual_seed(42))
    for transition in untrained_model.transitions:
        if isinstance(transition, MLP):
            inp = transition.first.weight.shape[1]
            hid = transition.first.weight.shape[0]
            s_in = 1 / math.sqrt(inp)
            s_hi = 1 / math.sqrt(hid)
            rng_u = torch.Generator().manual_seed(42)
            with torch.no_grad():
                transition.first.weight.uniform_(-s_in, s_in, generator=rng_u)
                transition.last.weight.uniform_(-s_hi, s_hi, generator=rng_u)
    control_specs.append(("memory_on_random", untrained_model, True, False, False, None))

    pi_rsa_all = []
    for c_name, c_model, mem_en, rnd_init, shuf, scale in control_specs:
        avg, per_seed, rsa_pairs, diag = run_probe_for_stream_or_control(
            c_model,
            "g",
            memory_enabled=mem_en,
            random_init=rnd_init,
            shuffle_labels=shuf,
            custom_transitions_scale=scale,
        )
        results_dict[c_name] = avg
        per_seed_dict[c_name] = per_seed
        rsa_pairs_dict[c_name] = rsa_pairs
        diagnostics_dict[c_name] = diag

        if c_name in (
            "pi_shared",
            "pi_random_init",
            "pi_random_transitions_0.1",
            "pi_random_transitions_0.3",
            "pi_random_transitions_1.0",
        ):
            pi_rsa_all.extend(rsa_pairs)

    pi_rsa_tensor = torch.tensor([v for v in pi_rsa_all if not math.isnan(v)], dtype=torch.float64)
    rsa_null_pi_p95 = float(pi_rsa_tensor.quantile(0.95)) if len(pi_rsa_tensor) else float("nan")
    rsa_margin = float(results_dict["g"]["rsa"] - rsa_null_pi_p95)

    elapsed = monotonic() - start_time
    print(f"Probe completed in {elapsed:.2f} seconds ({elapsed / 60:.2f} min)")

    clean = {
        key: {k: v if torch.isfinite(torch.tensor(v)) else None for k, v in row.items()}
        for key, row in results_dict.items()
    }
    clean_per_seed = {
        key: {
            seed_key: {
                k: v if torch.isfinite(torch.tensor(v)) else None for k, v in metrics.items()
            }
            for seed_key, metrics in seed_dict.items()
        }
        for key, seed_dict in per_seed_dict.items()
    }
    clean_rsa_pairs = {
        key: [v if not math.isnan(v) else None for v in pairs]
        for key, pairs in rsa_pairs_dict.items()
    }

    payload = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "git_sha": git_sha,
        "training_seed": train_seed,
        "updates": updates,
        "probe_seeds": probe_seeds,
        "worlds": args.worlds,
        "steps": args.steps,
        "burn": 300,
        "rsa_margin": rsa_margin,
        "rsa_null_pi_p95": rsa_null_pi_p95,
        "gridness": {
            "resolvable": args.size >= 16,
            "hex_gridness": None,
            "square_gridness": None,
            "reason": "Resolvable at 16x16" if args.size >= 16 else "8x8 is too small",
        },
        "probe_seconds": elapsed,
        "results": clean,
        "per_seed": clean_per_seed,
        "rsa_pairs": clean_rsa_pairs,
        "diagnostics": diagnostics_dict,
        "config": vars(base_model.config),
        "si_repeats": 32,
        "decoder_rsa_repeats": 100,
        "si_signed_warning": "Signed activations invalidate bits/spike interpretation",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    args.out.with_suffix(".md").write_text(
        table(
            {
                "g": results_dict["g"],
                "g_gen": results_dict["g_gen"],
                "p": results_dict["p"],
            }
        )
    )


if __name__ == "__main__":
    main()
