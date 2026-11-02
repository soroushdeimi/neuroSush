"""Spiking Tolman-Eichenbaum Machine (Kawahara & Fujisawa 2025, bioRxiv 10.1101/2025.10.16.682754).

Shows Spiking TEM with LIF layers, imposed 8 Hz theta oscillation, trainable gain,
and STDP fast memory on 2D arenas and 1D tracks.

Deviations from the paper:
    No DG/CA3 inference circuit, no LECII/III, no Algorithm 1 buffering.
    MECV pathway reduced. Theta imposed so precession is not emergent.
    Populations use 64 units (default) or configurable units. STDP memory M is environment-local.
    Parameter values differ between paper(), authors_code() and default config (default uses
    lower threshold=0.5, higher tonic=0.5, and theta_scale=0.1 to maintain healthy spiking
    activity at initialization; paper() uses strict paper Table 1 values; authors_code() uses
    released code parameters).
    Theta scaling and tonic drive are named config fields (theta_scale, tonic).
    STDP memory resets per walk in training although the authors' code never resets.

The script reports what it measures; it makes no claim of reproducing
the paper's numbers or of grid cells emerging.
Note: Theta oscillation is imposed.

Run: ``python examples/spiking_tem_kawahara.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import signal
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import torch

from neurosush.spatial.fields import (
    grid_fraction,
    spatial_information_vs_shuffle,
    summarize_classes,
    track_phase_slopes,
)
from neurosush.spatial.graph_world import GraphWorld, Walk
from neurosush.tem.spiking import SpikingTEM, SpikingTEMConfig, train_walks


def walk_from_nodes(world: GraphWorld, nodes: torch.Tensor) -> Walk:
    """Construct a Walk from a sequence of node IDs on a graph world."""
    actions = torch.where(nodes[1:] > nodes[:-1], torch.tensor(0), torch.tensor(1))
    executed_actions = torch.where(nodes[1:] == nodes[:-1], torch.tensor(2), actions)
    novel = []
    fractions = []
    visited = {int(nodes[0])}
    edges = set()
    for source, target in zip(nodes[:-1].tolist(), nodes[1:].tolist(), strict=True):
        novel.append(target != source and target in visited and (source, target) not in edges)
        fractions.append(len(visited) / world.size)
        visited.add(target)
        edges.add((source, target))
    return Walk(
        nodes=nodes,
        observations=world.labels[nodes],
        actions=actions,
        novel_edge_to_seen=torch.tensor(novel, dtype=torch.bool),
        fraction_seen=torch.tensor(fractions, dtype=torch.float32),
        executed_actions=executed_actions,
    )


def compute_stats(values: list[float]) -> tuple[float, float]:
    """Return mean and sample standard deviation."""
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, var**0.5


def run_experiment(
    arena: str,
    config: SpikingTEMConfig,
    steps: int,
    walk_len: int,
    seeds: list[int],
) -> dict:
    """Run SpikingTEM experiments on 2D arena and/or 1D track across seeds."""
    results: dict = {"2d": [], "1d": []}

    for seed in seeds:
        if arena in ("2d", "both"):
            train_worlds = [GraphWorld(width=8, sensory=20, seed=seed * 100 + i) for i in range(3)]
            train_walks_list = [w.random_walk(walk_len) for w in train_worlds]
            model_2d = SpikingTEM(config)
            train_walks(model_2d, train_walks_list, steps=steps, seed=seed)

            eval_world = GraphWorld(width=8, sensory=20, seed=seed + 9999)
            eval_walk = eval_world.random_walk(walk_len)
            model_2d.eval()
            with torch.no_grad():
                _, rates_2d, _ = model_2d(eval_walk, model_2d.initial_state())
                nodes_2d = eval_walk.nodes[1:]
                gf = grid_fraction(rates_2d["mecii"], nodes_2d, (8, 8))
                si = spatial_information_vs_shuffle(rates_2d["ca1"], nodes_2d, (8, 8), seed=seed)
            results["2d"].append({"seed": seed, "grid_fraction": gf, "spatial_information": si})

        if arena in ("1d", "both"):
            track_world = GraphWorld(width=40, track=True, sensory=20, seed=seed + 555)
            laps = max(1, walk_len // 40)
            shuttle_nodes = track_world.shuttle(laps)
            track_walk = walk_from_nodes(track_world, shuttle_nodes)

            model_1d = SpikingTEM(config)
            train_walks(model_1d, [track_walk], steps=steps, seed=seed)

            model_1d.eval()
            with torch.no_grad():
                _, rates_1d, _ = model_1d(track_walk, model_1d.initial_state())
                mecii_rates = rates_1d["mecii"]
                gen = torch.Generator().manual_seed(seed + 777)
                prob = mecii_rates.clamp(0.0, 1.0)
                spikes = torch.bernoulli(prob, generator=gen)

                num_steps_track = mecii_rates.shape[0]
                t_sec = torch.arange(1, num_steps_track + 1) * config.dt / 1000.0
                phases = (2.0 * math.pi * config.theta_freq * t_sec) % (2.0 * math.pi)
                positions = track_walk.nodes[1:]
                dirs_diff = track_walk.nodes[1:] - track_walk.nodes[:-1]
                dirs = torch.where(dirs_diff > 0, 1, torch.where(dirs_diff < 0, -1, 1))

                total_unit_spikes = spikes.sum(dim=0)
                strongest_unit = int(total_unit_spikes.argmax().item())
                unit_spikes = spikes[:, strongest_unit]

                spike_mask = unit_spikes > 0.5
                spike_positions = positions[spike_mask].float()
                spike_phases = phases[spike_mask]
                spike_directions = dirs[spike_mask]

                field_edges = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 40.0)]
                slopes = track_phase_slopes(
                    spike_positions,
                    spike_phases,
                    spike_directions,
                    [1, -1],
                    field_edges=field_edges,
                )
                summary = summarize_classes(slopes)
            results["1d"].append(
                {"seed": seed, "strongest_unit": strongest_unit, "phase_summary": summary}
            )

    return results


def run(
    arena: str = "both",
    config_choice: str = "default",
    steps: int = 60,
    walk_len: int = 200,
    seeds: list[int] | int = 0,
    no_theta: bool = False,
    no_stdp: bool = False,
    no_gain: bool = False,
    device: str = "cpu",
) -> dict:
    """Pure Spiking TEM evaluation run returning type-stable dict."""
    seed_list = [seeds] if isinstance(seeds, int) else list(seeds)

    if config_choice == "paper":
        base_cfg = SpikingTEMConfig.paper()
    elif config_choice == "authors":
        base_cfg = SpikingTEMConfig.authors_code()
    else:
        base_cfg = SpikingTEMConfig()

    config = dataclasses.replace(
        base_cfg,
        use_theta=not no_theta and base_cfg.use_theta,
        use_stdp=not no_stdp and base_cfg.use_stdp,
        use_gain=not no_gain and base_cfg.use_gain,
    )

    results = run_experiment(arena, config, steps, walk_len, seed_list)
    return {
        "arena": arena,
        "config_type": config_choice,
        "config": asdict(config),
        "seeds": seed_list,
        "2d": results.get("2d", []),
        "1d": results.get("1d", []),
    }


def print_report(arena: str, runs: list[dict]) -> None:
    """Print summary report for arena results."""
    if not runs:
        return
    print(f"\n=== Spiking TEM Arena: {arena.upper()} (seeds={len(runs)}) ===")
    if arena == "2d":
        hex_fracs = [r["grid_fraction"]["hex_fraction"] for r in runs]
        sq_fracs = [r["grid_fraction"]["square_fraction"] for r in runs]
        mean_infos = [r["spatial_information"]["mean_info"] for r in runs]
        above_p95 = [r["spatial_information"]["fraction_above_p95"] for r in runs]

        h_mean, h_sd = compute_stats(hex_fracs)
        s_mean, s_sd = compute_stats(sq_fracs)
        mi_mean, mi_sd = compute_stats(mean_infos)
        ap_mean, ap_sd = compute_stats(above_p95)

        print(f"Hexagonal Grid Fraction : {h_mean:.4f} +- {h_sd:.4f}")
        print(f"Square Grid Fraction    : {s_mean:.4f} +- {s_sd:.4f}")
        print(f"Mean CA1 Info           : {mi_mean:.4f} +- {mi_sd:.4f}")
        print(f"Fraction Above Shuffle p95: {ap_mean:.4f} +- {ap_sd:.4f}")
        label_res = runs[0]["grid_fraction"]["label"]
        resolvable_res = runs[0]["grid_fraction"]["resolvable"]
        print(f"Label / Resolvable      : {label_res} (resolvable={resolvable_res})")
        if "caveat" in runs[0]["grid_fraction"]:
            print(f"Caveat                  : {runs[0]['grid_fraction']['caveat']}")
    elif arena == "1d":
        print("Note: Theta oscillation is imposed.")
        print(f"Evaluated {len(runs)} seed runs on 1D track.")
        for r in runs:
            seed_val = r["seed"]
            unit_val = r["strongest_unit"]
            summary_val = r["phase_summary"]
            print(f"Seed {seed_val}: Strongest unit={unit_val}, summary={summary_val}")


def main() -> None:
    """Parse arguments, run Spiking TEM Kawahara experiments, and report/save results."""
    parser = argparse.ArgumentParser(
        description="Spiking TEM (Kawahara & Fujisawa 2025) Example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arena", choices=["2d", "1d", "both"], default="both", help="Environment arena type"
    )
    parser.add_argument(
        "--config",
        choices=["default", "paper", "authors"],
        default="default",
        help="Preset config (default, paper, or authors)",
    )
    parser.add_argument(
        "--steps", type=int, default=60, help="Training gradient update steps per walk"
    )
    parser.add_argument("--walk-len", type=int, default=200, help="Walk length in steps")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seed integers")
    parser.add_argument(
        "--no-theta", action="store_true", help="Disable imposed 8 Hz theta oscillation"
    )
    parser.add_argument("--no-stdp", action="store_true", help="Disable STDP plasticity")
    parser.add_argument("--no-gain", action="store_true", help="Disable trainable gain")
    parser.add_argument("--out", type=str, default=None, help="Path for JSON output")
    parser.add_argument("--max-seconds", type=int, default=570, help="Max execution seconds limit")
    args = parser.parse_args()

    has_alarm = hasattr(signal, "alarm")
    if has_alarm:
        signal.alarm(args.max_seconds)
    torch.set_num_threads(1)
    start = monotonic()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    try:
        results = run(
            arena=args.arena,
            config_choice=args.config,
            steps=args.steps,
            walk_len=args.walk_len,
            seeds=seeds,
            no_theta=args.no_theta,
            no_stdp=args.no_stdp,
            no_gain=args.no_gain,
        )

        if results["2d"]:
            print_report("2d", results["2d"])
        if results["1d"]:
            print_report("1d", results["1d"])

        total_seconds = monotonic() - start

        payload = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": 1,
            "arena": args.arena,
            "config_type": args.config,
            "config": results["config"],
            "seeds": seeds,
            "steps": args.steps,
            "walk_len": args.walk_len,
            "total_seconds": total_seconds,
            "results": {
                "2d": results["2d"],
                "1d": results["1d"],
            },
        }

        if args.out:
            dest = Path(args.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"\nSaved results to {dest}")

        print(f"\ntotal_seconds={total_seconds:.3f}", flush=True)
    finally:
        if has_alarm:
            signal.alarm(0)


if __name__ == "__main__":
    main()
