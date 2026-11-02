"""Tolman-Eichenbaum Machine (Whittington et al. 2020, Cell 183:1249) at laptop scale.

Runs rate-coded TEM training and evaluation (oracle, learned, or both) across seeds
and prints summary evaluation metrics.

Deviations from the paper:
    Modules have independent memories without cross-module hierarchy. All g units
    participate in binding. The decoder is linear and shared across modules.
    Defaults reduce population sizes and sensory labels for laptop experiments.

Results of the bundled run are reported by the script; no claim of
reproducing the paper's numbers is made.

Run: ``python examples/tem_whittington.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
import signal
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import torch

from neurosush.tem import TEMConfig, evaluate, train


def compute_stats(values: list[float]) -> tuple[float, float]:
    """Return mean and sample standard deviation."""
    n = len(values)
    mean = sum(values) / n
    if n <= 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, var**0.5


def run_experiment(
    mode: str,
    steps: int,
    batch: int | None,
    modules: tuple[int, ...] | None,
    seeds: list[int],
    widths: tuple[int, ...],
    walk_len: int = 400,
    device: str = "cpu",
) -> dict:
    """Run oracle and/or learned TEM across specified seeds and return results."""
    results: dict = {"oracle": [], "learned": []}

    for seed in seeds:
        if mode in ("oracle", "both"):
            o_modules = modules if modules is not None else (36,)
            o_batch = batch if batch is not None else 1
            oracle_config = TEMConfig(
                sensory=20,
                compressed=10,
                modules=o_modules,
                hidden=20,
                batch=o_batch,
                bptt=10,
                walk_len=400,
                widths=widths,
                lr_max=0.03,
                lr_min=0.003,
                decay_ramp=1,
                rate_ramp=1,
                latent_ramp=1,
            )
            oracle_res = train(oracle_config, steps=400, seed=seed, oracle=True, device=device)
            oracle_scores = evaluate(
                oracle_res.model,
                oracle_config,
                environments=4,
                steps=400,
                seed=seed + 1234,
                oracle=True,
            )
            results["oracle"].append(
                {
                    "seed": seed,
                    "config": asdict(oracle_config),
                    "training_seconds": oracle_res.seconds,
                    "evaluation": oracle_scores,
                }
            )

        if mode in ("learned", "both"):
            l_modules = modules if modules is not None else (12, 12, 8)
            l_batch = batch if batch is not None else 8
            learned_config = TEMConfig(
                modules=l_modules,
                batch=l_batch,
                widths=widths,
                decay_ramp=10,
                rate_ramp=80,
                latent_ramp=80,
                lr_every=400,
                walk_len=walk_len,
            )
            learned_res = train(learned_config, steps=steps, seed=seed, log_every=0, device=device)
            learned_scores = evaluate(
                learned_res.model,
                learned_config,
                environments=4,
                steps=400,
                seed=seed + 1234,
                oracle=False,
            )
            results["learned"].append(
                {
                    "seed": seed,
                    "config": asdict(learned_config),
                    "training_seconds": learned_res.seconds,
                    "evaluation": learned_scores,
                    "history": learned_res.history,
                }
            )

    return results


def run(
    mode: str = "both",
    steps: int = 12000,
    batch: int | None = None,
    modules: tuple[int, ...] | None = None,
    seeds: list[int] | int = 0,
    widths: tuple[int, ...] | None = None,
    walk_len: int = 400,
    device: str = "cpu",
) -> dict:
    """Pure evaluation run without I/O or side effects, returning type-stable dict."""
    seed_list = [seeds] if isinstance(seeds, int) else list(seeds)
    w = widths if widths is not None else (4,)
    results = run_experiment(
        mode,
        steps,
        batch,
        modules,
        seed_list,
        w,
        walk_len=walk_len,
        device=device,
    )
    return {
        "mode": mode,
        "seeds": seed_list,
        "widths": w,
        "oracle": results["oracle"],
        "learned": results["learned"],
    }


def print_table(title: str, runs: list[dict]) -> None:
    """Print a compact table of evaluation metrics mean and standard deviation."""
    if not runs:
        return
    print(f"\n=== {title} ===")
    metrics = [
        "accuracy",
        "sensory_only",
        "frequency",
        "conditional",
        "node_conditional",
        "position_r2",
        "shift_null_p95",
        "events",
    ]
    print(f"{'Metric':<18} | {'Mean':<10} | {'Std':<10}")
    print("-" * 44)
    for m in metrics:
        vals = [r["evaluation"][m] for r in runs]
        mean, sd = compute_stats(vals)
        print(f"{m:<18} | {mean:<10.4f} | {sd:<10.4f}")


def main() -> None:
    """Parse arguments, run TEM benchmark, and print/save results."""
    parser = argparse.ArgumentParser(
        description="Tolman-Eichenbaum Machine example script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["oracle", "learned", "both"],
        default="both",
        help="Evaluation mode (oracle training fixed at 400 steps)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12000,
        help="Training gradient update steps for learned model",
    )
    parser.add_argument("--batch", type=int, default=None, help="Batch size override")
    parser.add_argument("--modules", type=str, default=None, help="Comma-separated module sizes")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seed integers")
    parser.add_argument("--widths", type=str, default="4", help="Comma-separated world widths")
    parser.add_argument("--out", type=str, default=None, help="Path for JSON output")
    parser.add_argument("--max-seconds", type=int, default=570, help="Max execution seconds limit")
    args = parser.parse_args()

    has_alarm = hasattr(signal, "alarm")
    if has_alarm:
        signal.alarm(args.max_seconds)
    torch.set_num_threads(1)
    start = monotonic()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    widths = tuple(int(w.strip()) for w in args.widths.split(","))
    modules = (
        tuple(int(m.strip()) for m in args.modules.split(",")) if args.modules is not None else None
    )

    try:
        results = run(
            mode=args.mode,
            steps=args.steps,
            batch=args.batch,
            modules=modules,
            seeds=seeds,
            widths=widths,
        )

        if results["oracle"]:
            print_table(f"Oracle Evaluation (seeds={seeds})", results["oracle"])
        if results["learned"]:
            print_table(f"Learned Evaluation (seeds={seeds})", results["learned"])

        total_seconds = monotonic() - start
        payload = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": 1,
            "mode": args.mode,
            "seeds": seeds,
            "widths": widths,
            "total_seconds": total_seconds,
            "results": results,
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
