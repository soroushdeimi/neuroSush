"""Structured torus V3 runs with optional V1 parity and final-only evaluation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from neurosush.tem import TEMConfig, evaluate, train


def finite(value: Any) -> Any:
    """Replace undefined metrics with JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite(item) for item in value]
    return value


def write(path: Path, payload: dict[str, Any]) -> None:
    """Atomically preserve the last complete progress record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(finite(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    """Train for exactly the requested optimizer updates and score the final model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", choices=("v0", "v1"), default="v1")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    torch.set_num_threads(1)
    parity = args.base == "v1"
    config = TEMConfig(
        widths=(5, 6, 7, 8),
        inference_memory_loss=parity,
        calibrated_reductions=parity,
        bounded_attractor_activation=parity,
        transition="rotation",
    )
    payload: dict[str, Any] = {
        "variant": "v3",
        "base": args.base,
        "seed": args.seed,
        "requested_updates": args.updates,
        "config": asdict(config),
        "status": "running",
        "history": [],
        "evaluation_seeds": [1234, 1235, 1236],
    }
    write(args.out, payload)

    def progress(history: list[dict[str, float]]) -> None:
        if len(history) % 500 == 0:
            payload.update(history=history, completed_updates=len(history))
            write(args.out, payload)

    result = train(
        config,
        steps=args.updates * config.bptt,
        seed=args.seed,
        log_every=100,
        on_update=progress,
        log_grads=True,
        checkpoint_path=str(
            Path("experiments/results/checkpoints") / f"{'v3' + args.base}-s{args.seed}.pt"
        ),
        checkpoint_every=args.checkpoint_every,
    )
    payload.update(
        history=result.history,
        completed_updates=len(result.history),
        training_seconds=result.seconds,
        updates_per_second=len(result.history) / result.seconds,
    )
    write(args.out, payload)
    evaluations = [
        evaluate(result.model, config, environments=6, steps=600, seed=seed)
        for seed in (1234, 1235, 1236)
    ]
    norms = torch.tensor([row["grad_norm"] for row in result.history], dtype=torch.float64)
    payload.update(
        status="complete",
        evaluations=evaluations,
        max_grad=float(norms.max()),
        p95_grad=float(norms.quantile(0.95)),
        evaluation_checkpoint="final",
    )
    write(args.out, payload)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "variant",
                    "completed_updates",
                    "updates_per_second",
                    "max_grad",
                    "p95_grad",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
