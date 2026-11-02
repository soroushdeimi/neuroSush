"""Sequential, bounded TEM C3 experiments with persistent unfiltered results."""

from __future__ import annotations

import argparse
import json
import math
import platform
import signal
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from typing import Any

import torch

from neurosush.tem import TEMConfig, evaluate, train
from neurosush.tem.train import TrainResult

DESTINATION = Path(__file__).parent / "results" / "tem_c3.json"


def finite(value: Any) -> Any:
    """Represent empty evaluation bins as JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    return value


def summary(result: TrainResult) -> dict[str, Any]:
    """Summarize clipping, saturation, and gradient sources without dropping history."""
    history = result.history
    tail = history[-min(100, len(history)) :]
    groups = {}
    for name in history[-1]:
        if name.startswith("grad/"):
            group = name.split("/", 1)[1].split(".", 1)[0]
            groups[group] = groups.get(group, 0.0) + history[-1][name] ** 2
    return {
        "last": history[-1],
        "tail_mean": {key: sum(row[key] for row in tail) / len(tail) for key in tail[0]},
        "gradient_norm_max": max(row["grad_norm"] for row in history),
        "gradient_norm_p95": float(
            torch.tensor([row["grad_norm"] for row in history]).quantile(0.95)
        ),
        "clipped_fraction": sum(row["grad_norm"] > result.config.clip for row in history)
        / len(history),
        "last_gradient_groups": {name: value**0.5 for name, value in groups.items()},
        "training_seconds": result.seconds,
    }


def run(config: TEMConfig, *, seed: int, steps: int, oracle: bool) -> dict[str, Any]:
    """Fit once and measure in four fresh worlds with disjoint seeds."""
    result = train(
        config,
        steps=steps,
        seed=seed,
        oracle=oracle,
        diagnostics=True,
        log_every=0 if oracle else 100,
    )
    scores = evaluate(
        result.model, config, environments=4, steps=400, seed=1234 + seed, oracle=oracle
    )
    measured = {
        "seed": seed,
        "evaluation_seed": 1234 + seed,
        "config": asdict(config),
        "steps": steps,
        "oracle": oracle,
        "evaluation": scores,
        "diagnostics": summary(result),
        "history": result.history,
    }
    print(
        json.dumps(finite({key: value for key, value in measured.items() if key != "history"})),
        flush=True,
    )
    return measured


def main() -> None:
    """Run one numbered experiment; each has a 295-second wall-clock guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("oracle", "baseline"))
    args = parser.parse_args()
    signal.alarm(295)
    torch.set_num_threads(1)
    experiment_id = {"oracle": 1, "baseline": 2}[args.stage]
    payload = (
        json.loads(DESTINATION.read_text())
        if DESTINATION.exists()
        else {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": 1,
            "experiments": [],
        }
    )
    if any(e["id"] == experiment_id for e in payload["experiments"]):
        raise ValueError(f"experiment id={experiment_id} already recorded in {DESTINATION}")
    start = monotonic()
    if args.stage == "oracle":
        config = TEMConfig(
            sensory=20,
            compressed=10,
            modules=(36,),
            hidden=20,
            batch=1,
            bptt=10,
            walk_len=400,
            widths=(4,),
            lr_max=0.03,
            lr_min=0.003,
            decay_ramp=1,
            rate_ramp=1,
            latent_ramp=1,
        )
        changed = "C2 oracle settings unchanged; three training and held-out environment seeds."
        runs = [run(config, seed=seed, steps=400, oracle=True) for seed in (0, 1, 2)]
    else:
        config = TEMConfig(
            batch=8, widths=(4,), decay_ramp=10, rate_ramp=80, latent_ramp=80, lr_every=400
        )
        changed = (
            "C2 learned settings unchanged; read-only pre-clip gradients and saturation added."
        )
        runs = [run(config, seed=0, steps=12000, oracle=False)]
    payload["experiments"].append(
        {"id": experiment_id, "what_changed": changed, "runs": runs, "seconds": monotonic() - start}
    )
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(finite(payload), indent=2, allow_nan=False) + "\n")
    print(f"recorded experiment={experiment_id} seconds={monotonic() - start:.3f}", flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
