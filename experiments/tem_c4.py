"""Bounded C4 inference-gradient experiments; results include all module norms."""

from __future__ import annotations

import argparse
import json
import math
import platform
import signal
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import torch
from tem_c3 import finite, summary

from neurosush.tem import TEMConfig, evaluate, train

DESTINATION = Path(__file__).parent / "results" / "tem_c4.json"


def module_norms(history):
    """Aggregate Euclidean pre-clip gradient norms per stream and parameter family."""
    series = {}
    for row in history:
        squares = {}
        for key, value in row.items():
            if not key.startswith("grad/"):
                continue
            parts = key.split("/", 1)[1].split(".")
            module = ".".join(parts[:2]) if len(parts) > 1 and parts[1].isdigit() else parts[0]
            for group in {module, parts[0]}:
                squares[group] = squares.get(group, 0.0) + value**2
        for group, value in squares.items():
            series.setdefault(group, []).append(math.sqrt(value))
    return {
        group: {
            "max": max(values),
            "p95": float(torch.tensor(values, dtype=torch.float64).quantile(0.95)),
        }
        for group, values in series.items()
    }


def run(config, *, seed, oracle, steps=None):
    """Fit once; score six fresh worlds per evaluation seed for 600 transitions."""
    steps = steps if steps is not None else (400 if oracle else 12000)
    result = train(
        config,
        steps=steps,
        seed=seed,
        oracle=oracle,
        diagnostics=True,
        log_every=0 if oracle else 100,
    )
    evaluations = [
        evaluate(result.model, config, environments=6, steps=600, seed=1234 + offset, oracle=oracle)
        for offset in range(3)
    ]
    scores = {
        key: sum(e[key] for e in evaluations) / len(evaluations)
        for key in evaluations[0]
        if key not in {"by_fraction", "events"}
    }
    scores["events"] = sum(e["events"] for e in evaluations)
    diagnostics = summary(result)
    diagnostics["gradient_modules"] = module_norms(result.history)
    record = {
        "seed": seed,
        "config": asdict(config),
        "oracle": oracle,
        "steps": steps,
        "evaluations": evaluations,
        "evaluation_seeds": [1234, 1235, 1236],
        "evaluation_mean": scores,
        "diagnostics": diagnostics,
        "history": result.history,
    }
    print(
        json.dumps(
            finite({k: v for k, v in record.items() if k not in {"history", "evaluations"}})
        ),
        flush=True,
    )
    return record


def main():
    """Run the expanded oracle; retired C4 trials are archived in tem_c4.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("oracle",))
    parser.parse_args()
    torch.set_num_threads(1)
    signal.alarm(295)
    if DESTINATION.exists():
        raise ValueError(f"C4 results already exist at {DESTINATION}")
    start = monotonic()
    config = TEMConfig(
        modules=(36,),
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
    runs = [run(config, seed=seed, oracle=True) for seed in (0, 1, 2)]
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": 1,
        "experiments": [
            {
                "id": 1,
                "what_changed": "Expanded oracle evaluation.",
                "runs": runs,
                "seconds": monotonic() - start,
            }
        ],
    }
    DESTINATION.write_text(json.dumps(finite(payload), indent=2, allow_nan=False) + "\n")
    signal.alarm(0)


if __name__ == "__main__":
    main()
