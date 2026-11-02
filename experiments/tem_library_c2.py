"""Bounded CPU TEM diagnostic using only library training and evaluation."""

from __future__ import annotations

import json
import platform
import signal
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import torch

from neurosush.tem import TEMConfig, evaluate, train


def main() -> None:
    """Run one seed, abort after 570 seconds, and save the measured results."""
    signal.alarm(570)
    torch.set_num_threads(1)
    start = monotonic()
    oracle_config = TEMConfig(
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
    oracle = train(oracle_config, steps=400, seed=0, oracle=True)
    oracle_scores = evaluate(
        oracle.model, oracle_config, environments=4, steps=400, seed=1234, oracle=True
    )
    print("oracle", json.dumps(oracle_scores), flush=True)
    if oracle_scores["accuracy"] <= oracle_scores["frequency"] + 0.10:
        raise RuntimeError(f"oracle failed: {oracle_scores}")
    config = TEMConfig(
        batch=8, widths=(4,), decay_ramp=10, rate_ramp=80, latent_ramp=80, lr_every=400
    )
    result = train(config, steps=12000, seed=0, log_every=100)
    learned_scores = evaluate(result.model, config, environments=4, steps=400, seed=1234)
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 1,
        "training_seed": 0,
        "evaluation_seed": 1234,
        "environments": 4,
        "evaluation_steps": 400,
        "oracle": {
            "config": asdict(oracle_config),
            "training_steps": 400,
            "training_seconds": oracle.seconds,
            "evaluation": oracle_scores,
        },
        "learned": {
            "config": asdict(config),
            "training_steps": 12000,
            "training_seconds": result.seconds,
            "evaluation": learned_scores,
            "history": result.history,
        },
        "total_seconds": monotonic() - start,
    }
    destination = Path(__file__).parent / "results" / "tem_library_c2.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print("learned", json.dumps(learned_scores), flush=True)
    print(f"seconds={payload['total_seconds']:.3f} output={destination}", flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
