"""Regenerate T01 equation fixtures using the library (torch only)."""

import json
from pathlib import Path

import torch

from neurosush.memory.hebbian import attractor, hebbian_update


def generate() -> dict:
    """Exercise signed asymmetric weights, clipping, masks and zero iterations."""
    memory = torch.tensor([[0.9, -0.7], [0.2, -0.4]], dtype=torch.float64)
    pattern = torch.tensor([0.8, -0.6], dtype=torch.float64)
    predicted = torch.tensor([-0.3, 0.5], dtype=torch.float64)
    cue = torch.tensor([0.7, -0.8], dtype=torch.float64)
    updates = []
    for clip in (1.0, 0.2, None):
        opts = {"decay": 0.93, "rate": 0.7, "clip": clip}
        updates.append(
            {"opts": opts, "expected": hebbian_update(memory, pattern, predicted, **opts).tolist()}
        )
    retrievals = []
    for iterations in (0, 1, 5, 8):
        for mask in (None, [True, False]):
            opts = {"retention": 0.6, "negative_slope": 0.1, "cap": 0.5}
            result = attractor(
                cue, memory, iterations, **opts, mask=None if mask is None else torch.tensor(mask)
            )
            retrievals.append(
                {
                    "iterations": iterations,
                    "opts": {**opts, "mask": mask},
                    "expected": result.tolist(),
                }
            )
    return {
        "atol": 1e-12,
        "memory": memory.tolist(),
        "pattern": pattern.tolist(),
        "predicted": predicted.tolist(),
        "cue": cue.tolist(),
        "updates": updates,
        "retrievals": retrievals,
    }


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "tests/golden/tem_plate.json"
    destination.write_text(json.dumps(generate(), indent=2) + "\n", encoding="utf-8")
