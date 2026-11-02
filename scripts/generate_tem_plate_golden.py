"""Regenerate T01 equation fixtures using the library (torch only)."""

import json
from pathlib import Path

import torch

from neurosush.memory.hebbian import attractor, hebbian_update


def generate() -> dict:
    """Exercise signed asymmetric weights, clipping, masks and

    zero iterations, plus T03 recall cases.
    """
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
    # T03 golden recall cases
    n = 16
    torch.manual_seed(42)
    patterns = []
    for _i in range(4):
        p = torch.zeros(n, dtype=torch.float64)
        indices = torch.randperm(n)[:4]
        p[indices] = 1.0
        patterns.append(p)

    mem = torch.zeros((n, n), dtype=torch.float64)
    for p in patterns:
        mem = hebbian_update(mem, p, torch.zeros_like(p), decay=0.9999, rate=0.5, clip=1.0)

    target = patterns[0]
    t03_cue = target.clone()
    active_indices = (target > 0.5).nonzero(as_tuple=True)[0]
    corrupted_indices = active_indices[:2]
    t03_cue[corrupted_indices] = 0.0

    retrieval_history = []
    overlap_history = []
    h = t03_cue.clone()
    for _ in range(6):
        retrieval_history.append(h.tolist())
        num = torch.dot(h, target)
        den = h.norm() * target.norm() + 1e-8
        overlap = (num / den).item()
        overlap_history.append(overlap)
        h = attractor(h, mem, iterations=1, retention=0.8, negative_slope=0.01, cap=1.0)

    return {
        "atol": 1e-12,
        "memory": memory.tolist(),
        "pattern": pattern.tolist(),
        "predicted": predicted.tolist(),
        "cue": cue.tolist(),
        "updates": updates,
        "retrievals": retrievals,
        "t03": {
            "patterns": [p.tolist() for p in patterns],
            "memory": mem.tolist(),
            "target": target.tolist(),
            "cue": t03_cue.tolist(),
            "retrieval_history": retrieval_history,
            "overlap_history": overlap_history,
        },
    }


if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "tests/golden/tem_plate.json"
    destination.write_text(json.dumps(generate(), indent=2) + "\n", encoding="utf-8")
