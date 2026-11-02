"""Portable weights-only TEM snapshots (not optimizer-resume checkpoints)."""

from dataclasses import asdict
from pathlib import Path

import torch

from .config import TEMConfig
from .model import TEM


def save_checkpoint(path: str | Path, model: TEM, *, seed: int, updates: int) -> None:
    """Atomically save tensors and plain configuration values."""
    if updates < 0:
        raise ValueError("updates must be nonnegative")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(model.config),
            "seed": seed,
            "updates": updates,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: str | Path) -> tuple[TEM, int, int]:
    """Load on CPU using the restricted weights-only unpickler."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = TEMConfig(**payload["config"])
    model = TEM(config, generator=torch.Generator().manual_seed(payload["seed"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload["seed"], payload["updates"]
