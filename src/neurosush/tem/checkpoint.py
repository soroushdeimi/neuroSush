"""Portable weights-only TEM snapshots (not optimizer-resume checkpoints)."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

import neurosush

from .config import TEMConfig
from .model import TEM


def _git_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def save_checkpoint(path: str | Path, model: TEM, *, seed: int, updates: int) -> None:
    """Atomically save tensors and plain configuration values with metadata."""
    if updates < 0:
        raise ValueError("updates must be nonnegative")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    config_dict = asdict(model.config)
    metadata = {
        "config": config_dict,
        "seed": seed,
        "updates": updates,
        "update": updates,
        "git_sha": _git_sha(),
        "library_version": str(neurosush.__version__),
        "library": str(neurosush.__version__),
        "torch_version": str(torch.__version__),
        "torch": str(torch.__version__),
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config_dict,
            "seed": seed,
            "updates": updates,
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: str | Path) -> tuple[TEM, int, int]:
    """Load on CPU using the restricted weights-only unpickler."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = TEMConfig(**payload["config"])
    seed = payload["seed"]
    updates = payload.get("updates", payload.get("metadata", {}).get("updates", 0))
    model = TEM(config, generator=torch.Generator().manual_seed(seed))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, seed, updates


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Read metadata dictionary, supporting old-format files without metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "metadata" in payload:
        return dict(payload["metadata"])
    return {
        "config": payload.get("config"),
        "seed": payload.get("seed"),
        "updates": payload.get("updates", 0),
        "update": payload.get("updates", 0),
        "git_sha": "unknown",
        "library_version": str(getattr(neurosush, "__version__", "0.7.0")),
        "library": str(getattr(neurosush, "__version__", "0.7.0")),
        "torch_version": str(torch.__version__),
        "torch": str(torch.__version__),
    }
