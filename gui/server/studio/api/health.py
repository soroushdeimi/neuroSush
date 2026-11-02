"""Health check router."""

from __future__ import annotations

import importlib.metadata
from typing import Any

import torch
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    """Report process readiness and version info."""
    try:
        studio_ver = importlib.metadata.version("neurosush-studio")
    except Exception:
        studio_ver = "0.1.0"

    try:
        library_ver = importlib.metadata.version("neurosush")
    except Exception:
        import neurosush

        library_ver = getattr(neurosush, "__version__", "0.1.0")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    return {
        "status": "ok",
        "studio": studio_ver,
        "library": library_ver,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "devices": devices,
    }
