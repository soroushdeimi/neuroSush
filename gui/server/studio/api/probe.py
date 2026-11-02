"""Probe stub router (501 not implemented)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from studio.errors import ApiError

router = APIRouter()


@router.post("/probe")
def probe() -> Any:
    """Stub for model probe."""
    raise ApiError("not_implemented", "Model probe not implemented", 501)
