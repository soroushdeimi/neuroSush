"""Schema catalog router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from neurosush.introspect import catalog

router = APIRouter()


@router.get("/schema")
def schema() -> dict[str, Any]:
    """Return descriptors directly from the current library registry."""
    return catalog()
