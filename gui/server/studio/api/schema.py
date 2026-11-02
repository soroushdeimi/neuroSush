"""Schema catalog router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response

from neurosush.introspect import catalog, fingerprint

router = APIRouter()


@router.get("/schema")
def schema(request: Request, response: Response) -> Any:
    """Return descriptors directly from the current library registry with ETag support."""
    fp = fingerprint()
    etag = f'"{fp}"'
    response.headers["ETag"] = etag

    if_none_match = request.headers.get("If-None-Match")
    if if_none_match:
        incoming = [e.strip() for e in if_none_match.split(",")]
        if etag in incoming or fp in incoming or "*" in incoming or f'W/"{fp}"' in incoming:
            return Response(status_code=304)

    return catalog()
