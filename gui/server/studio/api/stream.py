"""WebSocket stream stub router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, WebSocket

from studio.errors import ApiError

router = APIRouter()


@router.websocket("/runs/{run_id}/stream")
async def run_stream(websocket: WebSocket, run_id: str) -> None:
    """Accept WebSocket connection and close with code 1011 (not implemented)."""
    await websocket.accept()
    await websocket.close(code=1011)


@router.get("/runs/{run_id}/stream")
def run_stream_http(run_id: str) -> Any:
    """HTTP fallback for stream stub."""
    raise ApiError("not_implemented", "Stream not implemented", 501)
