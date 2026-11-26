"""Live run WebSocket router."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, WebSocket

from studio.api.runs import get_manager
from studio.errors import ApiError
from studio.runner.manager import RunManager
from studio.stream.ws import relay

router = APIRouter()


@router.websocket("/runs/{run_id}/stream")
async def run_stream(
    websocket: WebSocket, run_id: str, manager: Annotated[RunManager, Depends(get_manager)]
) -> None:
    """Relay authenticated live run events."""
    await relay(websocket, run_id, manager)


@router.get("/runs/{run_id}/stream")
def run_stream_http(run_id: str) -> Any:
    """HTTP fallback for stream stub."""
    raise ApiError("not_implemented", "Stream not implemented", 501)
