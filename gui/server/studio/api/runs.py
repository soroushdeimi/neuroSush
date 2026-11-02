"""Run submission, controls and recorded traces."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from studio.config import config
from studio.errors import ApiError
from studio.runner import RunManager

router = APIRouter()


class RunRequest(BaseModel):
    """Validated simulation request."""

    spec: dict[str, Any]
    project_id: str | None = None
    steps: int = Field(ge=1, strict=True)
    chunk: int = Field(default=100, ge=1, strict=True)
    mode: Literal["watch", "fast"] = "watch"
    speed: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    stream: dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def get_manager() -> RunManager:
    """Create the scheduler on first use."""
    return RunManager(config)


@router.post("/runs", status_code=201)
def create_run(body: RunRequest, manager: Annotated[RunManager, Depends(get_manager)]) -> Any:
    """Queue a simulation."""
    return manager.submit(body.model_dump())


@router.get("/runs")
def list_runs(manager: Annotated[RunManager, Depends(get_manager)]) -> Any:
    """List submitted runs."""
    return manager.list()


@router.get("/runs/{run_id}")
def get_run(run_id: str, manager: Annotated[RunManager, Depends(get_manager)]) -> Any:
    """Return a run snapshot."""
    return manager.get(run_id)


@router.post("/runs/{run_id}/pause")
def pause_run(run_id: str, manager: Annotated[RunManager, Depends(get_manager)]) -> Any:
    """Pause at a chunk boundary."""
    return manager.control(run_id, "pause")


@router.post("/runs/{run_id}/resume")
def resume_run(run_id: str, manager: Annotated[RunManager, Depends(get_manager)]) -> Any:
    """Resume a paused run."""
    return manager.control(run_id, "resume")


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, manager: Annotated[RunManager, Depends(get_manager)]) -> Any:
    """Terminate a run."""
    return manager.control(run_id, "cancel")


@router.get("/runs/{run_id}/trace")
def get_run_trace(
    run_id: str, target: str, manager: Annotated[RunManager, Depends(get_manager)]
) -> Any:
    """Read a recorded target."""
    return manager.trace(run_id, target)


@router.get("/runs/{run_id}/export/html")
def export_run_html(run_id: str) -> Any:
    """Stub for exporting run HTML."""
    raise ApiError("not_implemented", "Runs not implemented", 501)


@router.get("/runs/{run_id}/export/atlas")
def export_run_atlas(run_id: str) -> Any:
    """Stub for exporting run Atlas zip."""
    raise ApiError("not_implemented", "Runs not implemented", 501)
