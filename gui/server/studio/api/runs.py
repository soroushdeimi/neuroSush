"""Runs stub router (501 not implemented)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from studio.errors import ApiError

router = APIRouter()


@router.post("/runs", status_code=201)
def create_run() -> Any:
    """Stub for creating a run."""
    raise ApiError("not_implemented", "Runs not implemented", 501)


@router.get("/runs")
def list_runs() -> Any:
    """Stub for listing runs."""
    raise ApiError("not_implemented", "Runs not implemented", 501)


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> Any:
    """Stub for getting a run."""
    raise ApiError("not_implemented", "Runs not implemented", 501)


@router.post("/runs/{run_id}/pause")
def pause_run(run_id: str) -> Any:
    """Stub for pausing a run."""
    raise ApiError("not_implemented", "Runs not implemented", 501)


@router.post("/runs/{run_id}/resume")
def resume_run(run_id: str) -> Any:
    """Stub for resuming a run."""
    raise ApiError("not_implemented", "Runs not implemented", 501)


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> Any:
    """Stub for cancelling a run."""
    raise ApiError("not_implemented", "Runs not implemented", 501)


@router.get("/runs/{run_id}/trace")
def get_run_trace(run_id: str) -> Any:
    """Stub for getting run trace."""
    raise ApiError("not_implemented", "Runs not implemented", 501)


@router.get("/runs/{run_id}/export/html")
def export_run_html(run_id: str) -> Any:
    """Stub for exporting run HTML."""
    raise ApiError("not_implemented", "Runs not implemented", 501)


@router.get("/runs/{run_id}/export/atlas")
def export_run_atlas(run_id: str) -> Any:
    """Stub for exporting run Atlas zip."""
    raise ApiError("not_implemented", "Runs not implemented", 501)
