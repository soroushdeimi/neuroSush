"""Projects stub router (501 not implemented)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from studio.errors import ApiError

router = APIRouter()


@router.get("/projects")
def list_projects() -> Any:
    """Stub for listing projects."""
    raise ApiError("not_implemented", "Projects not implemented", 501)


@router.post("/projects")
def create_project() -> Any:
    """Stub for creating a project."""
    raise ApiError("not_implemented", "Projects not implemented", 501)


@router.get("/projects/{project_id}")
def get_project(project_id: str) -> Any:
    """Stub for getting a project."""
    raise ApiError("not_implemented", "Projects not implemented", 501)


@router.put("/projects/{project_id}")
def update_project(project_id: str) -> Any:
    """Stub for updating a project."""
    raise ApiError("not_implemented", "Projects not implemented", 501)


@router.delete("/projects/{project_id}")
def delete_project(project_id: str) -> Any:
    """Stub for deleting a project."""
    raise ApiError("not_implemented", "Projects not implemented", 501)


@router.get("/projects/{project_id}/revisions")
def list_revisions(project_id: str) -> Any:
    """Stub for listing revisions."""
    raise ApiError("not_implemented", "Projects not implemented", 501)


@router.get("/projects/{project_id}/revisions/{revision}")
def get_revision(project_id: str, revision: int) -> Any:
    """Stub for getting a revision."""
    raise ApiError("not_implemented", "Projects not implemented", 501)
