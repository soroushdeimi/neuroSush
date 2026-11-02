"""Projects management router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from studio.errors import ApiError
from studio.projects import db

router = APIRouter()


@router.get("/projects")
def list_projects() -> list[dict[str, Any]]:
    """List all projects."""
    return db.list_projects()


@router.post("/projects", status_code=status.HTTP_201_CREATED)
def create_project(body: dict[str, Any]) -> dict[str, Any]:
    """Create a new project."""
    name = body.get("name")
    spec = body.get("spec")
    layout = body.get("layout")
    if not name or not isinstance(name, str):
        raise ApiError("invalid_spec", "Missing or invalid project name", 422)
    if spec is None or not isinstance(spec, dict):
        raise ApiError("invalid_spec", "Missing or invalid project spec", 422)
    return db.create_project(name=name, spec=spec, layout=layout)


@router.get("/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    """Get project by ID."""
    project = db.get_project(project_id)
    if not project:
        raise ApiError("not_found", f"Project {project_id} not found", 404)
    return project


@router.put("/projects/{project_id}")
def update_project(project_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Update project with base_revision check."""
    base_revision = body.get("base_revision")
    if base_revision is None or not isinstance(base_revision, int):
        raise ApiError("invalid_spec", "Missing or invalid base_revision", 422)
    name = body.get("name")
    spec = body.get("spec")
    layout = body.get("layout")

    return db.update_project(
        project_id=project_id,
        base_revision=base_revision,
        name=name,
        spec=spec,
        layout=layout,
    )


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: str) -> Response:
    """Delete a project."""
    success = db.delete_project(project_id)
    if not success:
        raise ApiError("not_found", f"Project {project_id} not found", 404)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/projects/{project_id}/revisions")
def list_revisions(project_id: str) -> list[dict[str, Any]]:
    """List revisions for a project."""
    revs = db.list_revisions(project_id)
    if revs is None:
        raise ApiError("not_found", f"Project {project_id} not found", 404)
    return revs


@router.get("/projects/{project_id}/revisions/{revision}")
def get_revision(project_id: str, revision: int) -> dict[str, Any]:
    """Get project state at a specific revision."""
    rev_proj = db.get_revision(project_id, revision)
    if not rev_proj:
        raise ApiError("not_found", f"Project {project_id} revision {revision} not found", 404)
    return rev_proj
