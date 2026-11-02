"""SQLite storage and canonical JSON file handling for Studio projects."""

from __future__ import annotations

import json
import secrets
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neurosush.introspect import fingerprint
from studio.config import config
from studio.errors import ApiError


def _get_db_path() -> Path:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    return config.data_dir / "studio.db"


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Provide a managed sqlite3 connection with foreign keys enabled."""
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Initialize SQLite database schema and migrations."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
            """
        )
        cursor = conn.execute("SELECT MAX(version) FROM schema_migrations")
        row = cursor.fetchone()
        version = row[0] if row and row[0] is not None else 0

        if version < 1:
            conn.execute(
                """
                CREATE TABLE projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    catalog_fingerprint TEXT NOT NULL,
                    spec TEXT NOT NULL,
                    layout TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE revisions (
                    project_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    saved_at TEXT NOT NULL,
                    catalog_fingerprint TEXT NOT NULL,
                    spec TEXT NOT NULL,
                    layout TEXT NOT NULL,
                    PRIMARY KEY (project_id, revision),
                    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
                )
                """
            )
            now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (1, now),
            )


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _row_to_project(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "revision": int(row["revision"]),
        "catalog_fingerprint": row["catalog_fingerprint"],
        "spec": json.loads(row["spec"]),
        "layout": json.loads(row["layout"]),
    }


def _write_canonical_file(project: dict[str, Any]) -> None:
    proj_dir = config.data_dir / "projects" / project["id"]
    proj_dir.mkdir(parents=True, exist_ok=True)
    file_path = proj_dir / "project.json"
    content = json.dumps(project, indent=2, sort_keys=True) + "\n"
    file_path.write_text(content, encoding="utf-8")


def list_projects() -> list[dict[str, Any]]:
    """List all projects ordered by updated_at descending."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT id, name, updated_at, revision FROM projects ORDER BY updated_at DESC"
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "updated_at": r["updated_at"],
                "revision": int(r["revision"]),
            }
            for r in rows
        ]


def create_project(
    name: str, spec: dict[str, Any], layout: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create a new project and revision 1, writing its canonical project.json."""
    init_db()
    project_id = "p_" + secrets.token_hex(8)
    now = _now_utc()
    revision = 1
    fp = fingerprint()
    layout_dict = layout if layout is not None else {}
    spec_str = json.dumps(spec)
    layout_str = json.dumps(layout_dict)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO projects
            (id, name, created_at, updated_at, revision, catalog_fingerprint, spec, layout)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (project_id, name, now, now, revision, fp, spec_str, layout_str),
        )
        conn.execute(
            """
            INSERT INTO revisions
            (project_id, revision, saved_at, catalog_fingerprint, spec, layout)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (project_id, revision, now, fp, spec_str, layout_str),
        )

    project = {
        "id": project_id,
        "name": name,
        "created_at": now,
        "updated_at": now,
        "revision": revision,
        "catalog_fingerprint": fp,
        "spec": spec,
        "layout": layout_dict,
    }
    _write_canonical_file(project)
    return project


def get_project(project_id: str) -> dict[str, Any] | None:
    """Retrieve project by ID or None if not found."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return _row_to_project(row)


def update_project(
    project_id: str,
    base_revision: int,
    name: str | None = None,
    spec: dict[str, Any] | None = None,
    layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update project with base_revision check, adding a new revision and pruning."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        if not row:
            raise ApiError("not_found", f"Project {project_id} not found", 404)
        current = _row_to_project(row)
        if current["revision"] != base_revision:
            msg = f"Stale base_revision {base_revision}, current is {current['revision']}"
            raise ApiError("conflict", msg, 409)

        new_revision = current["revision"] + 1
        now = _now_utc()
        new_name = name if name is not None else current["name"]
        new_spec = spec if spec is not None else current["spec"]
        new_layout = layout if layout is not None else current["layout"]
        fp = fingerprint()

        spec_str = json.dumps(new_spec)
        layout_str = json.dumps(new_layout)

        conn.execute(
            """
            UPDATE projects
            SET name = ?, updated_at = ?, revision = ?,
                catalog_fingerprint = ?, spec = ?, layout = ?
            WHERE id = ?
            """,
            (new_name, now, new_revision, fp, spec_str, layout_str, project_id),
        )
        conn.execute(
            """
            INSERT INTO revisions
            (project_id, revision, saved_at, catalog_fingerprint, spec, layout)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (project_id, new_revision, now, fp, spec_str, layout_str),
        )

        # Prune revisions to keep only the latest 50
        conn.execute(
            """
            DELETE FROM revisions
            WHERE project_id = ? AND revision NOT IN (
                SELECT revision FROM revisions
                WHERE project_id = ?
                ORDER BY revision DESC
                LIMIT 50
            )
            """,
            (project_id, project_id),
        )

    updated = {
        "id": project_id,
        "name": new_name,
        "created_at": current["created_at"],
        "updated_at": now,
        "revision": new_revision,
        "catalog_fingerprint": fp,
        "spec": new_spec,
        "layout": new_layout,
    }
    _write_canonical_file(updated)
    return updated


def delete_project(project_id: str) -> bool:
    """Delete project and its revisions from database and disk."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
        if not cursor.fetchone():
            return False
        conn.execute("DELETE FROM revisions WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    import shutil

    proj_dir = config.data_dir / "projects" / project_id
    if proj_dir.is_dir():
        shutil.rmtree(proj_dir)
    return True


def list_revisions(project_id: str) -> list[dict[str, Any]] | None:
    """List revisions for a project."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
        if not cursor.fetchone():
            return None
        cursor = conn.execute(
            """
            SELECT revision, saved_at FROM revisions
            WHERE project_id = ? ORDER BY revision DESC
            """,
            (project_id,),
        )
        rows = cursor.fetchall()
        return [{"revision": int(r["revision"]), "saved_at": r["saved_at"]} for r in rows]


def get_revision(project_id: str, revision: int) -> dict[str, Any] | None:
    """Get project state at a specific revision."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT id, name, created_at FROM projects WHERE id = ?", (project_id,)
        )
        p_row = cursor.fetchone()
        if not p_row:
            return None
        cursor = conn.execute(
            """
            SELECT * FROM revisions
            WHERE project_id = ? AND revision = ?
            """,
            (project_id, revision),
        )
        r_row = cursor.fetchone()
        if not r_row:
            return None
        return {
            "id": p_row["id"],
            "name": p_row["name"],
            "created_at": p_row["created_at"],
            "updated_at": r_row["saved_at"],
            "revision": int(r_row["revision"]),
            "catalog_fingerprint": r_row["catalog_fingerprint"],
            "spec": json.loads(r_row["spec"]),
            "layout": json.loads(r_row["layout"]),
        }
