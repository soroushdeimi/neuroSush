"""Tests for Studio projects CRUD, revisions, versioning, pruning, and error cases."""

from __future__ import annotations

import studio.projects.db as projects_db
from fastapi.testclient import TestClient
from studio.app import app
from studio.config import Config

client = TestClient(app)


def test_projects_crud_and_canonical_file(tmp_path, monkeypatch):
    monkeypatch.setattr(projects_db, "config", Config(data_dir=tmp_path))

    # 1. List empty
    res = client.get("/api/projects")
    assert res.status_code == 200
    assert res.json() == []

    # 2. Create project
    spec = {"version": 2, "network": {"dt": 1.0}}
    layout = {"node_1": {"x": 10.0, "y": 20.0}}
    res = client.post(
        "/api/projects",
        json={"name": "Test Project", "spec": spec, "layout": layout},
    )
    assert res.status_code == 201
    created = res.json()
    assert created["name"] == "Test Project"
    assert created["revision"] == 1
    assert created["spec"] == spec
    assert created["layout"] == layout
    assert "id" in created
    assert created["id"].startswith("p_")

    proj_id = created["id"]

    # Verify canonical file on disk
    canonical_file = tmp_path / "projects" / proj_id / "project.json"
    assert canonical_file.is_file()

    # 3. Get project
    res = client.get(f"/api/projects/{proj_id}")
    assert res.status_code == 200
    assert res.json()["name"] == "Test Project"
    assert res.json()["revision"] == 1

    # 4. List projects
    res = client.get("/api/projects")
    assert res.status_code == 200
    lst = res.json()
    assert len(lst) == 1
    assert lst[0]["id"] == proj_id
    assert lst[0]["revision"] == 1

    # 5. Update project with correct base_revision (1)
    new_spec = {"version": 2, "network": {"dt": 0.5}}
    res = client.put(
        f"/api/projects/{proj_id}",
        json={"spec": new_spec, "base_revision": 1},
    )
    assert res.status_code == 200
    updated = res.json()
    assert updated["revision"] == 2
    assert updated["spec"] == new_spec

    # 6. Test conflict (409) on stale base_revision (1 instead of 2)
    res = client.put(
        f"/api/projects/{proj_id}",
        json={"spec": new_spec, "base_revision": 1},
    )
    assert res.status_code == 409
    err = res.json()["error"]
    assert err["code"] == "conflict"

    # 7. List revisions
    res = client.get(f"/api/projects/{proj_id}/revisions")
    assert res.status_code == 200
    revs = res.json()
    assert len(revs) == 2
    assert revs[0]["revision"] == 2
    assert revs[1]["revision"] == 1

    # 8. Get specific revision
    res = client.get(f"/api/projects/{proj_id}/revisions/1")
    assert res.status_code == 200
    assert res.json()["revision"] == 1
    assert res.json()["spec"] == spec

    res = client.get(f"/api/projects/{proj_id}/revisions/2")
    assert res.status_code == 200
    assert res.json()["revision"] == 2
    assert res.json()["spec"] == new_spec

    # 9. Delete project
    res = client.delete(f"/api/projects/{proj_id}")
    assert res.status_code == 204

    # 10. 404 after delete
    res = client.get(f"/api/projects/{proj_id}")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "not_found"

    res = client.get(f"/api/projects/{proj_id}/revisions")
    assert res.status_code == 404


def test_revisions_pruning_to_50(tmp_path, monkeypatch):
    monkeypatch.setattr(projects_db, "config", Config(data_dir=tmp_path))

    spec = {"version": 2}
    res = client.post("/api/projects", json={"name": "Prune Project", "spec": spec})
    assert res.status_code == 201
    proj_id = res.json()["id"]

    # Perform 55 updates (revisions 2 to 56)
    for i in range(2, 57):
        rev = i - 1
        res = client.put(
            f"/api/projects/{proj_id}",
            json={"spec": {"version": i}, "base_revision": rev},
        )
        assert res.status_code == 200

    # Check revisions list length is capped at 50
    res = client.get(f"/api/projects/{proj_id}/revisions")
    assert res.status_code == 200
    revs = res.json()
    assert len(revs) == 50
    # Latest revision should be 56, oldest kept should be 7 (56 - 50 + 1 = 7)
    assert revs[0]["revision"] == 56
    assert revs[-1]["revision"] == 7

    # Requesting a pruned revision (e.g. revision 1) should return 404
    res = client.get(f"/api/projects/{proj_id}/revisions/1")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "not_found"


def test_project_not_found_404(tmp_path, monkeypatch):
    monkeypatch.setattr(projects_db, "config", Config(data_dir=tmp_path))

    fake_id = "p_0000000000000000"
    assert client.get(f"/api/projects/{fake_id}").status_code == 404
    assert (
        client.put(
            f"/api/projects/{fake_id}",
            json={"base_revision": 1},
        ).status_code
        == 404
    )
    assert client.delete(f"/api/projects/{fake_id}").status_code == 404
    assert client.get(f"/api/projects/{fake_id}/revisions").status_code == 404
    assert client.get(f"/api/projects/{fake_id}/revisions/1").status_code == 404
