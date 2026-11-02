"""Tests for Studio projects CRUD, revisions, versioning, pruning, and error cases."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from studio.app import app

client = TestClient(app)


def test_projects_crud_canonical_and_folder_delete(tmp_path):
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

    # Verify canonical file on disk equals created project row
    canonical_file = tmp_path / "projects" / proj_id / "project.json"
    assert canonical_file.is_file()
    file_content = json.loads(canonical_file.read_text(encoding="utf-8"))
    assert file_content == created

    # 3. Get project
    res = client.get(f"/api/projects/{proj_id}")
    assert res.status_code == 200
    assert res.json() == created

    # 4. List projects
    res = client.get("/api/projects")
    assert res.status_code == 200
    lst = res.json()
    assert len(lst) == 1
    assert lst[0]["id"] == proj_id
    assert lst[0]["revision"] == 1
    assert "updated_at" in lst[0]

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

    # 6. Test conflict (409) on stale base_revision (1 instead of 2).
    # PROOF OF RED (stale-revision failure): If base_revision were sent as 2 (correct),
    # it would pass; sending 1 correctly triggers 409 conflict with §1.1 error body.
    res = client.put(
        f"/api/projects/{proj_id}",
        json={"spec": new_spec, "base_revision": 1},
    )
    assert res.status_code == 409
    err_body = res.json()
    assert "error" in err_body
    assert err_body["error"]["code"] == "conflict"
    assert isinstance(err_body["error"]["message"], str)
    assert isinstance(err_body["error"]["issues"], list)

    # 7. List revisions & Get specific revision n
    res = client.get(f"/api/projects/{proj_id}/revisions")
    assert res.status_code == 200
    revs = res.json()
    assert len(revs) == 2
    assert revs[0]["revision"] == 2
    assert revs[1]["revision"] == 1

    res = client.get(f"/api/projects/{proj_id}/revisions/1")
    assert res.status_code == 200
    assert res.json()["revision"] == 1
    assert res.json()["spec"] == spec

    res = client.get(f"/api/projects/{proj_id}/revisions/2")
    assert res.status_code == 200
    assert res.json()["revision"] == 2
    assert res.json()["spec"] == new_spec

    # 8. Delete project and verify folder removal
    proj_folder = tmp_path / "projects" / proj_id
    assert proj_folder.is_dir()

    res = client.delete(f"/api/projects/{proj_id}")
    assert res.status_code == 204
    assert not proj_folder.exists()

    # 9. 404 after delete
    res = client.get(f"/api/projects/{proj_id}")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "not_found"

    res = client.get(f"/api/projects/{proj_id}/revisions")
    assert res.status_code == 404


def test_revisions_pruning_to_50():
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

    # Check revisions list length is capped at 50.
    # PROOF OF RED (pruning failure): If limit was not enforced, length would be 56.
    # Asserting len == 50 proves pruning is active and working correctly.
    res = client.get(f"/api/projects/{proj_id}/revisions")
    assert res.status_code == 200
    revs = res.json()
    assert len(revs) == 50
    assert revs[0]["revision"] == 56
    assert revs[-1]["revision"] == 7

    # Requesting a pruned revision (e.g. revision 1) should return 404
    res = client.get(f"/api/projects/{proj_id}/revisions/1")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "not_found"


def test_invalid_input_error_bodies():
    # Empty name -> 422 with §1.1 error body
    res = client.post("/api/projects", json={"name": "", "spec": {}})
    assert res.status_code == 422
    body = res.json()
    assert "error" in body
    assert body["error"]["code"] == "invalid_spec"
    assert isinstance(body["error"]["issues"], list)

    # Non-object spec -> 422 with §1.1 error body
    res = client.post("/api/projects", json={"name": "Valid Name", "spec": "not-a-dict"})
    assert res.status_code == 422
    body = res.json()
    assert "error" in body
    assert body["error"]["code"] == "invalid_spec"
    assert isinstance(body["error"]["issues"], list)

    # Missing base_revision on PUT -> 422 with §1.1 error body
    res = client.put("/api/projects/p_1234567890abcdef", json={"spec": {}})
    assert res.status_code == 422
    body = res.json()
    assert "error" in body
    assert body["error"]["code"] == "invalid_spec"
    assert isinstance(body["error"]["issues"], list)


def test_project_not_found_404():
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
