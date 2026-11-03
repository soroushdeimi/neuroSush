"""HTTP dispatch to isolated v2 preflight."""

from fastapi.testclient import TestClient
from studio.app import app


def test_validate_v2_initializes():
    with TestClient(app) as client:
        response = client.post("/api/validate", json={"version": 2})
    assert response.status_code == 200
    assert response.json() == {"valid": True, "errors": []}


def test_validate_v2_initialization_failure():
    with TestClient(app) as client:
        response = client.post(
            "/api/validate",
            json={"version": 2, "groups": {"g": {"shape": 1, "behaviors": [{"kind": "Fire"}]}}},
        )
    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert "needs a neuron model" in response.json()["errors"][0]["message"]


def test_validate_v2_memory_refusal():
    import math

    from studio.runner.resources import estimate_bytes, memory_limit

    limit = memory_limit()
    size = math.isqrt(limit // (4 * 8)) + 1
    payload = {
        "version": 2,
        "groups": {"g": {"shape": size}},
        "synapses": [{"src": "g", "dst": "g"}],
    }
    estimate = estimate_bytes(payload)
    assert estimate > limit
    with TestClient(app) as client:
        response = client.post("/api/validate", json=payload)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "memory_budget"
    assert f"{estimate / 2**20:.1f} MB" in error["issues"][0]["message"]
    assert "limit" in error["issues"][0]["message"]


def test_validate_v2_unknown_behavior_pointer():
    with TestClient(app) as client:
        response = client.post(
            "/api/validate",
            json={
                "version": 2,
                "groups": {"a/b": {"shape": 1, "behaviors": [{"kind": "unknown"}]}},
            },
        )
    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert response.json()["errors"][0]["path"] == "/groups/a~1b/behaviors/0/kind"
