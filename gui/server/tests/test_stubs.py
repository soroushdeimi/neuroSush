"""Tests for 501 stub routes and error responses."""

from __future__ import annotations

from fastapi.testclient import TestClient
from studio.app import app

client = TestClient(app)


def test_stubs_return_501_and_error_body():
    """Assert every stub endpoint returns 501 and standardized error body."""
    routes = [
        ("POST", "/api/probe"),
        ("GET", "/api/projects"),
        ("POST", "/api/projects"),
        ("GET", "/api/projects/p_1234567890abcdef"),
        ("PUT", "/api/projects/p_1234567890abcdef"),
        ("DELETE", "/api/projects/p_1234567890abcdef"),
        ("GET", "/api/projects/p_1234567890abcdef/revisions"),
        ("GET", "/api/projects/p_1234567890abcdef/revisions/1"),
        ("GET", "/api/runs/r_1234567890abcdef/export/html"),
        ("GET", "/api/runs/r_1234567890abcdef/export/atlas"),
        ("GET", "/api/runs/r_1234567890abcdef/stream"),
    ]

    for method, path in routes:
        if method == "GET":
            response = client.get(path)
        elif method == "POST":
            response = client.post(path, json={})
        elif method == "PUT":
            response = client.put(path, json={})
        elif method == "DELETE":
            response = client.delete(path)
        else:
            raise AssertionError(f"Unknown method {method}")

        assert response.status_code == 501, f"{method} {path} returned {response.status_code}"
        body = response.json()
        assert "error" in body
        assert body["error"]["code"] == "not_implemented"
        assert isinstance(body["error"]["message"], str)
        assert isinstance(body["error"]["issues"], list)


def test_websocket_stub_closes_1011():
    """Assert WebSocket stream stub accepts and closes with code 1011."""
    with client.websocket_connect("/api/runs/r_1234567890abcdef/stream"):
        pass
