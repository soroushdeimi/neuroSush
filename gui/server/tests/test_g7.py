"""Tests for G7 work package (health fields, schema ETag/304, and CLI)."""

from __future__ import annotations

import sys
import warnings

import pytest
from fastapi.testclient import TestClient
from studio import cli
from studio.app import app

from neurosush.introspect import catalog, fingerprint

client = TestClient(app)


def test_health_fields() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["studio"], str)
    assert isinstance(data["library"], str)
    assert isinstance(data["torch"], str)
    assert isinstance(data["cuda"], bool)
    assert isinstance(data["devices"], list)
    assert "cpu" in data["devices"]


def test_schema_etag_and_304() -> None:
    res1 = client.get("/api/schema")
    assert res1.status_code == 200
    etag = res1.headers.get("ETag")
    assert etag is not None
    assert etag.strip('"') == fingerprint()
    assert res1.json() == catalog()

    res2 = client.get("/api/schema", headers={"If-None-Match": etag})
    assert res2.status_code == 304

    res3 = client.get("/api/schema", headers={"If-None-Match": fingerprint()})
    assert res3.status_code == 304

    res4 = client.get("/api/schema", headers={"If-None-Match": "*"})
    assert res4.status_code == 304

    res5 = client.get("/api/schema", headers={"If-None-Match": '"invalid"'})
    assert res5.status_code == 200


def test_cli_help(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["neurosush-studio", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 0


def test_cli_non_loopback_warning(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["neurosush-studio", "--host", "0.0.0.0", "--port", "12345"])
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **kw: None)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        code = cli.main()
        assert code == 0
        assert len(w) >= 1
        assert any("non-loopback" in str(warn.message) for warn in w)
