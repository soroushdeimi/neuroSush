"""Real library construction and initialization preflight checks."""

from __future__ import annotations

import pytest
from studio.errors import ApiError
from studio.pool.validation import validate_network
from studio.runner import resources


def test_empty_network_initializes():
    assert validate_network({"version": 2}) == {"valid": True, "errors": []}


def test_unknown_behavior_preserves_pointer():
    result = validate_network(
        {"version": 2, "groups": {"a/b": {"shape": 1, "behaviors": [{"kind": "unknown"}]}}}
    )
    assert result["valid"] is False
    assert result["errors"][0]["path"] == "/groups/a~1b/behaviors/0/kind"


def test_initialization_failure_is_invalid():
    result = validate_network(
        {"version": 2, "groups": {"g": {"shape": 1, "behaviors": [{"kind": "Fire"}]}}}
    )
    assert result["valid"] is False
    assert "needs a neuron model" in result["errors"][0]["message"]


def test_memory_refusal_uses_runner_budget(monkeypatch):
    payload = {"version": 2, "groups": {"g": {"shape": 1}}}
    estimate = resources.estimate_bytes(payload)
    monkeypatch.setattr(resources, "memory_limit", lambda: estimate - 1)
    with pytest.raises(ApiError) as caught:
        validate_network(payload)
    assert caught.value.code == "memory_budget"
    assert caught.value.status == 422
    assert "MB" in caught.value.issues[0]["message"]
    assert "limit" in caught.value.issues[0]["message"]
