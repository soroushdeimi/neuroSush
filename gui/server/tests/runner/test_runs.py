"""Acceptance coverage for real spawned simulation processes."""

from __future__ import annotations

import json
import signal
import time

import pytest
import torch
from fastapi.testclient import TestClient
from studio.api.runs import get_manager
from studio.app import app
from studio.config import Config
from studio.errors import ApiError
from studio.runner import RunManager
from studio.runner.resources import estimate_bytes, memory_limit

from neurosush.structure.network_spec import build_network, load_spec
from neurosush.viz.trace import unpack_array


@pytest.fixture
def manager(tmp_path):
    instance = RunManager(Config(data_dir=tmp_path, max_jobs=1, max_steps=10_000_000))
    yield instance
    instance.close()


@pytest.fixture
def client(manager):
    app.dependency_overrides[get_manager] = lambda: manager
    with TestClient(app) as session:
        yield session
    app.dependency_overrides.clear()


def request(**overrides):
    return {
        "spec": {
            "version": 2,
            "network": {"seed": 7},
            "groups": {
                "exc": {
                    "shape": 4,
                    "behaviors": [
                        {
                            "kind": "LIF",
                            "params": {
                                "tau": 10.0,
                                "threshold": -50.0,
                                "v_reset": -70.0,
                                "v_rest": -65.0,
                            },
                        },
                        {"kind": "Fire"},
                    ],
                }
            },
            "recordings": [{"target": "exc", "attributes": ["v", "spikes"], "max_steps": 5000}],
        },
        "steps": 20,
        "chunk": 5,
        "mode": "fast",
        "speed": 1.0,
        **overrides,
    }


def wait(manager, run_id, statuses, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = manager.get(run_id)
        if info["status"] in statuses:
            return info
        assert info["status"] != "failed", info
        time.sleep(0.01)
    pytest.fail(f"Timed out: {manager.get(run_id)}")


def test_artifacts_and_trace(client, manager):
    response = client.post("/api/runs", json=request())
    assert response.status_code == 201
    run_id = response.json()["id"]
    assert len(run_id) == 18
    assert run_id.startswith("r_")
    info = wait(manager, run_id, {"completed"})
    assert info["step"] == 20
    assert info["started_at"].endswith("Z")
    assert info["finished_at"].endswith("Z")
    root = manager.config.data_dir / "runs" / run_id
    assert {p.name for p in root.iterdir()} == {
        "spec.json",
        "provenance.json",
        "log.jsonl",
        "traces.json.gz",
        "checkpoint.pt",
    }
    trace = client.get(f"/api/runs/{run_id}/trace", params={"target": "exc"}).json()
    assert len(trace["steps"]) == 20
    assert trace["meta"]["run_id"] == run_id
    assert trace["meta"]["spec_sha256"] == info["provenance"]["spec_sha256"]
    assert trace["meta"]["attributes"] == ["v", "spikes"]
    reference = build_network(load_spec(json.dumps(request()["spec"])))
    reference.net.run(20)
    expected = reference.recorders["exc"].trace()
    for attribute in ("v", "spikes"):
        torch.testing.assert_close(
            unpack_array(trace["series"][attribute]), expected.series[attribute]
        )
    checkpoint = torch.load(root / "checkpoint.pt", weights_only=True)
    assert checkpoint["iteration"] == 20
    assert client.get(f"/api/runs/{run_id}").json() == info
    assert client.get("/api/runs").json() == [info]
    assert client.get(f"/api/runs/{run_id}/trace?target=missing").status_code == 404
    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 409
    assert client.get("/api/runs/r_missing").status_code == 404
    assert all(
        json.loads(line)["id"] == run_id for line in (root / "log.jsonl").read_text().splitlines()
    )


def test_pause_resume_and_fifo(manager):
    first = manager.submit(request(steps=5000, mode="watch", chunk=10))["id"]
    wait(manager, first, {"running"})
    second = manager.submit(request())["id"]
    third = manager.submit(request())["id"]
    assert manager.get(second)["status"] == manager.get(third)["status"] == "queued"
    manager.control(first, "pause")
    paused = wait(manager, first, {"paused"})
    time.sleep(0.1)
    assert manager.get(first)["step"] == paused["step"]
    manager.control(first, "resume")
    wait(manager, first, {"running"})
    deadline = time.monotonic() + 2
    while manager.get(first)["step"] <= paused["step"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.get(first)["step"] > paused["step"]
    manager.control(first, "cancel")
    wait(manager, first, {"cancelled"})
    two = wait(manager, second, {"completed"})
    three = wait(manager, third, {"completed"})
    assert two["started_at"] < two["finished_at"] <= three["started_at"]


def test_cancel_mid_chunk(manager):
    run_id = manager.submit(request(steps=10_000_000, chunk=10_000_000))["id"]
    wait(manager, run_id, {"running"})
    start = time.monotonic()
    manager.control(run_id, "cancel")
    wait(manager, run_id, {"cancelled"}, timeout=6)
    assert time.monotonic() - start < 6
    assert manager.get(run_id)["step"] == 0


def stubborn_child(body, directory, pipe):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pipe.send({"status": "running", "step": 0})
    while True:
        time.sleep(0.1)


def test_cancel_escalates_to_kill(manager, monkeypatch):
    monkeypatch.setattr("studio.runner.manager.execute", stubborn_child)
    run_id = manager.submit(request())["id"]
    wait(manager, run_id, {"running"})
    start = time.monotonic()
    manager.control(run_id, "cancel")
    wait(manager, run_id, {"cancelled"}, timeout=6)
    elapsed = time.monotonic() - start
    assert 5 <= elapsed < 6


def test_crash_becomes_failed(manager):
    body = request()
    body["spec"]["groups"]["exc"]["behaviors"] = [{"kind": "DoesNotExist"}]
    run_id = manager.submit(body)["id"]
    deadline = time.monotonic() + 15
    while manager.get(run_id)["status"] != "failed" and time.monotonic() < deadline:
        time.sleep(0.01)
    info = manager.get(run_id)
    assert info["status"] == "failed"
    assert "DoesNotExist" in info["error"]["message"]
    assert "Traceback" in info["error"]["traceback"]


def test_limits_and_invalid_requests(client, monkeypatch):
    monkeypatch.setattr("studio.runner.resources.memory_limit", lambda: 2**30)
    body = request()
    body["spec"]["groups"] = {"exc": {"shape": 50000}}
    body["spec"]["synapses"] = [{"src": "exc", "dst": "exc"}]
    response = client.post("/api/runs", json=body)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "memory_budget"
    assert "MB" in error["issues"][0]["message"]
    assert "1024.0" in error["issues"][0]["message"]
    assert (
        client.post("/api/runs", json=request(steps=10_000_001)).json()["error"]["code"]
        == "too_many_steps"
    )
    for overrides in (
        {"steps": 0},
        {"chunk": 0},
        {"speed": 0},
        {"mode": "invalid"},
        {"steps": True},
    ):
        assert client.post("/api/runs", json=request(**overrides)).status_code == 422
    body = request()
    body["spec"]["version"] = 3
    assert client.post("/api/runs", json=body).json()["error"]["code"] == "invalid_spec"


def test_queued_cancel_and_bad_transitions(manager):
    first = manager.submit(request(steps=5000, mode="watch"))["id"]
    wait(manager, first, {"running"})
    second = manager.submit(request())["id"]
    with pytest.raises(ApiError, match="Cannot resume"):
        manager.control(first, "resume")
    assert manager.control(second, "cancel")["status"] == "cancelled"
    manager.control(first, "cancel")
    wait(manager, first, {"cancelled"})
    assert manager.get(second)["started_at"] is None


def test_memory_estimate_includes_columns_and_recordings():
    spec = request()["spec"]
    base = estimate_bytes(spec)
    spec["recordings"][0]["max_steps"] *= 2
    assert estimate_bytes(spec) > base
    spec["columns"] = {"c": {"layers": {"l": {"groups": {"g": {"shape": [2, 3, 4]}}}}}}
    assert estimate_bytes(spec) > base
    assert memory_limit() > 0


def hard_crash(body, directory, pipe):
    import os

    os._exit(17)


def test_abnormal_exit_is_failed(manager, monkeypatch):
    monkeypatch.setattr("studio.runner.manager.execute", hard_crash)
    run_id = manager.submit(request())["id"]
    info = wait(manager, run_id, {"failed"})
    assert "17" in info["error"]["message"]
    assert info["finished_at"] is not None


def test_cgroup_limit(monkeypatch):
    def read_limit(path):
        if str(path) == "/sys/fs/cgroup/memory.max":
            return "max\n"
        if str(path) == "/sys/fs/cgroup/memory/memory.limit_in_bytes":
            return "1048576\n"
        raise OSError("absent")

    monkeypatch.setattr("studio.runner.resources.os.sysconf", lambda key: 2**20)
    monkeypatch.setattr("studio.runner.resources.Path.read_text", read_limit)
    assert memory_limit() == 524288
