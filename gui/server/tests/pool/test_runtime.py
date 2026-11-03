"""Real spawned worker failures and recovery through HTTP."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection

import pytest
from fastapi.testclient import TestClient
from studio.app import app
from studio.pool import runtime
from studio.pool.runtime import WarmPool, _serve

from neurosush.neurons.models import LIF
from neurosush.structure.spec import _REGISTRY

PARAMS = {"tau": 10.0, "threshold": -50.0, "v_reset": -65.0, "v_rest": -65.0}


class SleepingLIF(LIF):
    def forward(self, group):
        time.sleep(11)
        super().forward(group)


class SlowLIF(LIF):
    def forward(self, group):
        time.sleep(9)
        super().forward(group)


class CrashingLIF(LIF):
    def forward(self, group):
        os._exit(1)


def failing_worker(pipe: Connection) -> None:
    _REGISTRY["SlowLIF"] = SlowLIF
    _REGISTRY["SleepingLIF"] = SleepingLIF
    _REGISTRY["CrashingLIF"] = CrashingLIF
    _serve(pipe)


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(runtime, "_serve", failing_worker)
    monkeypatch.setitem(_REGISTRY, "SleepingLIF", SleepingLIF)
    monkeypatch.setitem(_REGISTRY, "CrashingLIF", CrashingLIF)
    instance = WarmPool()
    monkeypatch.setattr("studio.api.probe.get_pool", lambda: instance)
    yield instance
    instance.close()


def request(client, kind="LIF"):
    return client.post(
        "/api/probe", json={"model": {"kind": kind, "params": PARAMS}, "current": 30.0, "steps": 1}
    )


@pytest.mark.parametrize(
    ("kind", "status", "code"),
    [("SleepingLIF", 504, "probe_timeout"), ("CrashingLIF", 500, "probe_failed")],
)
def test_worker_failure_mapping_and_recovery(pool, kind, status, code):
    with TestClient(app) as client:
        assert request(client).status_code == 200
        previous = pool._process.pid
        response = request(client, kind)
        assert response.status_code == status, response.text
        assert response.json()["error"]["code"] == code
        assert response.json()["error"]["issues"] == []
        assert pool._process.is_alive()
        assert pool._process.pid != previous
        assert request(client).status_code == 200


def test_full_queue_mapping_and_recovery(pool):
    payload = {
        "model": {"kind": "LIF", "params": PARAMS, "enabled": True},
        "current": 30.0,
        "dt": 1.0,
        "steps": 1,
    }
    pool._lock.acquire()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(pool.call, "probe", payload) for _ in range(8)]
        try:
            deadline = time.monotonic() + 10
            while pool._waiting._value != 0 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert pool._waiting._value == 0
            with TestClient(app) as client:
                response = request(client)
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "busy"
            assert "retry later" in response.json()["error"]["message"]
            assert response.json()["error"]["issues"] == []
        finally:
            pool._lock.release()
        for future in futures:
            assert len(future.result()["v"]) == payload["steps"] + 1
    with TestClient(app) as client:
        assert request(client).status_code == 200


def test_queue_wait_expires_without_interrupting_execution(pool):
    payload = {
        "model": {"kind": "SlowLIF", "params": PARAMS, "enabled": True},
        "current": 30.0,
        "dt": 1.0,
        "steps": 1,
    }
    with TestClient(app) as client:
        assert request(client).status_code == 200
        previous = pool._process.pid
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(pool.call, "probe", payload) for _ in range(4)]
            deadline = time.monotonic() + 10
            while pool._waiting._value != 5 and time.monotonic() < deadline:
                time.sleep(0.001)
            assert pool._waiting._value == 5
            start = time.monotonic()
            response = request(client)
            elapsed = time.monotonic() - start
            print(f"queued request refusal after {elapsed:.3f} s")
            assert elapsed >= 30
            assert response.status_code == 503, response.text
            assert response.json()["error"]["code"] == "busy"
            assert "retry later" in response.json()["error"]["message"]
            assert any(not future.done() for future in futures)
            assert pool._process.pid == previous
            for future in futures:
                assert len(future.result()["v"]) == payload["steps"] + 1
        assert pool._process.pid == previous
        assert request(client).status_code == 200
