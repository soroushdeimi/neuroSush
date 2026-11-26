"""Probe timing conventions, parity and warm latency."""

from __future__ import annotations

import math
import time

import pytest
from fastapi.testclient import TestClient
from studio.app import app
from studio.pool.runtime import get_pool

import neurosush
from neurosush.core.network import Network, NeuronGroup
from neurosush.neurons.models import LIF, Fire

PARAMS = {"tau": 10.0, "threshold": -50.0, "v_reset": -65.0, "v_rest": -65.0}
BODY = {"model": {"kind": "LIF", "params": PARAMS}, "current": 30.0, "steps": 20}


@pytest.mark.parametrize(("steps", "dt"), [(1, 0.5), (20, 1.0), (5000, 1.0)])
def test_probe_matches_library_completed_steps(steps, dt):
    request_body = {**BODY, "steps": steps, "dt": dt}
    with TestClient(app) as client:
        response = client.post("/api/probe", json=request_body)
    assert response.status_code == 200, response.text
    data = response.json()
    net = Network(dt=dt)
    group = NeuronGroup(net, 1, [LIF(**PARAMS), Fire()])
    net.initialize()
    group.I.fill_(BODY["current"])
    voltages = [group.v.item()]
    spikes = []
    for _ in range(steps):
        net.step()
        voltages.append(group.v.item())
        if group.spikes.item():
            spikes.append(net.iteration)
    assert len(data["t"]) == len(data["v"]) == steps + 1
    assert data["v"][0] == PARAMS["v_rest"]
    assert data["v"] == voltages
    assert data["t"] == [k * net.dt for k in range(steps + 1)]
    if steps > 1:
        assert spikes
    assert data["spikes"] == spikes
    assert data["provenance"] == {
        "library": neurosush.__version__,
        "kind": "LIF",
        "params": PARAMS,
        "dt": dt,
        "steps": steps,
        "current": BODY["current"],
    }


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"steps": 5001}, "too_many_steps"),
        ({"model": {"kind": "unknown"}}, "invalid_spec"),
        ({"model": {"kind": "LIF", "params": {**PARAMS, "tau": -1}}}, "invalid_spec"),
    ],
)
def test_probe_rejects_invalid(changes, code):
    with TestClient(app) as client:
        response = client.post("/api/probe", json={**BODY, **changes})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code


def test_probe_warm_p95_latency():
    warm_body = {**BODY, "steps": 500}
    with TestClient(app) as client:
        assert client.post("/api/probe", json=warm_body).status_code == 200
        timings = []
        for _ in range(50):
            start = time.perf_counter()
            assert client.post("/api/probe", json=warm_body).status_code == 200
            timings.append((time.perf_counter() - start) * 1000)
    p95 = sorted(timings)[math.ceil(len(timings) * 0.95) - 1]
    print(f"probe warm p95 over {len(timings)} calls, {warm_body['steps']} steps: {p95:.3f} ms")
    assert p95 < 1000
    get_pool().close()
