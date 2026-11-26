"""Real simulation, cross-process transport, and socket acceptance tests."""

from __future__ import annotations

import fcntl
import json
import statistics
import time
from itertools import pairwise

import pytest
import torch
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from studio.api.runs import get_manager
from studio.app import app
from studio.config import Config
from studio.runner import RunManager
from studio.stream.frames import decode_frame as decode_wire_frame
from studio.stream.ring import Ring, materialize
from studio.stream.tap import StreamTap, attach
from studio.stream.ws import DisplayQueue

from neurosush.structure.network_spec import build_network, load_spec


def decode_frame(frame):
    return decode_wire_frame(materialize(frame))


def request(*, steps=2000, size=10, mode="watch", speed=1):
    return {
        "spec": {
            "version": 2,
            "network": {"seed": 7},
            "groups": {
                "exc": {
                    "shape": size,
                    "behaviors": [
                        {
                            "kind": "LIF",
                            "params": {"tau": 10, "threshold": -50, "v_reset": -70, "v_rest": -40},
                        },
                        {"kind": "Fire"},
                    ],
                }
            },
            "recordings": [{"target": "exc", "attributes": ["spikes"], "max_steps": steps}],
        },
        "steps": steps,
        "chunk": 100,
        "mode": mode,
        "speed": speed,
        "stream": {"spikes": ["exc"]},
    }


def built(body):
    return build_network(load_spec(json.dumps(body["spec"])))


def wait(manager, run_id, statuses, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = manager.get(run_id)
        if info["status"] in statuses:
            return info
        assert info["status"] != "failed", info
        time.sleep(0.01)
    pytest.fail(f"Timed out: {manager.get(run_id)}")


@pytest.fixture
def manager(tmp_path):
    instance = RunManager(Config(data_dir=tmp_path, max_jobs=1, max_steps=100000))
    yield instance
    instance.close()


@pytest.fixture
def client(manager):
    app.dependency_overrides[get_manager] = lambda: manager
    with TestClient(app) as session:
        yield session
    app.dependency_overrides.clear()


def connect(client, run_id, **kwargs):
    return client.websocket_connect(
        f"/api/runs/{run_id}/stream", headers={"origin": "http://testserver"}, **kwargs
    )


def test_spikes_equal_recorded_1000_neurons_10000_steps(tmp_path):
    body = request(steps=10000, size=1000, mode="fast")
    simulation = built(body)
    root = tmp_path / "runs" / "r_test"
    root.mkdir(parents=True)
    tap = attach(simulation, body, root)
    try:
        simulation.net.run(body["steps"])
        tap.flush(last=True)
        frames, lost = tap.ring.read()
        decoded = [decode_frame(frame) for frame in frames]
        streamed = sum(sum(byte.bit_count() for byte in f["payload"]) for f in decoded)
        recorded = int(simulation.recorders["exc"].trace().series["spikes"].sum())
        assert lost == 0
        assert streamed == recorded > 0
        assert sum(f["nsteps"] for f in decoded) == 10000
        assert decoded[0]["step0"] == 1
        assert decoded[-1]["flags"] & 2
        print(f"SPIKES streamed={streamed} recorded={recorded} steps=10000 neurons=1000")
    finally:
        tap.ring.close()


def test_tap_overhead(tmp_path):
    body = request(steps=10100, size=1000, mode="fast")
    baseline, tapped = [], []
    for repetition in range(7):
        for enabled in [False, True] if repetition % 2 else [True, False]:
            simulation = built(body)
            ring = Ring(tmp_path / "timing.ring", create=True)
            try:
                tap = StreamTap(simulation, body["stream"], ring)
                if enabled:
                    simulation.net.schedule.append((simulation.net, tap))
                simulation.net.run(100)
                start = time.perf_counter()
                simulation.net.run(10000)
                if enabled:
                    tap.flush(last=True)
                elapsed = time.perf_counter() - start
                (tapped if enabled else baseline).append(elapsed)
            finally:
                ring.close()
    plain, live = statistics.median(baseline), statistics.median(tapped)
    overhead = (live / plain - 1) * 100
    print(f"TIMING baseline={plain:.6f}s tap={live:.6f}s overhead={overhead:.3f}% n=7")
    assert overhead <= 10


def test_ring_wrap_and_nonblocking_contention(tmp_path):
    path = tmp_path / "ring"
    writer = Ring(path, create=True, slots=2, slot_size=64)
    reader = Ring(path, slots=2, slot_size=64)
    try:
        for value in [b"one", b"two", b"three"]:
            assert writer.write(value)
        assert reader.read() == ([b"two", b"three"], 1)
        assert reader.read() == ([], 0)
        fcntl.flock(reader.file, fcntl.LOCK_SH)
        start = time.perf_counter()
        assert writer.write(b"independent")
        elapsed = time.perf_counter() - start
        fcntl.flock(reader.file, fcntl.LOCK_UN)
        assert reader.read() == ([b"independent"], 0)
        print(f"CONTENTION writer returned in {elapsed:.6f}s")
        with pytest.raises(ValueError, match="slot size"):
            writer.write(b"x" * 49)
    finally:
        writer.close()
        reader.close()


def test_display_queue_enforces_window_and_drops_oldest():
    queue = DisplayQueue()
    queue.unacked.extend(range(1, 9))
    queue.push([bytes([i]) for i in range(12)], 9)
    assert list(queue.pending) == [bytes([i]) for i in range(4, 12)]
    assert queue.dropped == 4
    assert queue.from_seq == 9
    queue.ack(4)
    assert list(queue.unacked) == [5, 6, 7, 8]


def test_late_joiner_hello_started_finished(client, manager):
    run_id = manager.submit(request(steps=100, mode="fast"))["id"]
    wait(manager, run_id, {"completed"})
    with connect(client, run_id) as socket:
        hello, started = socket.receive_json(), socket.receive_json()
        assert hello["type"] == "hello"
        assert hello["protocol"] == 1
        assert started["type"] == "started"
        assert started["groups"][0]["size"] == 10
        assert started["schedule"][-1]["order"] == StreamTap.order
        events = [hello, started]
        while events[-1]["type"] != "finished":
            events.append(socket.receive_json())
        assert events[-1]["status"] == "completed"
        assert "traces.json.gz" in events[-1]["artifacts"]
        assert [e["seq"] for e in events] == sorted({e["seq"] for e in events})
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == 1000


def test_slow_client_dropped_and_run_completes(client, manager):
    run_id = manager.submit(request(steps=2200, speed=2))["id"]
    with connect(client, run_id) as socket:
        assert socket.receive_json()["type"] == "hello"
        time.sleep(2)
        info = wait(manager, run_id, {"completed"})
        binary, dropped, seqs = 0, 0, []
        while True:
            message = socket.receive()
            if message.get("bytes") is not None:
                frame = decode_frame(message["bytes"])
                binary += 1
                seqs.append(frame["seq"])
            elif message.get("text") is not None:
                event = json.loads(message["text"])
                seqs.append(event["seq"])
                if event["type"] == "dropped":
                    dropped += event["frames"]
                if event["type"] == "finished":
                    break
        assert binary == 8
        assert dropped > 0
        assert seqs == sorted(set(seqs))
        assert info["step"] == 2200
        print(
            f"SLOW_CLIENT binary_unacked={binary} dropped={dropped} completed_step={info['step']}"
        )


def test_unknown_run_closes_without_hang(client):
    with connect(client, "r_missing") as socket:
        event = socket.receive_json()
        assert event["error"] == {"code": "not_found", "message": "Run not found", "issues": []}
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == 1008


@pytest.mark.parametrize(
    ("origin", "token", "allowed"),
    [
        ("http://evil.test", "secret", "http://testserver"),
        ("http://testserver", "wrong", "http://testserver"),
        ("http://testserver", "غلط", "http://testserver"),
        ("http://testserver", None, "http://testserver"),
        (None, "secret", "*"),
        ("http://evil.test", "secret", "*"),
    ],
)
def test_origin_and_token_rejected(tmp_path, origin, token, allowed):
    manager = RunManager(Config(data_dir=tmp_path, auth_token="secret", allowed_origins=allowed))
    app.dependency_overrides[get_manager] = lambda: manager
    try:
        with TestClient(app) as client:
            headers = {"origin": origin} if origin else {}
            path = "/api/runs/r_missing/stream" + (f"?token={token}" if token else "")
            with client.websocket_connect(path, headers=headers) as socket:
                with pytest.raises(WebSocketDisconnect) as closed:
                    socket.receive_json()
                assert closed.value.code == 1008
                assert "Origin or token" in closed.value.reason
    finally:
        manager.close()
        app.dependency_overrides.clear()


def test_configured_origin_and_correct_token_accepted(tmp_path):
    manager = RunManager(
        Config(
            data_dir=tmp_path,
            auth_token="secret",
            allowed_origins="https://studio.test,http://elsewhere.test",
        )
    )
    app.dependency_overrides[get_manager] = lambda: manager
    try:
        with (
            TestClient(app) as client,
            client.websocket_connect(
                "/api/runs/r_missing/stream?token=secret", headers={"origin": "https://studio.test"}
            ) as socket,
        ):
            assert socket.receive_json()["error"]["code"] == "not_found"
    finally:
        manager.close()
        app.dependency_overrides.clear()


def test_socket_spikes_equal_trace_with_ack(client, manager):
    from neurosush.viz.trace import unpack_array

    run_id = manager.submit(request(steps=10000, size=1000, speed=5))["id"]
    streamed, steps = 0, 0
    with connect(client, run_id) as socket:
        assert socket.receive_json()["type"] == "hello"
        started = socket.receive_json()
        assert started["type"] == "started"
        last_seq = started["seq"]
        while True:
            message = socket.receive()
            if message.get("bytes") is not None:
                frame = decode_frame(message["bytes"])
                assert frame["seq"] > last_seq
                last_seq = frame["seq"]
                streamed += sum(byte.bit_count() for byte in frame["payload"])
                steps += frame["nsteps"]
                socket.send_json({"type": "ack", "seq": last_seq})
            else:
                event = json.loads(message["text"])
                assert event["seq"] > last_seq
                last_seq = event["seq"]
                assert event["type"] != "dropped", event
                if event["type"] == "finished":
                    assert event["status"] == "completed"
                    break
    recorded = int(unpack_array(manager.trace(run_id, "exc")["series"]["spikes"]).sum())
    assert steps == 10000
    assert streamed == recorded > 0
    print(f"SOCKET_SPIKES streamed={streamed} recorded={recorded} steps={steps}")


def test_vsample_scalar_batch_and_selection(tmp_path):
    body = request(steps=5)
    body["spec"]["network"]["batch_size"] = 2
    simulation = built(body)
    ring = Ring(tmp_path / "tap.ring", create=True)
    try:
        tap = StreamTap(
            simulation, {"spikes": "all", "vsample": {"exc": [0, 9]}, "scalars": ["rate:exc"]}, ring
        )
        simulation.net.schedule.append((simulation.net, tap))
        simulation.net.run(2)
        tap.select("exc", [1])
        simulation.net.run(3)
        tap.flush(last=True)
        frames, lost = ring.read()
        assert lost == 0
        decoded = [decode_frame(f) for f in frames]
        voltage = [f for f in decoded if f["type"] == 2]
        assert voltage[0]["indices"] == [0, 9]
        assert voltage[0]["nsteps"] == 2
        assert voltage[1]["indices"] == [1]
        assert voltage[1]["nsteps"] == 3
        torch.testing.assert_close(
            torch.tensor(voltage[1]["values"][-1]), simulation.groups["exc"].v[0, [1]]
        )
        scalar = [f for f in decoded if f["type"] == 3]
        assert scalar[-1]["values"][-1] == simulation.groups["exc"].spikes[0].float().mean()
        assert sum(f["nsteps"] for f in decoded if f["type"] == 1) == 5
        for indices in ([-1], [10], [True], list(range(65))):
            with pytest.raises(ValueError, match="indices"):
                tap.select("exc", indices)
    finally:
        ring.close()


def test_weight_block_means_and_interval(tmp_path):
    body = request(steps=2, size=257)
    body["spec"]["synapses"] = [{"name": "self", "src": "exc", "dst": "exc"}]
    simulation = built(body)
    weights = torch.arange(257 * 257, dtype=torch.float32).reshape(257, 257)
    simulation.synapses["self"].weights = weights
    ring = Ring(tmp_path / "weights.ring", create=True)
    try:
        tap = StreamTap(simulation, {"weights": {"self": 2}}, ring)
        simulation.net.schedule.append((simulation.net, tap))
        simulation.net.run(2)
        frames, lost = ring.read()
        assert lost == 0
        assert len(frames) == 1
        frame = decode_frame(frames[0])
        assert frame["type"] == 4
        assert frame["step0"] == 2
        assert (frame["rows"], frame["cols"], frame["row_stride"], frame["col_stride"]) == (
            129,
            129,
            2,
            2,
        )
        assert frame["flags"] & 1
        assert frame["weights"][0] == weights[:2, :2].mean()
        assert frame["weights"][-1] == weights[-1, -1]
    finally:
        ring.close()


def test_client_pace_select_and_invalid_messages(client, manager):
    run_id = manager.submit(request(steps=1000, speed=1))["id"]
    wait(manager, run_id, {"running"})
    with connect(client, run_id) as socket:
        assert socket.receive_json()["type"] == "hello"
        assert socket.receive_json()["type"] == "started"
        socket.send_json({"type": "select", "group": "exc", "indices": [2, 3]})
        socket.send_json({"type": "pace", "mode": "watch", "speed": 5})
        socket.send_json({"type": "select", "group": "exc", "indices": [10]})
        socket.send_json({"type": "ack", "seq": 1000000})
        issues, voltages = 0, []
        while True:
            message = socket.receive()
            if message.get("bytes") is not None:
                frame = decode_frame(message["bytes"])
                socket.send_json({"type": "ack", "seq": frame["seq"]})
                if frame["type"] == 2:
                    voltages.append(frame)
            else:
                event = json.loads(message["text"])
                if event["type"] == "issue":
                    issues += 1
                    assert set(event["issue"]) == {"path", "message", "fix"}
                if event["type"] == "finished":
                    assert event["status"] == "completed"
                    break
        assert issues == 2
        assert voltages
        assert all(frame["indices"] == [2, 3] for frame in voltages)


def ring_producer(path, connection):
    ring = Ring(path, slots=4, slot_size=65536)
    try:
        connection.send("ready")
        connection.recv()
        for i in range(1000):
            ring.write(i.to_bytes(8, "little") + bytes([i % 127]) * 32768)
        connection.send("done")
    finally:
        ring.close()
        connection.close()


def test_spawned_ring_never_returns_torn_slots(tmp_path):
    import multiprocessing as mp

    path = tmp_path / "concurrent.ring"
    ring = Ring(path, create=True, slots=4, slot_size=65536)
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=ring_producer, args=(path, child))
    process.start()
    child.close()
    received, dropped = [], 0
    try:
        assert parent.poll(15)
        assert parent.recv() == "ready"
        parent.send("go")
        deadline = time.monotonic() + 10
        while process.is_alive() and time.monotonic() < deadline:
            frames, lost = ring.read()
            received.extend(frames)
            dropped += lost
        process.join(timeout=1)
        assert process.exitcode == 0
        frames, lost = ring.read()
        received.extend(frames)
        dropped += lost
        assert len(received) + dropped == 1000
        assert received
        for frame in received:
            number = int.from_bytes(frame[:8], "little")
            assert frame[8:] == bytes([number % 127]) * 32768
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()
        parent.close()
        ring.close()


def test_fast_frame_rate_and_failure_finish(client, manager, monkeypatch):
    from fastapi import WebSocket

    arrivals = []
    send_bytes = WebSocket.send_bytes

    async def measured_send(socket, frame):
        if frame[1] & 1:
            arrivals.append(time.monotonic())
        await send_bytes(socket, frame)

    monkeypatch.setattr(WebSocket, "send_bytes", measured_send)
    body = request(steps=100000, speed=1)
    body["stream"]["vsample"] = {"exc": [0]}
    run_id = manager.submit(body)["id"]
    wait(manager, run_id, {"running"})
    with connect(client, run_id) as socket:
        socket.send_json({"type": "pace", "mode": "fast", "speed": 1})
        while True:
            message = socket.receive()
            if message.get("bytes") is not None:
                frame = decode_frame(message["bytes"])
                socket.send_json({"type": "ack", "seq": frame["seq"]})
            elif json.loads(message["text"])["type"] == "finished":
                break
        assert len(arrivals) >= 2
        assert all(b - a >= 1 / 30 for a, b in pairwise(arrivals))
        intervals = [b - a for a, b in pairwise(arrivals)]
        rate = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
        print(
            f"FAST binary_frames={len(arrivals)} "
            f"min_interval={min(intervals):.6f}s rate={rate:.3f}/s"
        )
    bad = request()
    bad["stream"] = {"vsample": {"exc": [1000]}}
    failed = manager.submit(bad)["id"]
    wait(manager, failed, {"failed"})
    with connect(client, failed) as socket:
        assert socket.receive_json()["type"] == "hello"
        while True:
            event = socket.receive_json()
            if event["type"] == "finished":
                assert event["status"] == "failed"
                assert event["error"]["message"]
                break


def test_unrecorded_spikes_copy_before_inplace_changes(tmp_path):
    body = request(steps=2)
    body["spec"]["recordings"] = []
    simulation = built(body)
    ring = Ring(tmp_path / "copy.ring", create=True)
    try:
        tap = StreamTap(simulation, body["stream"], ring)
        group = simulation.groups["exc"]
        simulation.net.iteration = 1
        group.spikes[[0, 9]] = True
        tap.forward(simulation.net)
        group.spikes.zero_()
        simulation.net.iteration = 2
        group.spikes[3] = True
        tap.forward(simulation.net)
        group.spikes.zero_()
        tap.flush(last=True)
        frames, lost = ring.read()
        assert lost == 0
        frame = decode_frame(frames[0])
        assert frame["payload"].hex() == "01020800"
        assert frame["nsteps"] == 2
        assert frame["step0"] == 1
        assert frame["flags"] == 2
    finally:
        ring.close()


def test_ring_cleanup_after_completed_and_cancelled(client, manager):
    for action in ("complete", "cancel"):
        body = request(steps=100 if action == "complete" else 10000)
        run_id = manager.submit(body)["id"]
        wait(manager, run_id, {"running"})
        if action == "cancel":
            manager.control(run_id, "cancel")
        wait(manager, run_id, {"completed" if action == "complete" else "cancelled"})
        deadline = time.monotonic() + 5
        with manager.lock:
            active = run_id in manager.children
        while active and time.monotonic() < deadline:
            time.sleep(0.01)
            with manager.lock:
                active = run_id in manager.children
        assert not active
        root = manager.config.data_dir / "runs" / ".streams" / run_id
        assert not (root / "stream.ring").exists()
        assert (root / "stream-started.json").exists()


def test_dense_weights_stream_sparse_issues_once(client, manager):
    body = request(steps=1000, speed=2)
    body["stream"] = {"weights": {"dense": 100, "sparse": 100}}
    body["spec"]["synapses"] = [
        {
            "name": "dense",
            "src": "exc",
            "dst": "exc",
            "behaviors": [
                {"kind": "WeightInit", "params": {"mode": "ones"}},
            ],
        },
        {
            "name": "sparse",
            "src": "exc",
            "dst": "exc",
            "behaviors": [
                {"kind": "WeightInit", "params": {"mode": "ones", "sparse": True}},
            ],
        },
    ]
    run_id = manager.submit(body)["id"]
    issues, weights = [], []
    with connect(client, run_id) as socket:
        assert socket.receive_json()["type"] == "hello"
        while True:
            message = socket.receive()
            if message.get("bytes") is not None:
                frame = decode_frame(message["bytes"])
                assert frame["type"] == 4
                weights.append(frame)
                socket.send_json({"type": "ack", "seq": frame["seq"]})
            else:
                event = json.loads(message["text"])
                if event["type"] == "started":
                    assert set(event) == {
                        "type",
                        "seq",
                        "dt",
                        "groups",
                        "synapses",
                        "scalars",
                        "vsample",
                        "schedule",
                    }
                elif event["type"] == "issue":
                    issues.append(event["issue"])
                elif event["type"] == "finished":
                    assert event["status"] == "completed"
                    break
    assert issues == [
        {
            "path": "/synapses/sparse",
            "message": "weights not streamable for sparse",
            "fix": "stream is limited to dense synapses in this version",
        }
    ]
    assert len(weights) == 10
    assert weights[-1]["flags"] & 2
    assert all(not frame["flags"] & 2 for frame in weights[:-1])
    assert all(frame["stream"] == 0 for frame in weights)
    assert all(frame["rows"] == 10 and frame["cols"] == 10 for frame in weights)
    assert all(frame["weights"] == [1.0] * 100 for frame in weights)


def test_compact_weights_emit_issue_and_no_frame(tmp_path):
    body = request(steps=2)
    body["spec"]["synapses"] = [{"name": "compact", "src": "exc", "dst": "exc"}]
    simulation = built(body)
    simulation.synapses["compact"].weights = torch.ones(10)
    ring = Ring(tmp_path / "compact.ring", create=True)
    try:
        tap = StreamTap(simulation, {"weights": {"compact": 1}}, ring)
        simulation.net.schedule.append((simulation.net, tap))
        simulation.net.run(2)
        assert ring.read() == ([], 0)
        assert tap.issues == [
            {
                "path": "/synapses/compact",
                "message": "weights not streamable for compact 1-D",
                "fix": "stream is limited to dense synapses in this version",
            }
        ]
        assert all("streamable" not in synapse for synapse in tap.started()["synapses"])
    finally:
        ring.close()
