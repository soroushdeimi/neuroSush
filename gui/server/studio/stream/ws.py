"""API-process relay and per-connection display backpressure."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import secrets
import struct
import time
from collections import deque
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from studio.errors import ApiError
from studio.runner.manager import TERMINAL, RunManager
from studio.stream.ring import Ring, materialize


def authorized(websocket: WebSocket, manager: RunManager) -> bool:
    """Require the configured origin policy and optional query token."""
    config = manager.config
    if config.auth_token and not secrets.compare_digest(
        websocket.query_params.get("token", "").encode(), config.auth_token.encode()
    ):
        return False
    origin = websocket.headers.get("origin")
    if origin is None:
        return False
    allowed = {s.strip() for s in config.allowed_origins.split(",") if s.strip()}
    if allowed != {"*"}:
        return origin in allowed
    scheme = "https" if websocket.url.scheme == "wss" else "http"
    return origin == f"{scheme}://{websocket.headers.get('host')}"


class DisplayQueue:
    """Retain at most eight pending frames and eight outstanding acknowledgements."""

    def __init__(self) -> None:
        self.pending: deque[bytes] = deque()
        self.unacked: deque[int] = deque()
        self.dropped = 0
        self.from_seq = 0

    def push(self, frames: list[bytes], seq: int) -> None:
        """Discard the oldest unsent display data when the window fills."""
        for frame in frames:
            if len(self.pending) == 8:
                self.pending.popleft()
                self.drop(1, seq)
            self.pending.append(frame)

    def drop(self, count: int, seq: int) -> None:
        """Accumulate one loss event with its first affected sequence."""
        if count and not self.dropped:
            self.from_seq = seq
        self.dropped += count

    def ack(self, seq: int) -> None:
        """Release binary frames through a cumulative acknowledgement."""
        while self.unacked and self.unacked[0] <= seq:
            self.unacked.popleft()


def control(manager: RunManager, run_id: str, message: dict[str, Any]) -> None:
    """Validate socket controls before forwarding them to the run process."""
    kind = message.get("type")
    if kind == "pace":
        speed = message.get("speed")
        if (
            message.get("mode") not in {"fast", "watch"}
            or not isinstance(speed, (int, float))
            or isinstance(speed, bool)
            or not math.isfinite(speed)
            or speed <= 0
        ):
            raise ValueError("Expected watch/fast mode and finite positive speed")
        payload = {"type": kind, "mode": message["mode"], "speed": speed}
    elif kind == "select":
        group, indices = message.get("group"), message.get("indices")
        request = manager.requests[run_id]
        spec = request["spec"]
        groups = dict(spec.get("groups", {}))
        for column_name, column in spec.get("columns", {}).items():
            for layer_name, layer in column["layers"].items():
                for name, value in layer["groups"].items():
                    groups[f"{column_name}.{layer_name}.{name}"] = value
        if not isinstance(group, str) or group not in groups:
            raise ValueError("Unknown voltage group")
        shape = groups[group]["shape"]
        size = shape if isinstance(shape, int) else math.prod(shape)
        if (
            not isinstance(indices, list)
            or len(indices) > 64
            or any(type(i) is not int or not 0 <= i < size for i in indices)
        ):
            raise ValueError("Expected at most 64 valid neuron indices")
        behaviors = groups[group].get("behaviors", [])
        if not any(
            b["kind"] in {"LIF", "ELIF", "AdaptiveELIF", "Izhikevich", "HodgkinHuxley"}
            for b in behaviors
        ):
            raise ValueError("Selected group has no voltage model")
        payload = {"type": kind, "group": group, "indices": indices}
    else:
        raise ValueError("Unknown client message")
    with manager.lock:
        if run_id not in manager.children or manager.get(run_id)["status"] in TERMINAL:
            raise ValueError("Run is not active")
        manager.children[run_id][1].send(payload)


async def relay(websocket: WebSocket, run_id: str, manager: RunManager) -> None:
    """Send hello first, replay metadata, and relay only live display history."""
    await websocket.accept()
    if not authorized(websocket, manager):
        await websocket.close(code=1008, reason="Origin or token rejected")
        return
    try:
        info = manager.get(run_id)
    except ApiError:
        await websocket.send_json(
            {
                "type": "error",
                "seq": 1,
                "error": {"code": "not_found", "message": "Run not found", "issues": []},
            }
        )
        await websocket.close(code=1008, reason="Run not found")
        return
    joined_before_start = info["status"] in {"queued", "building"}
    seq = 0

    async def event(kind: str, **body: Any) -> None:
        nonlocal seq
        seq += 1
        await websocket.send_json({"type": kind, "seq": seq, **body})

    await event("hello", protocol=1, run=info)
    root = manager.config.data_dir / "runs" / run_id
    stream_root = root.parent / ".streams" / root.name
    queue = DisplayQueue()
    ring = None
    started = False
    previous = ""
    log_offset = (root / "log.jsonl").stat().st_size
    progress_at = 0.0
    binary_at = 0.0
    previous_step, previous_at = info["step"], time.monotonic()
    receiver = asyncio.create_task(websocket.receive_json())
    try:
        while True:
            info = manager.get(run_id)
            metadata = stream_root / "stream-started.json"
            if not started and not metadata.exists() and info["status"] not in TERMINAL:
                await asyncio.sleep(0.01)
                continue
            if not started and metadata.exists():
                await event("started", **json.loads(metadata.read_text()))
                started = True
                for issue in json.loads((stream_root / "stream-issues.json").read_text()):
                    await event("issue", issue=issue)
                try:
                    ring = Ring(stream_root / "stream.ring")
                except FileNotFoundError:
                    ring = None
                if ring is not None:
                    ring.cursor = 0 if joined_before_start else ring.head
            transitions = []
            with (root / "log.jsonl").open() as log:
                log.seek(log_offset)
                for line in log:
                    if not line.endswith("\n"):
                        break
                    transitions.append(json.loads(line))
                    log_offset += len(line.encode())
            info = manager.get(run_id)
            transitions.append(info)
            for transition in transitions:
                if transition["status"] != previous:
                    await event("status", status=transition["status"], step=transition["step"])
                    await event(
                        "log",
                        level="error" if transition["status"] == "failed" else "info",
                        message=f"Run {transition['status']}",
                    )
                    previous = transition["status"]
            if receiver.done():
                try:
                    message = receiver.result()
                    if not isinstance(message, dict):
                        raise ValueError("Expected a JSON object")
                    if message.get("type") == "ack":
                        ack = message.get("seq")
                        if type(ack) is not int or not 0 <= ack <= seq:
                            raise ValueError("Expected an acknowledgement of a sent sequence")
                        queue.ack(ack)
                    else:
                        control(manager, run_id, message)
                        if message["type"] == "pace":
                            with manager.lock:
                                manager.requests[run_id]["mode"] = message["mode"]
                except (ValueError, KeyError, TypeError) as exc:
                    await event(
                        "issue",
                        issue={
                            "path": "/stream",
                            "message": str(exc),
                            "fix": "Send a valid client message",
                        },
                    )
                receiver = asyncio.create_task(websocket.receive_json())
            if ring is not None:
                frames, lost = ring.read()
                queue.drop(lost, seq + 1)
                queue.push(frames, seq + 1)
            current = time.monotonic()
            fast = manager.requests[run_id]["mode"] == "fast"
            while queue.pending and len(queue.unacked) < 8:
                if fast and current - binary_at < 1 / 30:
                    break
                frame = bytearray(materialize(queue.pending.popleft()))
                seq += 1
                struct.pack_into("<I", frame, 4, seq)
                if fast:
                    frame[1] |= 1
                await websocket.send_bytes(bytes(frame))
                queue.unacked.append(seq)
                binary_at = time.monotonic()
            if queue.dropped:
                await event("dropped", frames=queue.dropped, from_seq=queue.from_seq)
                queue.dropped = 0
            if current - progress_at >= 0.25:
                await event(
                    "progress",
                    step=info["step"],
                    steps=info["steps"],
                    steps_per_s=(info["step"] - previous_step) / max(current - previous_at, 1e-9),
                )
                previous_step, previous_at, progress_at = info["step"], current, current
            if info["status"] in TERMINAL:
                if queue.pending:
                    await event("dropped", frames=len(queue.pending), from_seq=seq + 1)
                await event(
                    "finished",
                    status=info["status"],
                    step=info["step"],
                    error=info["error"],
                    artifacts=sorted(
                        p.name
                        for p in root.iterdir()
                        if p.name
                        in {
                            "checkpoint.pt",
                            "traces.json.gz",
                            "provenance.json",
                            "log.jsonl",
                            "spec.json",
                        }
                    ),
                )
                await websocket.close(code=1000)
                return
            await asyncio.wait({receiver}, timeout=0.01)
    except (WebSocketDisconnect, RuntimeError, OSError):
        return
    finally:
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
            await receiver
        if ring is not None:
            ring.close()
