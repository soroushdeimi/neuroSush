"""FIFO scheduling and lifecycle supervision for spawned runs."""

from __future__ import annotations

import atexit
import copy
import gzip
import hashlib
import json
import multiprocessing as mp
import secrets
import threading
import time
from collections import deque
from datetime import UTC, datetime
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

import torch

import neurosush
from neurosush.introspect import fingerprint
from neurosush.structure.network_spec import NetworkSpec, dump_spec, load_spec
from studio.config import Config
from studio.errors import ApiError
from studio.runner.resources import preflight
from studio.runner.worker import execute
from studio.stream.ring import remove_ring

TERMINAL = {"completed", "failed", "cancelled"}


def now() -> str:
    """Return a contract UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class RunManager:
    """Own child processes, serialize controls, and expose consistent snapshots."""

    def __init__(self, config: Config) -> None:
        if config.max_jobs < 1:
            raise ValueError("NS_MAX_JOBS must be positive")
        self.config = config
        self.lock = threading.RLock()
        self.runs: dict[str, dict[str, Any]] = {}
        self.requests: dict[str, dict[str, Any]] = {}
        self.estimates: dict[str, int] = {}
        self.queue: deque[str] = deque()
        self.children: dict[str, tuple[BaseProcess, Connection]] = {}
        self.cancelling: dict[str, float] = {}
        self.pending: dict[str, str] = {}
        self.closed = False
        self.thread = threading.Thread(target=self._supervise, daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def _root(self, run_id: str) -> Path:
        return self.config.data_dir / "runs" / run_id

    def _get(self, run_id: str) -> dict[str, Any]:
        if run_id not in self.runs:
            raise ApiError("not_found", "Run not found", 404)
        return self.runs[run_id]

    def get(self, run_id: str) -> dict[str, Any]:
        """Return an independent snapshot."""
        with self.lock:
            return copy.deepcopy(self._get(run_id))

    def list(self) -> list[dict[str, Any]]:
        """List runs in submission order."""
        with self.lock:
            return copy.deepcopy(list(self.runs.values()))

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        """Validate and persist a request before making it schedulable."""
        if request["steps"] > self.config.max_steps:
            raise ApiError("too_many_steps", f"Maximum steps is {self.config.max_steps}", 422)
        try:
            spec = load_spec(json.dumps(request["spec"], allow_nan=False))
            if not isinstance(spec, NetworkSpec):
                raise ValueError("/spec/version: expected version 2")
            canonical = dump_spec(spec)
            estimate = preflight(json.loads(canonical))
        except (ValueError, TypeError, KeyError) as exc:
            raise ApiError("invalid_spec", str(exc), 422) from exc
        provenance = {
            "library": neurosush.__version__,
            "torch": torch.__version__,
            "device": spec.network.device,
            "dtype": spec.network.dtype,
            "seed": spec.network.seed,
            "catalog_fingerprint": fingerprint(),
            "spec_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        }
        run_id = "r_" + secrets.token_hex(8)
        info = {
            "id": run_id,
            "project_id": request.get("project_id"),
            "status": "queued",
            "step": 0,
            "steps": request["steps"],
            "created_at": now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "provenance": provenance,
        }
        with self.lock:
            if self.closed:
                raise ApiError("conflict", "Runner is closed", 409)
            root = self._root(run_id)
            root.mkdir(parents=True)
            (root / "spec.json").write_text(canonical)
            (root / "provenance.json").write_text(json.dumps(provenance, sort_keys=True) + "\n")
            self.runs[run_id] = info
            self.requests[run_id] = copy.deepcopy(request)
            self.estimates[run_id] = estimate
            self.queue.append(run_id)
            self._log(run_id)
            return copy.deepcopy(info)

    def _log(self, run_id: str) -> None:
        with (self._root(run_id) / "log.jsonl").open("a") as output:
            output.write(json.dumps({"time": now(), **self.runs[run_id]}) + "\n")

    def _update(self, run_id: str, update: dict[str, Any]) -> None:
        info = self.runs[run_id]
        if info["status"] in TERMINAL:
            return
        previous = info["status"]
        info.update(update)
        if info["status"] in TERMINAL:
            info["finished_at"] = now()
        if previous != info["status"]:
            self._log(run_id)

    def control(self, run_id: str, action: str) -> dict[str, Any]:
        """Request boundary controls or immediately terminate a cancelled child."""
        with self.lock:
            info = self._get(run_id)
            allowed = {
                "pause": {"running"},
                "resume": {"paused"},
                "cancel": {"queued", "building", "running", "paused"},
            }
            if (
                info["status"] not in allowed[action]
                or run_id in self.cancelling
                or (run_id in self.pending and action != "cancel")
            ):
                raise ApiError("bad_transition", f"Cannot {action} from {info['status']}", 409)
            if action == "cancel":
                if run_id in self.children:
                    self.cancelling[run_id] = time.monotonic()
                    self.children[run_id][0].terminate()
                else:
                    self.queue.remove(run_id)
                    self._update(run_id, {"status": "cancelled"})
            else:
                self.pending[run_id] = action
                self.children[run_id][1].send({"type": action})
            return copy.deepcopy(info)

    def trace(self, run_id: str, target: str) -> dict[str, Any]:
        """Read the lossless recording artifact after completion."""
        with self.lock:
            info = self._get(run_id)
            if info["status"] != "completed":
                raise ApiError("conflict", "Trace is available after completion", 409)
            with gzip.open(self._root(run_id) / "traces.json.gz", "rt") as source:
                traces = json.load(source)["traces"]
            if target not in traces:
                raise ApiError("not_found", "Recording target not found", 404)
            return dict(traces[target])

    def _tick(self) -> None:
        for run_id, (process, pipe) in list(self.children.items()):
            try:
                while pipe.poll():
                    update = pipe.recv()
                    if run_id not in self.cancelling:
                        self._update(run_id, update)
                        if "status" in update:
                            self.pending.pop(run_id, None)
            except (EOFError, OSError):
                pass
            if (
                run_id in self.cancelling
                and time.monotonic() - self.cancelling[run_id] >= 5
                and process.is_alive()
            ):
                process.kill()
            if not process.is_alive():
                process.join()
                remove_ring(self._root(run_id))
                if run_id in self.cancelling:
                    self._update(run_id, {"status": "cancelled"})
                elif self.runs[run_id]["status"] not in TERMINAL:
                    self._update(
                        run_id,
                        {
                            "status": "failed",
                            "error": {
                                "message": f"Child exited with code {process.exitcode}",
                                "traceback": "",
                            },
                        },
                    )
                pipe.close()
                process.close()
                del self.children[run_id]
                self.cancelling.pop(run_id, None)
                self.pending.pop(run_id, None)
        while self.queue and len(self.children) < self.config.max_jobs:
            run_id = self.queue[0]
            reserved = sum(self.estimates[key] for key in self.children)
            try:
                preflight(self.requests[run_id]["spec"], reserved)
            except ApiError:
                if self.children:
                    break
                self.queue.popleft()
                self._update(
                    run_id,
                    {
                        "status": "failed",
                        "error": {
                            "message": "Memory budget changed before launch",
                            "traceback": "",
                        },
                    },
                )
                continue
            self.queue.popleft()
            parent, child = mp.get_context("spawn").Pipe()
            process = mp.get_context("spawn").Process(
                target=execute, args=(self.requests[run_id], str(self._root(run_id)), child)
            )
            self._update(run_id, {"status": "building", "started_at": now()})
            try:
                process.start()
            except Exception as exc:
                parent.close()
                self._update(
                    run_id, {"status": "failed", "error": {"message": str(exc), "traceback": ""}}
                )
            else:
                self.children[run_id] = (process, parent)
            finally:
                child.close()

    def _supervise(self) -> None:
        while True:
            with self.lock:
                if self.closed:
                    return
                self._tick()
            time.sleep(0.01)

    def close(self) -> None:
        """Reap every child before releasing the manager."""
        with self.lock:
            if self.closed:
                return
            for run_id, (process, _) in self.children.items():
                if process.is_alive():
                    process.terminate()
                self.cancelling[run_id] = time.monotonic()
            for run_id in self.queue:
                self._update(run_id, {"status": "cancelled"})
            self.queue.clear()
        deadline = time.monotonic() + 6
        while self.children and time.monotonic() < deadline:
            time.sleep(0.02)
        with self.lock:
            self.closed = True
        self.thread.join(timeout=1)
        atexit.unregister(self.close)
