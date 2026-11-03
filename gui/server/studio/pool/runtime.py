"""Bounded single-process execution with replacement after process failure."""

from __future__ import annotations

import atexit
import multiprocessing
import threading
import time
from functools import lru_cache
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, cast

import torch

from studio.errors import ApiError
from studio.pool.worker import execute


def _serve(pipe: Connection) -> None:
    torch.set_num_threads(1)
    pipe.send(None)
    while True:
        operation, payload = pipe.recv()
        try:
            result = execute(operation, payload)
            pipe.send(("ok", result))
        except ApiError as error:
            pipe.send(("api", (error.code, error.message, error.status, error.issues)))
        except Exception as error:
            pipe.send(("invalid", str(error)))


class WarmPool:
    """One spawned worker, eight waiting calls, and a wall-clock execution deadline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiting = threading.BoundedSemaphore(8)
        self._process: BaseProcess | None = None
        self._pipe: Connection | None = None
        atexit.register(self.close)

    def close(self) -> None:
        """Release the worker and communication pipe."""
        if self._process is not None:
            if self._process.is_alive():
                self._process.kill()
            self._process.join()
            self._process.close()
            self._process = None
        if self._pipe is not None:
            self._pipe.close()
            self._pipe = None

    def _start(self) -> None:
        context = multiprocessing.get_context("spawn")
        self._pipe, child = context.Pipe()
        self._process = context.Process(target=_serve, args=(child,), daemon=True)
        self._process.start()
        child.close()
        if not self._pipe.poll(10):
            self.close()
            raise ApiError("probe_timeout", "Worker startup timed out", 504)
        self._pipe.recv()

    def call(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one request; restart the process before reporting timeout or crash."""
        if not self._waiting.acquire(blocking=False):
            raise ApiError("busy", "Worker queue is full; retry later", 503)
        acquired = self._lock.acquire(timeout=30)
        self._waiting.release()
        if not acquired:
            raise ApiError("busy", "Worker queue wait exceeded 30 s; retry later", 503)
        try:
            if self._process is None:
                self._start()
            assert self._pipe is not None
            deadline = time.monotonic() + 10
            try:
                self._pipe.send((operation, payload))
                if not self._pipe.poll(max(0, deadline - time.monotonic())):
                    self.close()
                    self._start()
                    raise ApiError("probe_timeout", "Probe exceeded its wall-clock timeout", 504)
                status, result = self._pipe.recv()
            except (EOFError, BrokenPipeError, OSError):
                self.close()
                self._start()
                raise ApiError(
                    "probe_failed", "Worker crashed and has been restarted", 500
                ) from None
            if status == "api":
                raise ApiError(*result)
            if status == "invalid":
                return {
                    "valid": False,
                    "errors": [
                        {
                            "path": "/",
                            "message": result,
                            "fix": "Correct the spec to satisfy the library constraint.",
                        }
                    ],
                }
            return cast(dict[str, Any], result)
        finally:
            self._lock.release()


@lru_cache(maxsize=1)
def get_pool() -> WarmPool:
    """Reuse the worker across requests."""
    return WarmPool()
