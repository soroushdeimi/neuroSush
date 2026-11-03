"""Spawn entry point; simulation state stays inside its child process."""

from __future__ import annotations

import gzip
import json
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from neurosush.checkpoint import save
from neurosush.structure.network_spec import NetworkSpec, build_network, load_spec
from neurosush.viz.trace import Trace
from studio.stream.tap import attach


def execute(request: dict[str, Any], directory: str, pipe: Connection) -> None:
    """Run chunks, acknowledge controls at boundaries, and persist complete traces."""
    root = Path(directory)
    tap = None
    try:
        spec = load_spec(json.dumps(request["spec"]))
        if not isinstance(spec, NetworkSpec):
            raise ValueError("Expected NetworkSpec v2")
        built = build_network(spec)
        tap = attach(built, request, root)
        pipe.send({"status": "running", "step": built.net.iteration})
        mode, speed = request["mode"], request["speed"]
        paused = False
        while built.net.iteration < request["steps"]:
            while paused or pipe.poll():
                control = pipe.recv()
                if control["type"] in {"pause", "resume"}:
                    paused = control["type"] == "pause"
                    pipe.send(
                        {"status": "paused" if paused else "running", "step": built.net.iteration}
                    )
                elif control["type"] == "pace":
                    mode, speed = control["mode"], control["speed"]
                elif control["type"] == "select":
                    tap.select(control["group"], control["indices"])
            count = min(request["chunk"], request["steps"] - built.net.iteration)
            start = time.monotonic()
            tap.flags = int(mode == "fast")
            built.net.run(count)
            tap.flush(last=built.net.iteration == request["steps"])
            pipe.send({"step": built.net.iteration})
            if mode == "watch":
                delay = count * built.net.dt / (1000 * speed) - (time.monotonic() - start)
                if delay > 0:
                    pipe.poll(delay)
        provenance = json.loads((root / "provenance.json").read_text())
        traces = {
            target: recorder.trace(
                **Trace.make_meta(
                    built.net,
                    run_id=root.name,
                    spec_sha256=provenance["spec_sha256"],
                    catalog_fingerprint=provenance["catalog_fingerprint"],
                    target=target,
                    attributes=list(recorder.attributes),
                )
            ).to_dict()
            for target, recorder in built.recorders.items()
        }
        with gzip.open(root / "traces.json.gz", "wt", encoding="utf-8") as output:
            json.dump({"format": "neurosush.run-traces/1", "traces": traces}, output)
            output.write("\n")
        save(built.net, root / "checkpoint.pt")
        pipe.send({"status": "completed", "step": built.net.iteration})
    except Exception as exc:
        pipe.send(
            {
                "status": "failed",
                "error": {"message": str(exc), "traceback": traceback.format_exc()},
            }
        )
    finally:
        if tap is not None:
            tap.ring.close()
        pipe.close()
