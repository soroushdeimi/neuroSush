"""Library-backed network preflight for execution inside the worker."""

from __future__ import annotations

import json
from typing import Any

from neurosush.structure.network_spec import NetworkSpec, build_network, load_spec
from studio.runner.resources import preflight


def validate_network(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse, budget and initialize a v2 network, preserving library error pointers."""
    try:
        spec = load_spec(json.dumps(payload, allow_nan=False))
        if not isinstance(spec, NetworkSpec):
            raise ValueError("/version: expected NetworkSpec v2")
        preflight(payload)
        build_network(spec)
    except (ValueError, TypeError, KeyError, RuntimeError, AttributeError) as error:
        message = str(error)
        prefix, separator, detail = message.partition(":")
        path = prefix if separator and prefix.startswith("/") else "/"
        return {
            "valid": False,
            "errors": [
                {
                    "path": path,
                    "message": detail.strip() if path != "/" else message,
                    "fix": "Correct the spec to satisfy the library constraint.",
                }
            ],
        }
    return {"valid": True, "errors": []}
