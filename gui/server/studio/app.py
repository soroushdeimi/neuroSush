"""Serve the schema, spec validation and the built Compose interface."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, ValidationError

from neurosush.introspect import catalog, describe
from neurosush.structure.spec import _REGISTRY, BehaviorSpec, from_json

app = FastAPI(title="neuroSush Studio", version="0.1.0")


class _Data(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class _Behavior(_Data):
    kind: str
    params: dict[str, Any]


class _Group(_Data):
    shape: int | list[int]
    behaviors: list[_Behavior]
    inhibitory: bool
    tags: list[str]


class _Synapse(_Data):
    src: str
    dst: str
    behaviors: list[_Behavior]
    compartment: str


class _Layer(_Data):
    groups: dict[str, _Group]
    synapses: list[_Synapse]
    inputs: dict[str, list[str]] | None
    outputs: dict[str, list[str]] | None


class _Column(_Data):
    layers: dict[str, _Layer]
    synapses: list[_Synapse]
    inputs: dict[str, str]
    outputs: dict[str, str]


def _pointer(parts: tuple[str | int, ...]) -> str:
    return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def _issue(path: str, message: str, fix: str) -> dict[str, str]:
    return {"path": path, "message": message, "fix": fix}


def _matches(value: Any, schema: dict[str, Any]) -> bool:
    if "anyOf" in schema:
        return any(_matches(value, alternative) for alternative in schema["anyOf"])
    if "enum" in schema:
        return value in schema["enum"]
    kind = schema.get("type")
    if kind == "number":
        return type(value) in (int, float)
    if kind == "integer":
        return type(value) is int
    if kind == "boolean":
        return type(value) is bool
    if kind == "null":
        return value is None
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        items = schema["items"]
        return (
            isinstance(value, list)
            and len(value) >= schema.get("minItems", 0)
            and len(value) <= schema.get("maxItems", float("inf"))
            and all(_matches(item, items[min(i, len(items) - 1)]) for i, item in enumerate(value))
        )
    return False


def _behaviors(items: list[_Behavior], path: tuple[str | int, ...]) -> list[dict[str, str]]:
    errors = []
    for index, behavior in enumerate(items):
        location = (*path, "behaviors", index)
        kind, params = behavior.kind, behavior.params
        if kind not in _REGISTRY:
            errors.append(
                _issue(
                    _pointer((*location, "kind")),
                    f"Unknown behavior {kind!r}",
                    "Choose a registered behavior from /api/schema.",
                )
            )
            continue
        fields = {p["name"]: p for p in describe(_REGISTRY[kind])["params"]}
        before = len(errors)
        for name in sorted(params.keys() | fields.keys()):
            field = fields.get(name)
            if field is None or (name not in params and field["required"]):
                errors.append(
                    _issue(
                        _pointer((*location, "params", name)),
                        f"{name} = {params.get(name)!r}: unknown or missing",
                        "Use the named parameters in /api/schema.",
                    )
                )
            elif name in params and not _matches(params[name], field["type"]):
                errors.append(
                    _issue(
                        _pointer((*location, "params", name)),
                        f"{name} = {params[name]!r}: incompatible type",
                        f"Enter a value matching {field['type']}.",
                    )
                )
        if len(errors) != before:
            continue
        try:
            BehaviorSpec(kind, params).build()
        except (ValueError, TypeError) as error:
            message = str(error)
            name = next((n for n in params if n in message), None)
            suffix = ("params", name) if name else ("params",)
            errors.append(
                _issue(
                    _pointer((*location, *suffix)),
                    message,
                    "Correct the named value to satisfy this constraint.",
                )
            )
    return errors


@app.get("/api/health")
def health() -> dict[str, str]:
    """Report process readiness."""
    return {"status": "ok"}


@app.get("/api/schema")
def schema() -> dict[str, Any]:
    """Return descriptors directly from the current library registry."""
    return catalog()


@app.post("/api/validate")
def validate(payload: Annotated[Any, Body()]) -> dict[str, Any]:
    """Parse v1 specs using the library and check constructor arguments without running."""
    try:
        parsed = _Column.model_validate(payload)
    except ValidationError as error:
        return {
            "valid": False,
            "errors": [
                _issue(
                    _pointer(e["loc"]),
                    f"{e['input']!r}: {e['msg']}",
                    "Supply the required v1 ColumnSpec field with the indicated type.",
                )
                for e in error.errors()
            ],
        }
    data = parsed.model_dump()
    try:
        from_json(json.dumps(data, allow_nan=False))
    except (ValueError, TypeError, KeyError) as error:
        return {
            "valid": False,
            "errors": [_issue("/", str(error), "Correct the v1 spec shape.")],
        }
    errors = []
    for index, synapse in enumerate(parsed.synapses):
        errors.extend(_behaviors(synapse.behaviors, ("synapses", index)))
    for layer_name, layer in parsed.layers.items():
        for index, synapse in enumerate(layer.synapses):
            errors.extend(_behaviors(synapse.behaviors, ("layers", layer_name, "synapses", index)))
        for group_name, group in layer.groups.items():
            errors.extend(_behaviors(group.behaviors, ("layers", layer_name, "groups", group_name)))
            if (isinstance(group.shape, int) and group.shape <= 0) or (
                isinstance(group.shape, list) and any(n <= 0 for n in group.shape)
            ):
                errors.append(
                    _issue(
                        _pointer(("layers", layer_name, "groups", group_name, "shape")),
                        f"shape = {group.shape!r}: dimensions must be positive",
                        "Enter a positive integer or three positive integers.",
                    )
                )
    return {"valid": not errors, "errors": errors}


web_root = Path(os.environ.get("NS_WEB_ROOT", Path(__file__).parents[2] / "web" / "dist"))
if web_root.is_dir():
    app.mount("/", StaticFiles(directory=web_root, html=True), name="web")
