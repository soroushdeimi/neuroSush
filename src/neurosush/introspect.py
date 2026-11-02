"""Describe registered constructors without instantiating behaviors."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from neurosush.core.behavior import Behavior
from neurosush.structure.spec import _REGISTRY

_OPTIONAL = object()


def _type(annotation: Any) -> dict[str, Any]:
    primitives = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
        type(None): "null",
    }
    if annotation in primitives:
        return {"type": primitives[annotation]}
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        variants = [_type(a) for a in args]
        usable = [v for v in variants if v.get("type") != "unsupported"]
        return {"anyOf": usable, "unsupported": [v for v in variants if v not in usable]}
    if origin is Literal:
        return {"enum": list(args)}
    if origin in (list, tuple):
        schema: dict[str, Any] = {
            "type": "array",
            "items": [_type(a) for a in args if a is not Ellipsis],
        }
        if origin is tuple and Ellipsis not in args:
            schema.update(minItems=len(args), maxItems=len(args))
        return schema
    return {"type": "unsupported", "python": str(annotation)}


def _docs(cls: type[Behavior]) -> dict[str, str]:
    result: dict[str, str] = {}
    for base in reversed(cls.__mro__):
        in_args = False
        current = ""
        for line in (inspect.getdoc(base) or "").splitlines():
            if line == "Args:":
                in_args = True
                continue
            if not in_args:
                continue
            stripped = line.strip()
            if line.startswith("    ") and ":" in stripped:
                current, text = stripped.split(":", 1)
                result[current] = text.strip()
            elif current and line.startswith("        "):
                result[current] += " " + stripped
            elif stripped:
                in_args = False
    return result


def _parameters(cls: type[Behavior]) -> list[tuple[str, Any, Any, inspect._ParameterKind]]:
    init = cls.__init__
    if init is object.__init__:
        return []
    # Resolve annotations without Python 3.10's implicit Optional for None defaults.
    hints = get_type_hints(
        types.SimpleNamespace(__annotations__=init.__annotations__),
        globalns=init.__globals__,
        include_extras=True,
    )
    result: list[tuple[str, Any, Any, inspect._ParameterKind]] = []
    for name, parameter in inspect.signature(init).parameters.items():
        if name == "self":
            continue
        annotation = hints.get(name)
        if parameter.kind is inspect.Parameter.VAR_KEYWORD and "Unpack" in str(
            get_origin(annotation)
        ):
            typed = get_args(annotation)[0]
            parent = next(
                (
                    base
                    for base in cls.__mro__[1:]
                    if issubclass(base, Behavior) and base.__init__ is not init
                ),
                None,
            )
            inherited = {p[0]: p for p in _parameters(parent)} if parent else {}
            for key, hint in get_type_hints(typed, include_extras=True).items():
                required = key in typed.__required_keys__
                wrapper = get_origin(hint)
                # Explicit wrappers take precedence over older TypedDict metadata.
                wrapper_name = getattr(wrapper, "_name", None)
                if wrapper_name in ("Required", "NotRequired"):
                    required = wrapper_name == "Required"
                    hint = get_args(hint)[0]
                default = (
                    inherited[key][2]
                    if key in inherited
                    else (inspect.Parameter.empty if required else _OPTIONAL)
                )
                result.append((key, hint, default, inspect.Parameter.KEYWORD_ONLY))
        else:
            result.append((name, annotation, parameter.default, parameter.kind))
    return result


def describe(cls: type[Behavior]) -> dict[str, Any]:
    """Return JSON parameters, inherited defaults and unsupported argument reasons."""
    params: list[dict[str, Any]] = []
    limitations: list[str] = []
    usable = True
    docs = _docs(cls)
    try:
        parameters = _parameters(cls)
    except (NameError, TypeError) as error:
        parameters = []
        limitations.append(f"{cls.__name__}: unresolved annotation {error}; fix its type hints")
        usable = False
    for name, annotation, default, kind in parameters:
        schema = _type(annotation)
        unsupported = schema.pop("unsupported", [])
        if "anyOf" in schema and len(schema["anyOf"]) == 1:
            schema = schema["anyOf"][0]
        required = default is inspect.Parameter.empty
        if unsupported or schema.get("type") == "unsupported":
            limitations.append(f"{name}: {annotation} needs a JSON literal adapter")
        if kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            limitations.append(
                f"{name}: {kind.name} cannot be built with keyword JSON; fix signature"
            )
            usable = False
        if required and (schema.get("type") == "unsupported" or schema.get("anyOf") == []):
            usable = False
        entry: dict[str, Any] = {
            "name": name,
            "type": schema,
            "required": required,
            "doc": docs.get(name, ""),
        }
        if not required and default is not _OPTIONAL:
            try:
                json.dumps(default, allow_nan=False)
                entry["default"] = default
            except (TypeError, ValueError):
                limitations.append(
                    f"{name}: default {default!r} is not finite JSON; omit this value"
                )
        params.append(entry)
    doc = inspect.getdoc(cls) or ""
    order = getattr(cls, "order", None)
    return {
        "name": cls.__name__,
        "module": cls.__module__,
        "summary": doc.split("\n")[0],
        "docstring": doc,
        "params": params,
        "limitations": limitations,
        "usable": usable,
        "order": int(order) if order is not None else None,
        "lane": getattr(order, "name", None),
        "reads": list(cls.reads),
        "writes": list(cls.writes),
        "graph_safe": cls.graph_safe,
        "independent_ok": cls.independent_ok,
        "supported": list(getattr(cls, "supported", ())),
    }


def catalog() -> dict[str, Any]:
    """Describe every entry in the existing registry in stable name order."""
    return {"version": 1, "behaviors": [describe(_REGISTRY[name]) for name in sorted(_REGISTRY)]}


def fingerprint() -> str:
    """Hash the canonical catalog for drift checks."""
    return hashlib.sha256(json.dumps(catalog(), sort_keys=True).encode()).hexdigest()


def main() -> int:
    """List registrations that cannot currently accept JSON constructor arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = catalog()
    if args.check:
        unavailable = [b for b in result["behaviors"] if not b["usable"]]
        for behavior in unavailable:
            print(f"{behavior['name']}: {'; '.join(behavior['limitations'])}")
        return int(bool(unavailable))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
