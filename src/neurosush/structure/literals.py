"""Decode explicit, closed-set literals without importing user supplied names."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import torch

from neurosush.core.network import Network
from neurosush.synapses.init import fixed_in_degree, sparse_random
from neurosush.viz.trace import unpack_array

ALLOWED_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sparse_random": sparse_random,
    "fixed_in_degree": fixed_in_degree,
}
AssetResolver = Callable[[str], Any] | Mapping[str, Any]


def pointer(parent: str, key: str | int) -> str:
    """Append an escaped JSON pointer component."""
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def resolve_asset(name: str, assets: AssetResolver | None, path: str) -> Any:
    """Resolve an asset exclusively through the caller's resolver."""
    if assets is None:
        raise ValueError(f"{path}: asset {name!r} needs a resolver")
    try:
        return assets(name) if callable(assets) else assets[name]
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f"{path}: cannot resolve asset {name!r}: {error}") from error


def decode_literals(
    value: Any, net: Network, assets: AssetResolver | None = None, path: str = ""
) -> Any:
    """Decode nested parameter literals using the network's device and generator."""
    if isinstance(value, list):
        return [decode_literals(v, net, assets, pointer(path, i)) for i, v in enumerate(value)]
    if not isinstance(value, dict):
        return value
    markers = set(value) & {"$tensor", "$asset", "$init", "$fn"}
    if not markers:
        return {k: decode_literals(v, net, assets, pointer(path, k)) for k, v in value.items()}
    if len(markers) != 1:
        raise ValueError(f"{path}: conflicting literal markers")
    try:
        if "$asset" in value:
            if set(value) != {"$asset"}:
                raise ValueError("$asset accepts only an asset name")
            return resolve_asset(value["$asset"], assets, path)
        if "$tensor" in value:
            if set(value) != {"$tensor"}:
                raise ValueError("unexpected tensor keys")
            data = value["$tensor"]
            if "asset" in data:
                if set(data) != {"asset", "shape"}:
                    raise ValueError("asset tensor requires asset and shape")
                tensor = torch.as_tensor(resolve_asset(data["asset"], assets, path)).clone()
                if list(tensor.shape) != data["shape"]:
                    raise ValueError("asset tensor shape mismatch")
            else:
                if set(data) != {"dtype", "shape", "data"}:
                    raise ValueError("packed tensor requires dtype, shape and data")
                tensor = unpack_array(data)
            return tensor.to(device=net.device)
        if "$fn" in value:
            name = value["$fn"]
            if name not in ALLOWED_FUNCTIONS or set(value) - {"$fn", "kwargs"}:
                raise ValueError(f"unknown or invalid function {name!r}")
            kwargs = decode_literals(value.get("kwargs", {}), net, assets, pointer(path, "kwargs"))
            return partial(ALLOWED_FUNCTIONS[name], **kwargs)
        options = dict(value)
        kind = options.pop("$init")
        shape = options.pop("shape")
        dtype_name = options.pop("dtype", str(net.dtype).removeprefix("torch."))
        if dtype_name not in {"float32", "float64", "float16", "bfloat16"}:
            raise ValueError(f"invalid initializer dtype {dtype_name!r}")
        dtype = getattr(torch, dtype_name)
        if kind == "constant":
            result = torch.full(shape, options.pop("value", 0.0), dtype=dtype, device=net.device)
        elif kind in {"uniform", "normal"}:
            draw = torch.rand if kind == "uniform" else torch.randn
            result = draw(shape, generator=net.generator, dtype=dtype, device=net.device)
            if kind == "uniform":
                low, high = options.pop("low", 0.0), options.pop("high", 1.0)
                if high < low:
                    raise ValueError("high must be at least low")
                result = result * (high - low) + low
            else:
                mean, std = options.pop("mean", 0.0), options.pop("std", 1.0)
                if std < 0:
                    raise ValueError("std must be non-negative")
                result = result * std + mean
        else:
            raise ValueError(f"unknown initializer {kind!r}")
        if options:
            raise ValueError(f"unknown initializer options {sorted(options)}")
        return result
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{path}: {error}") from error
