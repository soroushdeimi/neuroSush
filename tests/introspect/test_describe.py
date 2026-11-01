import json

from neurosush.introspect import catalog, describe, fingerprint
from neurosush.neurons.models import ELIF, LIF, Fire
from neurosush.structure.spec import registered
from neurosush.synapses.plasticity import RSTDP


def test_lif_and_inherited_options():
    lif = describe(LIF)
    params = {p["name"]: p for p in lif["params"]}
    assert params["tau"]["required"] is True
    assert params["resistance"]["default"] == 1.0
    assert params["v_init"]["default"] is None
    assert params["threshold"]["type"]["type"] == "number"
    assert lif["reads"] == ["I"]
    assert describe(Fire)["params"] == []
    elif_params = {p["name"]: p for p in describe(ELIF)["params"]}
    assert set(elif_params) == set(params) | {"delta", "theta_rh"}
    assert elif_params["resistance"]["default"] == 1.0
    assert "tau_c" in {p["name"] for p in describe(RSTDP)["params"]}


def test_catalog_complete_and_json_safe():
    result = catalog()
    assert [b["name"] for b in result["behaviors"]] == registered()
    json.dumps(result, allow_nan=False)
    assert len(fingerprint()) == 64
    assert fingerprint() == fingerprint()
    weight = next(b for b in result["behaviors"] if b["name"] == "WeightInit")
    assert any("fn" in reason for reason in weight["limitations"])


def test_literal_optional_arrays_and_unusable_required():
    from typing import Literal

    from neurosush.core.behavior import Behavior

    class Options(Behavior):
        def __init__(
            self,
            mode: Literal["a", "b"],
            count: int | None = None,
            shape: tuple[int, int] = (2, 3),
            callback: object = None,
        ):
            pass

    parameters = {p["name"]: p for p in describe(Options)["params"]}
    assert parameters["mode"]["type"] == {"enum": ["a", "b"]}
    assert parameters["count"]["type"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    assert parameters["shape"]["default"] == (2, 3)
    assert parameters["callback"]["type"]["type"] == "unsupported"


def test_unpack_without_parent_signature_and_positional_only():
    from typing import TypedDict

    from typing_extensions import Unpack

    from neurosush.core.behavior import Behavior

    class Extra(TypedDict):
        size: int

    class Options(Behavior):
        def __init__(self, **kwargs: Unpack[Extra]):
            pass

    class Positional(Behavior):
        def __init__(self, size: int, /):
            pass

    assert describe(Options)["params"] == [
        {"name": "size", "type": {"type": "integer"}, "required": True, "doc": ""}
    ]
    assert describe(Positional)["usable"] is False


def test_check_cli_and_cross_process_fingerprint():
    import os
    import subprocess
    import sys

    environment = {**os.environ, "PYTHONPATH": "src", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "neurosush.introspect", "--check"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    unavailable = [b["name"] for b in catalog()["behaviors"] if not b["usable"]]
    assert result.returncode == int(bool(unavailable))
    assert {line.split(":")[0] for line in result.stdout.splitlines()} == set(unavailable)
    code = "from neurosush.introspect import fingerprint; print(fingerprint())"
    other = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=True
    )
    assert other.stdout.strip() == fingerprint()


def test_unpack_optional_keys_and_fixed_array_shape():
    from typing import TypedDict

    from typing_extensions import NotRequired, Unpack

    from neurosush.core.behavior import Behavior

    class Extra(TypedDict):
        pair: tuple[int, int]
        label: NotRequired[str]

    class Options(Behavior):
        def __init__(self, **kwargs: Unpack[Extra]):
            pass

    params = {p["name"]: p for p in describe(Options)["params"]}
    assert params["label"]["required"] is False
    assert "default" not in params["label"]
    assert params["pair"]["type"]["minItems"] == 2
    assert params["pair"]["type"]["maxItems"] == 2


def test_check_reports_registered_non_json_constructor(monkeypatch, capsys):
    import sys

    import torch

    from neurosush.core.behavior import Behavior
    from neurosush.introspect import main
    from neurosush.structure.spec import _REGISTRY

    class RequiredTensor(Behavior):
        def __init__(self, weights: torch.Tensor):
            pass

    monkeypatch.setitem(_REGISTRY, "RequiredTensor", RequiredTensor)
    monkeypatch.setattr(sys, "argv", ["neurosush.introspect", "--check"])
    assert describe(RequiredTensor)["usable"] is False
    assert main() == 1
    message = capsys.readouterr().out
    assert "RequiredTensor: weights:" in message
    assert "JSON literal adapter" in message


def test_non_finite_default_is_reported_without_invalid_json():
    from neurosush.core.behavior import Behavior

    class Options(Behavior):
        def __init__(self, upper: float = float("inf")):
            pass

    description = describe(Options)
    assert description["params"] == [
        {"name": "upper", "type": {"type": "number"}, "required": False, "doc": ""},
    ]
    assert any("upper" in reason and "inf" in reason for reason in description["limitations"])
    json.dumps(description, allow_nan=False)
