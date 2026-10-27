from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any, cast

import pytest
import torch
from scripts import make_golden
from scripts.make_golden import OUTPUT_PATH, case_tolerance, generate_golden

from neurosush.neurons.dynamics import (
    adex_terms,
    elif_terms,
    euler_step,
    exponential_boost,
    izhikevich_terms,
    lif_derivative,
    lif_terms,
)


def t(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_golden_file_is_up_to_date() -> None:
    assert OUTPUT_PATH.exists(), f"Golden file missing: {OUTPUT_PATH}"
    committed_data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    _assert_golden_matches(generate_golden(), committed_data)


def _assert_values_match(actual: Any, expected: Any, tolerance: dict[str, Any], path: str) -> None:
    assert type(actual) is type(expected), f"{path}: type {type(actual)} != {type(expected)}"
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (
            f"{path}: keys {actual.keys()} != {expected.keys()}"
        )
        for key in expected:
            _assert_values_match(actual[key], expected[key], tolerance, f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            _assert_values_match(value, reference, tolerance, f"{path}[{index}]")
    else:
        assert math.isfinite(actual), f"{path}: nonfinite actual {actual}"
        assert math.isfinite(expected), f"{path}: nonfinite expected {expected}"
        assert math.isclose(actual, expected, rel_tol=tolerance["rel"], abs_tol=tolerance["abs"]), (
            f"{path}: {actual} != {expected}; tolerance={tolerance}; "
            "tests/golden/equations.json is stale; run 'python scripts/make_golden.py' to update"
        )


def _assert_golden_matches(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys(), f"equations: keys {actual.keys()} != {expected.keys()}"
    assert actual["generated_by"] == expected["generated_by"]
    for name in expected.keys() - {"generated_by"}:
        assert len(actual[name]) == len(expected[name]), f"{name}: case count changed"
        for index, (case, reference) in enumerate(zip(actual[name], expected[name], strict=True)):
            path = f"{name}[{index}]"
            assert case.keys() == reference.keys(), f"{path}: keys changed"
            assert case["inputs"] == reference["inputs"], f"{path}: inputs changed"
            tolerance = case_tolerance(reference["inputs"])
            assert case["tolerance"] == reference["tolerance"] == tolerance, (
                f"{path}: tolerance changed"
            )
            for key in reference.keys() - {"inputs", "tolerance"}:
                _assert_values_match(case[key], reference[key], tolerance, f"{path}.{key}")


def _committed_golden() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(OUTPUT_PATH.read_text(encoding="utf-8")))


def test_golden_lif_terms() -> None:
    data = _committed_golden()
    for case in data["lif_terms"]:
        inp = case["inputs"]
        terms = lif_terms(
            t(inp["v"]),
            inp["current"],
            v_rest=inp["v_rest"],
            resistance=inp["resistance"],
            tau=inp["tau"],
            dt=inp["dt"],
        )
        for key, expected_vals in case["outputs"].items():
            assert terms[key].tolist() == pytest.approx(
                expected_vals, rel=case["tolerance"]["rel"], abs=case["tolerance"]["abs"]
            )
        total = cast(torch.Tensor, sum(terms.values())).tolist()
        assert total == pytest.approx(
            case["total"], rel=case["tolerance"]["rel"], abs=case["tolerance"]["abs"]
        )


def test_golden_elif_terms() -> None:
    data = _committed_golden()
    for case in data["elif_terms"]:
        inp = case["inputs"]
        terms = elif_terms(
            t(inp["v"]),
            inp["current"],
            v_rest=inp["v_rest"],
            resistance=inp["resistance"],
            delta=inp["delta"],
            theta_rh=inp["theta_rh"],
            tau=inp["tau"],
            dt=inp["dt"],
        )
        for key, expected_vals in case["outputs"].items():
            assert terms[key].tolist() == pytest.approx(
                expected_vals, rel=case["tolerance"]["rel"], abs=case["tolerance"]["abs"]
            )
        total = cast(torch.Tensor, sum(terms.values())).tolist()
        assert total == pytest.approx(
            case["total"], rel=case["tolerance"]["rel"], abs=case["tolerance"]["abs"]
        )


def test_golden_adex_terms() -> None:
    data = _committed_golden()
    for case in data["adex_terms"]:
        inp = case["inputs"]
        terms = adex_terms(
            t(inp["v"]),
            inp["current"],
            inp["omega"],
            v_rest=inp["v_rest"],
            resistance=inp["resistance"],
            delta=inp["delta"],
            theta_rh=inp["theta_rh"],
            tau=inp["tau"],
            dt=inp["dt"],
        )
        for key, expected_vals in case["outputs"].items():
            assert terms[key].tolist() == pytest.approx(
                expected_vals, rel=case["tolerance"]["rel"], abs=case["tolerance"]["abs"]
            )
        total = cast(torch.Tensor, sum(terms.values())).tolist()
        assert total == pytest.approx(
            case["total"], rel=case["tolerance"]["rel"], abs=case["tolerance"]["abs"]
        )


def test_golden_izhikevich_terms() -> None:
    data = _committed_golden()
    for case in data["izhikevich_terms"]:
        inp = case["inputs"]
        terms = izhikevich_terms(
            t(inp["v"]),
            torch.tensor(inp["u"], dtype=torch.float64),
            inp["current"],
            dt=inp["dt"],
            substeps=inp["substeps"],
        )
        for key, expected_vals in case["outputs"].items():
            assert terms[key].tolist() == pytest.approx(
                expected_vals, rel=case["tolerance"]["rel"], abs=case["tolerance"]["abs"]
            )
        total = cast(torch.Tensor, sum(terms.values())).tolist()
        assert total == pytest.approx(
            case["total"], rel=case["tolerance"]["rel"], abs=case["tolerance"]["abs"]
        )


def test_golden_primitives() -> None:
    data = _committed_golden()
    for case in data["lif_derivative"]:
        inp = case["inputs"]
        out = lif_derivative(
            t(inp["v"]),
            inp["current"],
            v_rest=inp["v_rest"],
            resistance=inp["resistance"],
        )
        assert out.tolist() == pytest.approx(
            case["output"], rel=case["tolerance"]["rel"], abs=case["tolerance"]["abs"]
        )

    for case in data["exponential_boost"]:
        inp = case["inputs"]
        out = exponential_boost(
            t(inp["v"]),
            delta=inp["delta"],
            theta_rh=inp["theta_rh"],
        )
        assert out.tolist() == pytest.approx(
            case["output"], rel=case["tolerance"]["rel"], abs=case["tolerance"]["abs"]
        )

    for case in data["euler_step"]:
        inp = case["inputs"]
        out = euler_step(
            t(inp["v"]),
            t(inp["tau_dv_dt"]),
            tau=inp["tau"],
            dt=inp["dt"],
        )
        assert out.tolist() == pytest.approx(
            case["output"], rel=case["tolerance"]["rel"], abs=case["tolerance"]["abs"]
        )

    for case in data["trace_decay"]:
        inp = case["inputs"]
        expected = [(1.0 - inp["dt"] / inp["tau"]) ** k for k in range(11)]
        assert case["output"] == pytest.approx(
            expected, rel=case["tolerance"]["rel"], abs=case["tolerance"]["abs"]
        )


@pytest.mark.parametrize("name", ["lif_terms", "elif_terms", "adex_terms", "exponential_boost"])
def test_golden_accepts_one_ulp_changes(name: str) -> None:
    expected = _committed_golden()
    perturbed = deepcopy(expected)
    for case in perturbed[name]:
        outputs = case["outputs"]
        arrays = outputs.values() if isinstance(outputs, dict) else [outputs]
        for values in arrays:
            values[:] = [math.nextafter(value, math.inf) for value in values]
        if "total" in case:
            case["total"] = [math.nextafter(value, math.inf) for value in case["total"]]
    _assert_golden_matches(perturbed, expected)


@pytest.mark.parametrize(("name", "index"), [("lif_terms", 0), ("elif_terms", 3)])
def test_golden_rejects_relative_equation_drift(name: str, index: int) -> None:
    expected = _committed_golden()
    perturbed = deepcopy(expected)
    term = "leak" if name == "lif_terms" else "spike_initiation"
    perturbed[name][index]["outputs"][term][-1] *= 1.0 + 1e-6
    with pytest.raises(AssertionError, match=rf"outputs\.{term}"):
        _assert_golden_matches(perturbed, expected)


@pytest.mark.parametrize("change", ["equation", "case", "input", "term", "length", "tolerance"])
def test_golden_rejects_contract_changes(change: str) -> None:
    expected = _committed_golden()
    perturbed = deepcopy(expected)
    case = perturbed["lif_terms"][0]
    if change == "equation":
        perturbed["renamed_equation"] = perturbed.pop("euler_step")
    elif change == "case":
        perturbed["lif_terms"].pop()
    elif change == "input":
        case["inputs"]["v"][0] = math.nextafter(case["inputs"]["v"][0], math.inf)
    elif change == "term":
        case["outputs"]["renamed_leak"] = case["outputs"].pop("leak")
    elif change == "length":
        case["outputs"]["leak"].pop()
    else:
        case["tolerance"]["rel"] = 1e-3
    with pytest.raises(AssertionError):
        _assert_golden_matches(perturbed, expected)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_golden_rejects_nonfinite_values(value: float) -> None:
    expected = _committed_golden()
    perturbed = deepcopy(expected)
    perturbed["lif_terms"][0]["outputs"]["leak"][0] = value
    with pytest.raises(AssertionError, match="nonfinite"):
        _assert_golden_matches(perturbed, expected)


@pytest.mark.parametrize(("argument", "rel", "digits"), [(19.0, 1e-12, 14), (20.0, 1e-11, 13)])
def test_golden_tolerance_follows_exponential_argument(
    argument: float, rel: float, digits: int
) -> None:
    tolerance = case_tolerance({"v": [-50.0, argument - 50.0], "theta_rh": -50.0, "delta": 1.0})
    assert tolerance == {"rel": rel, "abs": 1e-14, "significant_digits": digits}
    assert case_tolerance({"v": [1e9]})["rel"] == 1e-12


def test_golden_rounding_preserves_numerical_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    rounded = generate_golden()
    monkeypatch.setattr(make_golden, "_round_output", lambda value, digits: value)
    raw = generate_golden()
    _assert_golden_matches(raw, rounded)
    for name in rounded.keys() - {"generated_by"}:
        for case in rounded[name]:
            digits = case["tolerance"]["significant_digits"]
            outputs = case["outputs"]
            arrays = outputs.values() if isinstance(outputs, dict) else [outputs]
            for values in arrays:
                assert all(value == float(format(value, f".{digits}g")) for value in values)


def test_golden_generation_is_repeatable() -> None:
    first = json.dumps(generate_golden(), sort_keys=True)
    assert json.dumps(generate_golden(), sort_keys=True) == first


@pytest.mark.parametrize("argument", [10.0, 25.0])
def test_golden_relative_tolerance_boundary(argument: float) -> None:
    tolerance = case_tolerance({"v": [argument], "theta_rh": 0.0, "delta": 1.0})
    expected = math.exp(argument)
    _assert_values_match(expected * (1.0 + tolerance["rel"] / 2), expected, tolerance, "value")
    with pytest.raises(AssertionError, match="value"):
        _assert_values_match(expected * (1.0 + 2 * tolerance["rel"]), expected, tolerance, "value")
