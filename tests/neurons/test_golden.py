import json

import pytest
import torch
from scripts.make_golden import OUTPUT_PATH, generate_golden

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
    committed_content = OUTPUT_PATH.read_text(encoding="utf-8")
    in_memory_data = generate_golden()
    expected_content = json.dumps(in_memory_data, indent=2, sort_keys=True) + "\n"
    assert committed_content == expected_content, (
        "tests/golden/equations.json is stale; run 'python scripts/make_golden.py' to update"
    )


def test_golden_lif_terms() -> None:
    data = generate_golden()
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
            assert terms[key].tolist() == pytest.approx(expected_vals, abs=1e-12)
        total = sum(terms.values()).tolist()
        assert total == pytest.approx(case["total"], abs=1e-12)


def test_golden_elif_terms() -> None:
    data = generate_golden()
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
            assert terms[key].tolist() == pytest.approx(expected_vals, abs=1e-12)
        total = sum(terms.values()).tolist()
        assert total == pytest.approx(case["total"], abs=1e-12)


def test_golden_adex_terms() -> None:
    data = generate_golden()
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
            assert terms[key].tolist() == pytest.approx(expected_vals, abs=1e-12)
        total = sum(terms.values()).tolist()
        assert total == pytest.approx(case["total"], abs=1e-12)


def test_golden_izhikevich_terms() -> None:
    data = generate_golden()
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
            assert terms[key].tolist() == pytest.approx(expected_vals, abs=1e-12)
        total = sum(terms.values()).tolist()
        assert total == pytest.approx(case["total"], abs=1e-10)


def test_golden_primitives() -> None:
    data = generate_golden()
    for case in data["lif_derivative"]:
        inp = case["inputs"]
        out = lif_derivative(
            t(inp["v"]),
            inp["current"],
            v_rest=inp["v_rest"],
            resistance=inp["resistance"],
        )
        assert out.tolist() == pytest.approx(case["output"], abs=1e-12)

    for case in data["exponential_boost"]:
        inp = case["inputs"]
        out = exponential_boost(
            t(inp["v"]),
            delta=inp["delta"],
            theta_rh=inp["theta_rh"],
        )
        assert out.tolist() == pytest.approx(case["output"], abs=1e-12)

    for case in data["euler_step"]:
        inp = case["inputs"]
        out = euler_step(
            t(inp["v"]),
            t(inp["tau_dv_dt"]),
            tau=inp["tau"],
            dt=inp["dt"],
        )
        assert out.tolist() == pytest.approx(case["output"], abs=1e-12)

    for case in data["trace_decay"]:
        inp = case["inputs"]
        expected = [(1.0 - inp["dt"] / inp["tau"]) ** k for k in range(11)]
        assert case["output"] == pytest.approx(expected, abs=1e-12)
