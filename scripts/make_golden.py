"""Generate numerical golden vectors for neuroSush Atlas equations.

Outputs have 14 significant digits (13 for explosive exponential cases), sorted
keys and no timestamps. Cross-platform agreement is numerical, not byte-for-byte:
rounding cannot guarantee identical text at a decimal rounding boundary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import torch

# Ensure neurosush can be imported when running script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from neurosush.neurons.dynamics import (  # noqa: E402
    adex_terms,
    elif_terms,
    euler_step,
    exponential_boost,
    izhikevich_terms,
    lif_derivative,
    lif_terms,
)

OUTPUT_PATH = REPO_ROOT / "tests" / "golden" / "equations.json"

V_GRID_STANDARD = [-80.0, -75.0, -70.0, -65.0, -60.0, -55.0, -50.0, -45.0, -40.0]
V_GRID_THRESHOLD = [-55.0, -50.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, 0.0]


def _t(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def generate_golden() -> dict[str, Any]:
    """Generate golden test cases for all neuron dynamics equations and primitives.

    Computations use float64; output rounding and tolerances are defined by case_tolerance.

    Returns:
        Dictionary of golden equation cases with a 'generated_by' note.
    """
    # 1. lif_terms
    lif_cases = []
    lif_params: list[dict[str, Any]] = [
        {
            "v": V_GRID_STANDARD,
            "current": 0.0,
            "v_rest": -65.0,
            "resistance": 1.0,
            "tau": 20.0,
            "dt": 0.5,
        },
        {
            "v": V_GRID_STANDARD,
            "current": 2.0,
            "v_rest": -65.0,
            "resistance": 1.0,
            "tau": 20.0,
            "dt": 0.5,
        },
        {
            "v": V_GRID_STANDARD,
            "current": 10.0,
            "v_rest": -70.0,
            "resistance": 2.0,
            "tau": 10.0,
            "dt": 1.0,
        },
    ]
    for p in lif_params:
        v_tensor = _t(p["v"])
        terms = lif_terms(
            v_tensor,
            p["current"],
            v_rest=p["v_rest"],
            resistance=p["resistance"],
            tau=p["tau"],
            dt=p["dt"],
        )
        total = cast(torch.Tensor, sum(terms.values())).tolist()
        outputs = {k: v.tolist() for k, v in terms.items()}
        lif_cases.append({"inputs": dict(p), "outputs": outputs, "total": total})

    # 2. elif_terms
    elif_cases = []
    elif_params: list[dict[str, Any]] = [
        {
            "v": V_GRID_STANDARD,
            "current": 0.0,
            "v_rest": -65.0,
            "resistance": 1.0,
            "delta": 2.0,
            "theta_rh": -50.0,
            "tau": 20.0,
            "dt": 0.5,
        },
        {
            "v": V_GRID_STANDARD,
            "current": 2.0,
            "v_rest": -65.0,
            "resistance": 1.0,
            "delta": 2.0,
            "theta_rh": -50.0,
            "tau": 20.0,
            "dt": 0.5,
        },
        {
            "v": V_GRID_STANDARD,
            "current": 5.0,
            "v_rest": -70.0,
            "resistance": 1.5,
            "delta": 1.0,
            "theta_rh": -55.0,
            "tau": 15.0,
            "dt": 1.0,
        },
        {
            "v": V_GRID_THRESHOLD,
            "current": 0.0,
            "v_rest": -65.0,
            "resistance": 1.0,
            "delta": 2.0,
            "theta_rh": -50.0,
            "tau": 20.0,
            "dt": 0.5,
        },
        {
            "v": V_GRID_THRESHOLD,
            "current": 2.0,
            "v_rest": -65.0,
            "resistance": 1.0,
            "delta": 2.0,
            "theta_rh": -50.0,
            "tau": 20.0,
            "dt": 0.5,
        },
    ]
    for p in elif_params:
        v_tensor = _t(p["v"])
        terms = elif_terms(
            v_tensor,
            p["current"],
            v_rest=p["v_rest"],
            resistance=p["resistance"],
            delta=p["delta"],
            theta_rh=p["theta_rh"],
            tau=p["tau"],
            dt=p["dt"],
        )
        total = cast(torch.Tensor, sum(terms.values())).tolist()
        outputs = {k: v.tolist() for k, v in terms.items()}
        elif_cases.append({"inputs": dict(p), "outputs": outputs, "total": total})

    # 3. adex_terms
    adex_cases = []
    adex_params: list[dict[str, Any]] = [
        {
            "v": V_GRID_STANDARD,
            "current": 0.0,
            "omega": 0.0,
            "v_rest": -65.0,
            "resistance": 1.0,
            "delta": 2.0,
            "theta_rh": -50.0,
            "tau": 20.0,
            "dt": 0.5,
        },
        {
            "v": V_GRID_STANDARD,
            "current": 2.0,
            "omega": 1.5,
            "v_rest": -65.0,
            "resistance": 2.0,
            "delta": 2.0,
            "theta_rh": -50.0,
            "tau": 20.0,
            "dt": 1.0,
        },
        {
            "v": V_GRID_STANDARD,
            "current": 5.0,
            "omega": 2.0,
            "v_rest": -70.0,
            "resistance": 1.5,
            "delta": 1.0,
            "theta_rh": -55.0,
            "tau": 15.0,
            "dt": 1.0,
        },
        {
            "v": V_GRID_THRESHOLD,
            "current": 0.0,
            "omega": 1.0,
            "v_rest": -65.0,
            "resistance": 1.0,
            "delta": 2.0,
            "theta_rh": -50.0,
            "tau": 20.0,
            "dt": 0.5,
        },
        {
            "v": V_GRID_THRESHOLD,
            "current": 2.0,
            "omega": 1.5,
            "v_rest": -65.0,
            "resistance": 1.0,
            "delta": 2.0,
            "theta_rh": -50.0,
            "tau": 20.0,
            "dt": 0.5,
        },
    ]
    for p in adex_params:
        v_tensor = _t(p["v"])
        terms = adex_terms(
            v_tensor,
            p["current"],
            p["omega"],
            v_rest=p["v_rest"],
            resistance=p["resistance"],
            delta=p["delta"],
            theta_rh=p["theta_rh"],
            tau=p["tau"],
            dt=p["dt"],
        )
        total = cast(torch.Tensor, sum(terms.values())).tolist()
        outputs = {k: v.tolist() for k, v in terms.items()}
        adex_cases.append({"inputs": dict(p), "outputs": outputs, "total": total})

    # 4. izhikevich_terms
    izh_cases = []
    izh_params: list[dict[str, Any]] = [
        {"v": V_GRID_STANDARD, "u": -13.0, "current": 0.0, "dt": 1.0, "substeps": 2},
        {"v": V_GRID_STANDARD, "u": -13.0, "current": 10.0, "dt": 1.0, "substeps": 2},
        {"v": V_GRID_STANDARD, "u": -10.0, "current": 5.0, "dt": 0.5, "substeps": 2},
        {"v": V_GRID_STANDARD, "u": 0.0, "current": 10.0, "dt": 1.0, "substeps": 1},
    ]
    for p in izh_params:
        v_tensor = _t(p["v"])
        u_tensor = torch.tensor(p["u"], dtype=torch.float64)
        terms = izhikevich_terms(
            v_tensor,
            u_tensor,
            p["current"],
            dt=p["dt"],
            substeps=p["substeps"],
        )
        total = cast(torch.Tensor, sum(terms.values())).tolist()
        outputs = {k: v.tolist() for k, v in terms.items()}
        izh_cases.append({"inputs": dict(p), "outputs": outputs, "total": total})

    # 5. lif_derivative
    deriv_cases = []
    deriv_params: list[dict[str, Any]] = [
        {"v": V_GRID_STANDARD, "current": 0.0, "v_rest": -65.0, "resistance": 1.0},
        {"v": V_GRID_STANDARD, "current": 2.0, "v_rest": -65.0, "resistance": 1.0},
        {"v": V_GRID_STANDARD, "current": 10.0, "v_rest": -70.0, "resistance": 2.0},
    ]
    for p in deriv_params:
        v_tensor = _t(p["v"])
        out = lif_derivative(v_tensor, p["current"], v_rest=p["v_rest"], resistance=p["resistance"])
        out_list = out.tolist()
        deriv_cases.append({"inputs": dict(p), "output": out_list, "outputs": out_list})

    # 6. exponential_boost
    boost_cases = []
    boost_params: list[dict[str, Any]] = [
        {"v": V_GRID_STANDARD, "delta": 2.0, "theta_rh": -50.0},
        {"v": V_GRID_THRESHOLD, "delta": 2.0, "theta_rh": -50.0},
        {"v": V_GRID_STANDARD, "delta": 1.0, "theta_rh": -55.0},
    ]
    for p in boost_params:
        v_tensor = _t(p["v"])
        out = exponential_boost(v_tensor, delta=p["delta"], theta_rh=p["theta_rh"])
        out_list = out.tolist()
        boost_cases.append({"inputs": dict(p), "output": out_list, "outputs": out_list})

    # 7. euler_step
    euler_cases = []
    euler_params: list[dict[str, Any]] = [
        {
            "v": V_GRID_STANDARD,
            "tau_dv_dt": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "tau": 20.0,
            "dt": 0.5,
        },
        {
            "v": V_GRID_STANDARD,
            "tau_dv_dt": [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "tau": 10.0,
            "dt": 1.0,
        },
    ]
    for p in euler_params:
        v_tensor = _t(p["v"])
        tau_dv_dt_tensor = _t(p["tau_dv_dt"])
        out = euler_step(v_tensor, tau_dv_dt_tensor, tau=p["tau"], dt=p["dt"])
        out_list = out.tolist()
        euler_cases.append({"inputs": dict(p), "output": out_list, "outputs": out_list})

    # 8. trace_decay
    trace_decay_cases = []
    decay_params: list[dict[str, Any]] = [
        {"dt": 1.0, "tau": 20.0},
        {"dt": 0.5, "tau": 10.0},
        {"dt": 2.0, "tau": 100.0},
        {"dt": 0.1, "tau": 5.0},
    ]
    for p in decay_params:
        factors = [(1.0 - p["dt"] / p["tau"]) ** k for k in range(11)]
        trace_decay_cases.append({"inputs": dict(p), "output": factors, "outputs": factors})

    data: dict[str, Any] = {
        "adex_terms": adex_cases,
        "elif_terms": elif_cases,
        "euler_step": euler_cases,
        "exponential_boost": boost_cases,
        "generated_by": "scripts/make_golden.py",
        "izhikevich_terms": izh_cases,
        "lif_derivative": deriv_cases,
        "lif_terms": lif_cases,
        "trace_decay": trace_decay_cases,
    }
    for name, cases in data.items():
        if name == "generated_by":
            continue
        for case in cases:
            tolerance = case_tolerance(case["inputs"])
            case["tolerance"] = tolerance
            for key in case.keys() - {"inputs", "tolerance"}:
                case[key] = _round_output(case[key], tolerance["significant_digits"])
    return data


def case_tolerance(inputs: dict[str, Any]) -> dict[str, Any]:
    """Choose the numerical contract from the exponential argument of a case.

    Stable cases allow 1e-12 relative error, about 4500 float64 epsilons, for
    kernel differences and 14-digit rounding (at most 5e-14 relative error).
    Arguments >= 20 produce exp(20)..exp(25) in this grid, reaching billions
    before reset. They get a tenfold guard, 1e-11, and 13 digits (at most 5e-13
    relative rounding error). A one-ULP difference there is about 1e-6 absolute,
    but still about 2e-16 relative. Neither budget admits 1e-6 relative drift.
    The 1e-14 absolute floor covers cancellation near zero, not large values.

    Args:
        inputs: Unrounded equation inputs, including optional exponential parameters.

    Returns:
        Relative/absolute tolerances and output precision to store beside the case.
    """
    explosive = (
        "delta" in inputs
        and max((v - inputs["theta_rh"]) / inputs["delta"] for v in inputs["v"]) >= 20.0
    )
    return {
        "rel": 1e-11 if explosive else 1e-12,
        "abs": 1e-14,
        "significant_digits": 13 if explosive else 14,
    }


def _round_output(value: Any, digits: int) -> Any:
    if isinstance(value, dict):
        return {key: _round_output(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_output(item, digits) for item in value]
    return float(format(value, f".{digits}g"))


build_golden = generate_golden


def main() -> None:
    """Write the rounded numerical contract to OUTPUT_PATH.

    Returns:
        None.
    """
    data = generate_golden()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, sort_keys=True) + "\n"
    OUTPUT_PATH.write_text(serialized, encoding="utf-8")
    print(f"Wrote {len(serialized)} bytes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
