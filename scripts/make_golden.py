"""Generate deterministic golden vectors for neuroSush Atlas equations.

Outputs tests/golden/equations.json with sorted keys and no timestamps, so regeneration
is byte-identical.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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

    All computations are performed in float64 for exact precision.

    Returns:
        Dictionary of golden equation cases with a 'generated_by' note.
    """
    # 1. lif_terms
    lif_cases = []
    lif_params = [
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
        total = sum(terms.values()).tolist()
        outputs = {k: v.tolist() for k, v in terms.items()}
        lif_cases.append({"inputs": dict(p), "outputs": outputs, "total": total})

    # 2. elif_terms
    elif_cases = []
    elif_params = [
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
        total = sum(terms.values()).tolist()
        outputs = {k: v.tolist() for k, v in terms.items()}
        elif_cases.append({"inputs": dict(p), "outputs": outputs, "total": total})

    # 3. adex_terms
    adex_cases = []
    adex_params = [
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
        total = sum(terms.values()).tolist()
        outputs = {k: v.tolist() for k, v in terms.items()}
        adex_cases.append({"inputs": dict(p), "outputs": outputs, "total": total})

    # 4. izhikevich_terms
    izh_cases = []
    izh_params = [
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
        total = sum(terms.values()).tolist()
        outputs = {k: v.tolist() for k, v in terms.items()}
        izh_cases.append({"inputs": dict(p), "outputs": outputs, "total": total})

    # 5. lif_derivative
    deriv_cases = []
    deriv_params = [
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
    boost_params = [
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
    euler_params = [
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
    decay_params = [
        {"dt": 1.0, "tau": 20.0},
        {"dt": 0.5, "tau": 10.0},
        {"dt": 2.0, "tau": 100.0},
        {"dt": 0.1, "tau": 5.0},
    ]
    for p in decay_params:
        factors = [(1.0 - p["dt"] / p["tau"]) ** k for k in range(11)]
        trace_decay_cases.append({"inputs": dict(p), "output": factors, "outputs": factors})

    return {
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


build_golden = generate_golden


def main() -> None:
    data = generate_golden()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, sort_keys=True) + "\n"
    OUTPUT_PATH.write_text(serialized, encoding="utf-8")
    print(f"Wrote {len(serialized)} bytes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
