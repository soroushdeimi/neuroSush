from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from experiments.tem_overnight_report import analyze_variant, generate_report, load_results


def make_run_json(
    variant: str, seed: int, status: str = "complete", accuracy: float = 0.4, frequency: float = 0.1
) -> dict:
    """Generate synthetic run dictionary for testing."""
    return {
        "variant": variant,
        "seed": seed,
        "status": status,
        "completed_updates": 16000,
        "max_grad": 100.0,
        "p95_grad": 50.0,
        "evaluations": [
            {
                "accuracy": accuracy,
                "frequency": frequency,
                "sensory_only": 0.15,
                "conditional": 0.12,
                "node_conditional": 0.08,
                "generative_ce": 3.0,
                "position_r2": 0.2,
                "shift_null_p95": 0.05,
                "events": 500,
            }
        ],
    }


def test_pass_synthetic(tmp_path: Path) -> None:
    """Test n=3 passing synthetic leads to PASS."""
    for i in range(3):
        run = make_run_json("v0", i, "complete", accuracy=0.45 + i * 0.01, frequency=0.1)
        p = tmp_path / f"v0-s{i}.json"
        p.write_text(json.dumps(run))

    aggregated, skipped = load_results(tmp_path)
    assert len(aggregated["v0"]) == 3
    assert len(skipped) == 0

    res = analyze_variant(aggregated["v0"])
    assert res["n"] == 3
    assert "PASS" in res["verdict"]
    assert res["accuracy_mean"] == pytest.approx(0.46)
    assert res["accuracy_sd"] == pytest.approx(0.01)
    assert res["accuracy_ci"] == "[0.4352, 0.4848]"


def test_fail_seed_below_frequency(tmp_path: Path) -> None:
    """Test n=3 with one seed below its frequency results in FAIL."""
    run0 = make_run_json("v1", 0, "complete", accuracy=0.45, frequency=0.1)
    run1 = make_run_json("v1", 1, "complete", accuracy=0.46, frequency=0.1)
    run2 = make_run_json("v1", 2, "complete", accuracy=0.05, frequency=0.1)

    for i, run in enumerate([run0, run1, run2]):
        p = tmp_path / f"v1-s{i}.json"
        p.write_text(json.dumps(run))

    aggregated, _skipped = load_results(tmp_path)
    res = analyze_variant(aggregated["v1"])
    assert res["n"] == 3
    assert res["verdict"] == "FAIL"


def test_n1_inconclusive(tmp_path: Path) -> None:
    """Test n=1 reports INCONCLUSIVE."""
    run0 = make_run_json("v2", 0, "complete", accuracy=0.4, frequency=0.1)
    p0 = tmp_path / "v2-s0.json"
    p0.write_text(json.dumps(run0))

    aggregated, _skipped = load_results(tmp_path)
    res = analyze_variant(aggregated["v2"])
    assert res["n"] == 1
    assert res["verdict"] == "INCONCLUSIVE (n=1 seeds, need 3)"
    assert res["accuracy_ci"] == "n=1, no CI"
    assert res["accuracy_sd"] == 0.0


def test_incomplete_file_skipped(tmp_path: Path) -> None:
    """Test incomplete or running files are skipped and status reason is listed."""
    run_complete = make_run_json("v0", 0, "complete", accuracy=0.4, frequency=0.1)
    run_running = make_run_json("v0", 1, "running", accuracy=0.4, frequency=0.1)

    p0 = tmp_path / "v0-s0.json"
    p1 = tmp_path / "v0-s1.json"
    p0.write_text(json.dumps(run_complete))
    p1.write_text(json.dumps(run_running))

    aggregated, skipped = load_results(tmp_path)
    assert len(aggregated["v0"]) == 1
    assert len(skipped) == 1
    assert "v0-s1.json" in skipped[0]
    assert "running" in skipped[0]

    report = generate_report(tmp_path)
    assert "v0-s1.json" in report
    assert "running" in report


def test_report_without_numpy(tmp_path: Path) -> None:
    script = """
import importlib.abc
import sys
from pathlib import Path

class NoNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("numpy is unavailable")

sys.meta_path.insert(0, NoNumpy())
from experiments.tem_overnight_report import generate_report
assert "NO DATA" in generate_report(Path(sys.argv[1]))
assert "numpy" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)
