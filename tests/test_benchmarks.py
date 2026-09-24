"""The benchmark comparison: ratios, the regression rule and the report."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "compare", Path(__file__).resolve().parents[1] / "benchmarks" / "compare.py"
)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def results(**normalized):
    return {"cases": {name: {"normalized": value} for name, value in normalized.items()}}


def test_ratios_compare_normalized_rates():
    changes = compare.ratios(results(a=2.0, b=1.0), results(a=1.0, b=1.5, c=3.0))
    assert changes == {"a": 0.5, "b": 1.5, "c": None}


def test_only_drops_beyond_the_tolerance_regress():
    changes = {"slow": 0.6, "noisy": 0.75, "fast": 1.4, "new": None}
    assert compare.regressions(changes, tolerance=0.3) == ["slow"]


def test_table_reports_every_case():
    report = compare.table({"a": 0.5, "b": 1.1, "c": None}, tolerance=0.3)
    assert "| a | -50.0% | regression |" in report
    assert "| b | +10.0% | ok |" in report
    assert "| c | n/a | missing on one side |" in report


@pytest.mark.parametrize(("current", "code"), [(1.0, 0), (0.6, 0), (0.4, 1)])
def test_exit_code(tmp_path, monkeypatch, current, code):
    (tmp_path / "base.json").write_text(json.dumps(results(a=1.0)))
    (tmp_path / "now.json").write_text(json.dumps(results(a=current)))
    monkeypatch.setattr(
        "sys.argv", ["compare", str(tmp_path / "base.json"), str(tmp_path / "now.json")]
    )
    assert compare.main() == code


def test_missing_baseline_is_not_an_error(tmp_path, monkeypatch, capsys):
    (tmp_path / "now.json").write_text(json.dumps(results(a=1.0)))
    monkeypatch.setattr(
        "sys.argv", ["compare", str(tmp_path / "none.json"), str(tmp_path / "now.json")]
    )
    assert compare.main() == 0
    assert "No baseline" in capsys.readouterr().out
