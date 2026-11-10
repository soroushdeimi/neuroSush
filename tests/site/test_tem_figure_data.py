import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "digitize_tem_figures.py"
DATA = ROOT / "scripts" / "site_assets" / "story-tem" / "paper-figures" / "native-data.json"
FACTS = ROOT / "scripts" / "site_assets" / "story-tem" / "facts.json"

EXPECTED_KEYS = ["11", "12", "13", "14", "15", "16", "17"]
EXPECTED_KINDS = {
    "11": "line", "12": "histograms", "13": "heatmaps",
    "14": "bars", "15": "bars", "16": "heatmaps", "17": "bars",
}


def _facts():
    return {f["id"]: f["value"] for f in json.loads(FACTS.read_text(encoding="utf-8"))}


def test_generator_is_reproducible():
    # regenerate, then confirm --check reports the committed file matches
    subprocess.run([sys.executable, str(SCRIPT)], check=True)
    result = subprocess.run([sys.executable, str(SCRIPT), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reproducible" in result.stdout


def test_all_seven_entries_present_with_required_fields():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    assert sorted(data.keys()) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        entry = data[key]
        assert entry["kind"] == EXPECTED_KINDS[key]
        for field in ("title", "source", "provenance", "approx", "xLabel", "yLabel"):
            assert field in entry, f"entry {key} missing {field}"
        assert entry["provenance"] in ("reported", "digitized")
        assert isinstance(entry["approx"], bool)


def test_cartesian_entries_have_series_points():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    for key in ("11", "12", "14", "15", "17"):
        entry = data[key]
        assert "series" in entry
        for series in entry["series"]:
            assert series["color"] in ("accent", "blue")
            assert len(series["points"]) > 0
            for point in series["points"]:
                assert isinstance(point["y"], (int, float))
                assert point["y"] >= 0


def test_heatmaps_have_finite_normalized_square_grids():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    for key in ("13", "16"):
        for group in data[key]["groups"]:
            assert "name" in group
            for map in group["maps"]:
                values = map["values"]
                n = len(values)
                assert n == len(values[0]), "grid is not square"
                for row in values:
                    assert len(row) == n
                    for v in row:
                        assert 0.0 <= v <= 1.0, f"heatmap value {v} out of 0..1"


def test_exact_table_values_match_ledger():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    facts = _facts()

    # 14 ablations: every mean and SD is exact from Table 2
    ablation = data["14"]["series"][0]["points"]
    conds = [
        ("Baseline", "grid_mean", "grid_sd"),
        ("No feedback", "no_feedback", "no_feedback_sd"),
        ("No theta", "no_theta", "no_theta_sd"),
        ("No STDP", "no_stdp", "no_stdp_sd"),
        ("No sparsity", "no_sparsity", "no_sparsity_sd"),
        ("MSE only", "mse", "mse_sd"),
        ("No gain", "no_gain", "no_gain_sd"),
    ]
    assert data["14"]["provenance"] == "reported"
    assert data["14"]["approx"] is False
    for point, (name, m, s) in zip(ablation, conds):
        assert point["x"] == name
        assert point["y"] == facts[m]
        assert point["sd"] == facts[s]

    # 11 endpoint is the exact grid_mean/grid_sd
    endpoint = data["11"]["series"][0]["points"][-1]
    assert endpoint["x"] == 24000
    assert endpoint["y"] == facts["grid_mean"]
    assert endpoint["sd"] == facts["grid_sd"]

    # 17 dominant behaviors are exact
    phase = {(s["name"], p["x"]): p["y"]
             for s in data["17"]["series"] for p in s["points"]}
    assert phase[("MEC II", "precession")] == facts["precession_mean"]
    assert phase[("MEC III", "locking")] == facts["locking_mean"]


def test_digitized_entries_are_marked_approximate():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    for key in ("11", "12", "13", "15", "16", "17"):
        assert data[key]["provenance"] == "digitized"
        assert data[key]["approx"] is True


def test_line_chart_x_geometry_matches_iterations():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    xs = [p["x"] for p in data["11"]["series"][0]["points"]]
    assert xs == [0, 4000, 8000, 12000, 16000, 20000, 24000]


def test_histogram_bins_are_the_published_edges():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    bins = [p["x"] for p in data["12"]["series"][0]["points"]]
    assert bins == ["0-0.74", "0.74-1.49", "1.49-2.23", "2.23-2.98", "2.98-3.72",
                    "3.72-4.47", "4.47-5.21", "5.21-5.96", "5.96-6.70", "6.70-7.45"]
    # all three conditions share the same bins
    for series in data["12"]["series"]:
        assert [p["x"] for p in series["points"]] == bins


def test_missing_source_raises_clear_error(tmp_path, monkeypatch):
    # point the module at a directory without the figures, expect FileNotFoundError
    import importlib.util
    spec = importlib.util.spec_from_file_location("digitize_tem_figures", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="missing source figure"):
        mod._open("grid-emergence.png")
