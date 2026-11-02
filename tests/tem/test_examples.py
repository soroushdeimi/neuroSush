"""Tests for TEM example scripts (tem_whittington and spiking_tem_kawahara)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_tem_whittington_oracle_run_keys_and_values():
    mod = load("tem_whittington")
    res = mod.run(mode="oracle", steps=3, walk_len=40, seeds=0)
    assert "oracle" in res
    assert len(res["oracle"]) == 1
    eval_dict = res["oracle"][0]["evaluation"]
    expected_keys = [
        "accuracy",
        "sensory_only",
        "frequency",
        "conditional",
        "node_conditional",
        "events",
        "position_r2",
        "shift_null_p95",
    ]
    for key in expected_keys:
        assert key in eval_dict
    for key in ("accuracy", "sensory_only", "frequency", "conditional", "node_conditional"):
        val = eval_dict[key]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0


def test_tem_whittington_seeded_determinism():
    mod = load("tem_whittington")
    r1 = mod.run(mode="oracle", steps=3, walk_len=40, seeds=0)
    r2 = mod.run(mode="oracle", steps=3, walk_len=40, seeds=0)
    assert r1["oracle"][0]["evaluation"] == r2["oracle"][0]["evaluation"]


def test_tem_whittington_json_serializable():
    mod = load("tem_whittington")
    res = mod.run(mode="oracle", steps=3, walk_len=40, seeds=0)
    text = json.dumps(res)
    assert isinstance(text, str)


def test_spiking_tem_kawahara_2d_and_1d():
    mod = load("spiking_tem_kawahara")
    res_2d = mod.run(
        arena="2d",
        config_choice="default",
        steps=2,
        walk_len=30,
        seeds=0,
    )
    assert "2d" in res_2d
    assert len(res_2d["2d"]) == 1
    gf = res_2d["2d"][0]["grid_fraction"]
    assert "hex_fraction" in gf
    assert "square_fraction" in gf
    assert 0.0 <= gf["hex_fraction"] <= 1.0
    assert 0.0 <= gf["square_fraction"] <= 1.0
    assert "label" in gf
    assert "resolvable" in gf

    # (f) 2D result at 8x8 states resolvable False and does not contain "grid cells emerged"
    assert gf["resolvable"] is False
    all_strings = " ".join(str(v) for v in res_2d.values() if isinstance(v, (str, dict, list)))
    assert "grid cells emerged" not in all_strings.lower()

    res_1d = mod.run(
        arena="1d",
        config_choice="default",
        steps=2,
        walk_len=30,
        seeds=0,
    )
    assert "1d" in res_1d
    assert len(res_1d["1d"]) == 1
    summary = res_1d["1d"][0]["phase_summary"]
    assert "counts" in summary
    counts = summary["counts"]
    total_counts = sum(counts.values())
    slopes = res_1d["1d"][0].get("slopes", [])
    if slopes:
        assert total_counts == len(slopes)


def test_spiking_tem_ablation_flags():
    mod = load("spiking_tem_kawahara")
    res_default = mod.run(
        arena="2d",
        config_choice="default",
        steps=2,
        walk_len=20,
        seeds=0,
        no_theta=False,
        no_stdp=False,
        no_gain=False,
    )
    res_ablated = mod.run(
        arena="2d",
        config_choice="default",
        steps=2,
        walk_len=20,
        seeds=0,
        no_theta=True,
        no_stdp=True,
        no_gain=True,
    )
    assert res_default["config"]["use_theta"] is True
    assert res_default["config"]["use_stdp"] is True
    assert res_default["config"]["use_gain"] is True

    assert res_ablated["config"]["use_theta"] is False
    assert res_ablated["config"]["use_stdp"] is False
    assert res_ablated["config"]["use_gain"] is False

    assert res_default["config"] != res_ablated["config"]
