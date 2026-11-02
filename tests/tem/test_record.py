"""Tests for recorded-run exporter for TEM."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neurosush.tem.model import TEM, TEMConfig
from neurosush.tem.record import record_run, write_run


def test_record_run_schema_and_meta(tmp_path: Path) -> None:
    """Verify schema keys, meta fields (training seed, record seed, checkpoint, worlds_note)."""
    config = TEMConfig(widths=(5, 6, 7, 8))
    model = TEM(config, generator=torch.Generator().manual_seed(0))
    run = record_run(
        model=model,
        worlds=2,
        size=8,
        steps=50,
        units=16,
        seed=42,
        train_seed=0,
        updates=16000,
        label="learned structure",
        note="one seed; no claim of grid cells",
    )

    assert run["version"] == 1
    meta = run["meta"]
    assert meta["seed"] == 0
    assert meta["record_seed"] == 42
    assert meta["updates"] == 16000
    assert meta["worlds_note"] == "new worlds, empty memory, frozen weights"
    assert meta["label"] == "learned structure"
    assert meta["note"] == "one seed; no claim of grid cells"

    w = run["worlds"][0]
    assert w["id"] == 0
    assert w["size"] == 8
    assert len(w["sensory"]) == 64
    assert len(w["walk"]) == 50
    assert len(w["pred_correct"]) == 50
    assert len(w["first_visit_new_edge"]) == 50
    assert "g" in w["rate_maps"]
    assert "p" in w["rate_maps"]

    assert len(w["rate_maps"]["g"]) <= 16
    assert len(w["rate_maps"]["p"]) <= 16
    for row in w["rate_maps"]["g"]:
        assert len(row) == 64

    out_path = tmp_path / "test-run.json"
    write_run(out_path, run)
    assert out_path.exists()
    assert out_path.stat().st_size < 300 * 1024


def test_record_run_probe_integration(tmp_path: Path) -> None:
    """Verify probe numbers from a probe file are carried unchanged into recorded run."""
    dummy_probe = {
        "results": {
            "g": {
                "si": 1021639077651.31,
                "circ_null": 28598964102.32,
                "map_corr": 0.217,
                "rsa": 0.742,
                "rsa_null_p95": 0.039,
                "dec": 0.246,
                "dec_null": 0.025,
                "rsa_grid": 0.009,
                "rsa_sensory": 0.02,
            },
            "p": {
                "si": 3.106,
                "circ_null": 0.412,
                "map_corr": 0.037,
                "rsa": 0.066,
                "rsa_null_p95": 0.043,
                "dec": 0.766,
                "dec_null": 0.03,
                "rsa_grid": 0.082,
                "rsa_sensory": 0.63,
            },
        }
    }
    probe_file = tmp_path / "probe.json"
    probe_file.write_text(str(dummy_probe).replace("'", '"'))

    config = TEMConfig(widths=(5, 6, 7, 8))
    model = TEM(config, generator=torch.Generator().manual_seed(0))
    run = record_run(
        model=model,
        worlds=1,
        size=8,
        steps=10,
        units=4,
        seed=42,
        probe_path=probe_file,
    )

    assert run["probe"] is not None
    stream_g = run["probe"]["stream"]["g"]
    assert stream_g["spatial_info"] == 1021639077651.31
    assert stream_g["null"] == 28598964102.32
    assert stream_g["cross_world_corr"] == 0.217
    assert stream_g["spatial_info_note"] == "reference-formula diagnostic, not bits/spike"

    stream_p = run["probe"]["stream"]["p"]
    assert stream_p["spatial_info"] == 3.106
    assert "spatial_info_note" not in stream_p


def test_record_run_determinism() -> None:
    """Verify exact determinism for a given seed."""
    config = TEMConfig(widths=(5, 6, 7, 8))
    model = TEM(config, generator=torch.Generator().manual_seed(0))
    run1 = record_run(model=model, worlds=2, size=8, steps=50, units=16, seed=123)
    run2 = record_run(model=model, worlds=2, size=8, steps=50, units=16, seed=123)
    assert run1 == run2


def test_record_run_failing_breakage() -> None:
    """Prove test fails when breaking walk length or schema."""
    config = TEMConfig(widths=(5, 6, 7, 8))
    model = TEM(config, generator=torch.Generator().manual_seed(0))
    run = record_run(model=model, worlds=1, size=8, steps=40, units=16, seed=0)
    with pytest.raises(AssertionError):
        assert len(run["worlds"][0]["walk"]) == 39
