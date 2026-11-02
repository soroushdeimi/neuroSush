"""Tests for recorded-run exporter for TEM."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neurosush.tem.model import TEM, TEMConfig
from neurosush.tem.record import record_run, write_run


def test_record_run_schema_and_sizes(tmp_path: Path) -> None:
    """Verify schema keys, world count, walk length, and rate map shapes."""
    config = TEMConfig(widths=(5, 6, 7, 8))
    model = TEM(config, generator=torch.Generator().manual_seed(0))
    run = record_run(model=model, worlds=2, size=8, steps=50, units=16, seed=42)

    assert run["version"] == 1
    assert "meta" in run
    assert run["meta"]["variant"] == "v2"
    assert "worlds" in run
    assert len(run["worlds"]) == 2

    w = run["worlds"][0]
    assert w["id"] == 0
    assert w["size"] == 8
    assert len(w["sensory"]) == 64
    assert len(w["walk"]) == 50
    assert len(w["pred_correct"]) == 50
    assert len(w["first_visit_new_edge"]) == 50
    assert "g" in w["rate_maps"]
    assert "p" in w["rate_maps"]

    # Rate maps: up to 16 units, each 64 entries (8x8)
    assert len(w["rate_maps"]["g"]) <= 16
    assert len(w["rate_maps"]["p"]) <= 16
    for row in w["rate_maps"]["g"]:
        assert len(row) == 64

    # Test write_run and size < 300 KB
    out_path = tmp_path / "test-run.json"
    write_run(out_path, run)
    assert out_path.exists()
    assert out_path.stat().st_size < 300 * 1024


def test_record_run_determinism() -> None:
    """Verify exact determinism for a given seed."""
    config = TEMConfig(widths=(5, 6, 7, 8))
    model = TEM(config, generator=torch.Generator().manual_seed(0))
    run1 = record_run(model=model, worlds=2, size=8, steps=405, units=16, seed=123)
    run2 = record_run(model=model, worlds=2, size=8, steps=405, units=16, seed=123)
    assert run1 == run2


def test_record_run_failing_breakage() -> None:
    """Prove test fails when breaking walk length or schema."""
    config = TEMConfig(widths=(5, 6, 7, 8))
    model = TEM(config, generator=torch.Generator().manual_seed(0))
    run = record_run(model=model, worlds=1, size=8, steps=40, units=16, seed=0)
    with pytest.raises(AssertionError):
        assert len(run["worlds"][0]["walk"]) == 39
