"""T02 teaching plate catalog, Python reference and standalone export contracts."""

import json
from pathlib import Path

from scripts.generate_tem_t02_golden import generate

from neurosush.viz.catalog import PLATES, public_models
from neurosush.viz.export import available_plates, render_plate


def test_t02_catalog_and_export():
    spec = PLATES["T02"]
    assert spec.covers == ("neurosush.memory.hebbian.HebbianMemory", "neurosush.tem.model.TEM")
    assert set(spec.covers) <= set(public_models())
    assert "T02" in available_plates()
    page = render_plate("T02", source="live")
    for text in ("TEACHING: structure imposed", "Same map, new world", "docs/TEM.md"):
        assert text in page
    plate = Path("src/neurosush/viz/web/plates/T02.js").read_bytes()
    assert len(plate) < 15_000


def test_t02_golden_is_generated_from_library():
    fixture = json.loads(Path("tests/golden/tem_t02.json").read_text(encoding="utf-8"))
    expected = generate()

    def compare(actual, reference):
        if isinstance(reference, dict):
            assert actual.keys() == reference.keys()
            for key in reference:
                compare(actual[key], reference[key])
        elif isinstance(reference, list):
            assert len(actual) == len(reference)
            for a, b in zip(actual, reference, strict=True):
                compare(a, b)
        elif isinstance(reference, float):
            assert abs(actual - reference) <= fixture["atol"]
        else:
            assert actual == reference

    compare(fixture, expected)
