"""TEM teaching plate catalog, Python reference and standalone export contracts."""

import json
from pathlib import Path

from scripts.generate_tem_plate_golden import generate

from neurosush.viz.catalog import PLATES, public_models
from neurosush.viz.export import available_plates, render_plate


def test_tem_catalog_and_export():
    spec = PLATES["T01"]
    assert spec.covers == ("neurosush.memory.hebbian.HebbianMemory", "neurosush.tem.model.TEM")
    assert set(spec.covers) <= set(public_models())
    assert "T01" in available_plates()
    page = render_plate("T01", source="live")
    for text in ("TEACHING", "GIVEN / imposed", "not the trained TEM", "docs/TEM.md"):
        assert text in page
    plate = Path("src/neurosush/viz/web/plates/T01.js").read_bytes()
    assert len(plate) < 150_000


def test_tem_golden_is_generated_from_library():
    fixture = json.loads(Path("tests/golden/tem_plate.json").read_text(encoding="utf-8"))
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
