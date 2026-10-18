"""Tests for neurosush.viz.export (self-contained HTML exporter)."""

from __future__ import annotations

import array
import base64
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from neurosush.viz.export import available_plates, render_plate, write_html


def _make_dummy_trace(steps: int = 50) -> dict[str, Any]:
    v = [-65.0 + 0.2 * i for i in range(steps)]
    spikes = [1 if i % 15 == 0 else 0 for i in range(steps)]
    v_arr = array.array("f", v)
    spk_arr = array.array("B", spikes)
    return {
        "steps": steps,
        "meta": {"version": "0.6.0", "threshold": -50.0, "theory": True},
        "series": {
            "v": {
                "dtype": "float32",
                "shape": [steps],
                "data": base64.b64encode(v_arr.tobytes()).decode("ascii"),
            },
            "spikes": {
                "dtype": "uint8",
                "shape": [steps],
                "data": base64.b64encode(spk_arr.tobytes()).decode("ascii"),
            },
            "v_theory": {
                "dtype": "float32",
                "shape": [steps],
                "data": base64.b64encode(v_arr.tobytes()).decode("ascii"),
            },
        },
    }


class _StrictHTMLValidator(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.scripts: dict[str, str] = {}
        self._current_script_id: str | None = None
        self._current_script_data: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        attr_dict = dict(attrs)
        if tag == "script" and attr_dict.get("type") == "application/json":
            self._current_script_id = attr_dict.get("id")
            self._current_script_data = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._current_script_id:
            self.scripts[self._current_script_id] = "".join(self._current_script_data)
            self._current_script_id = None
            self._current_script_data = []

    def handle_data(self, data: str) -> None:
        if self._current_script_id:
            self._current_script_data.append(data)


def test_available_plates() -> None:
    plates = available_plates()
    assert isinstance(plates, list)
    assert "demo" in plates


def test_render_plate_no_external_urls() -> None:
    trace = _make_dummy_trace()
    html_doc = render_plate("demo", trace)

    # Must contain no external URLs
    forbidden_patterns = [
        re.compile(r"https?://", re.IGNORECASE),
        re.compile(r'src=["\']//', re.IGNORECASE),
        re.compile(r'href=["\']//', re.IGNORECASE),
    ]
    for pattern in forbidden_patterns:
        matches = pattern.findall(html_doc)
        assert not matches, f"Found forbidden external URL reference: {matches}"


def test_render_plate_valid_html_structure() -> None:
    trace = _make_dummy_trace()
    html_doc = render_plate("demo", trace, title="Demo Membrane Plate")

    parser = _StrictHTMLValidator()
    parser.feed(html_doc)
    parser.close()

    assert "html" in parser.tags
    assert "head" in parser.tags
    assert "body" in parser.tags
    assert "style" in parser.tags
    assert "script" in parser.tags


def test_render_plate_trace_round_trip() -> None:
    trace = _make_dummy_trace(steps=80)
    html_doc = render_plate("demo", trace)

    parser = _StrictHTMLValidator()
    parser.feed(html_doc)
    parser.close()

    assert "atlas-trace" in parser.scripts
    embedded_raw = parser.scripts["atlas-trace"]
    round_tripped = json.loads(embedded_raw)

    assert round_tripped["steps"] == trace["steps"]
    assert round_tripped["meta"] == trace["meta"]
    assert round_tripped["series"]["v"]["data"] == trace["series"]["v"]["data"]


def test_render_plate_hostile_strings_escaped() -> None:
    trace = _make_dummy_trace()
    hostile_title = "<script>alert('pwned')</script>"
    hostile_notes = {"danger": "</script><script>alert('pwned')</script>"}

    html_doc = render_plate(
        "demo",
        trace,
        title=hostile_title,
        notes=hostile_notes,
    )

    # Title must be HTML escaped
    assert "<title>&lt;script&gt;alert(&#x27;pwned&#x27;)&lt;/script&gt;</title>" in html_doc
    assert "<title><script>alert" not in html_doc

    # Notes script block must have safe JSON escaping (no raw </script>)
    parser = _StrictHTMLValidator()
    parser.feed(html_doc)
    parser.close()

    assert "atlas-notes" in parser.scripts
    notes_parsed = json.loads(parser.scripts["atlas-notes"])
    assert notes_parsed["danger"] == hostile_notes["danger"]


def test_render_plate_unknown_plate_raises() -> None:
    trace = _make_dummy_trace()
    with pytest.raises(FileNotFoundError) as exc_info:
        render_plate("non_existent_plate_xyz", trace)

    err_msg = str(exc_info.value)
    assert "non_existent_plate_xyz" in err_msg
    assert "demo" in err_msg


def test_render_plate_always_lang_en_ltr() -> None:
    trace = _make_dummy_trace()
    html_doc = render_plate("demo", trace)

    assert '<html lang="en" dir="ltr"' in html_doc


def test_render_plate_invalid_source_raises() -> None:
    trace = _make_dummy_trace()
    with pytest.raises(ValueError, match="source must be one of"):
        render_plate("demo", trace, source="invalid_source")


def test_write_html(tmp_path: Path) -> None:
    trace = _make_dummy_trace()
    target = tmp_path / "subdir" / "plate_demo.html"
    returned_path = write_html(target, "demo", trace)

    assert returned_path == target
    assert target.is_file()
    content = target.read_text(encoding="utf-8")
    assert '<div id="atlas-root"></div>' in content
    assert "<style>" in content


def test_chrome_rendered_controls_match_strings(tmp_path: Path) -> None:
    chrome_bin = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome_bin:
        pytest.skip("Chrome binary is not found on PATH")

    trace = _make_dummy_trace()
    html_path = write_html(tmp_path / "chrome_test.html", "demo", trace)

    result = subprocess.run(
        [chrome_bin, "--headless=new", "--disable-gpu", "--dump-dom", f"file://{html_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Chrome dump-dom failed: {result.stderr}"
    rendered_dom = result.stdout

    # Assert all core control labels and badges are present in the rendered DOM
    expected_tokens = [
        "PLATE DEMO",
        "Theme: auto",
        "Play",
        "Loop",
        "Fork",
        "Theory",
        "Mute",
        "Source",
        "Replay",
        "neuroSush",
        "Toggle loop",
        "Fork the run [F]",
        "Mute sound [M]",
    ]
    for token in expected_tokens:
        assert token in rendered_dom, f"Expected token {token!r} missing from rendered DOM"
