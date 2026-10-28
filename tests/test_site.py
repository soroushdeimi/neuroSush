import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_site import build_index, main

from neurosush.viz.export import available_plates


def _build(out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["build_site.py", "--out", str(out)])
    main()


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_build_writes_every_plate_and_support_files(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    for pid in available_plates():
        assert (tmp_path / "plates" / f"{pid}.html").exists() == (pid != "demo")
    for rel in ("index.html", "404.html", ".nojekyll", "demos/the-neuron/index.html"):
        assert (tmp_path / rel).exists(), rel
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    for pid in available_plates():
        if pid != "demo":
            assert f'plates/{pid}.html"' in index
    assert "demos/the-neuron/index.html" in index


def test_build_is_deterministic(tmp_path, monkeypatch):
    _build(tmp_path / "a", monkeypatch)
    _build(tmp_path / "b", monkeypatch)
    assert _files(tmp_path / "a") == _files(tmp_path / "b")


def test_pages_make_no_external_requests(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    pattern = re.compile(
        r"""(?:src|href)\s*=\s*["']https?://|url\(\s*["']?https?://|@import\s+["']?https?://"""
    )
    for path in tmp_path.rglob("*.html"):
        assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_index_escapes_titles():
    page = build_index([("X1", "<script>alert(1)</script> & co")], "1.0")
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page
    assert "&amp; co" in page


def test_pages_workflow_deploys_the_built_site():
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/pages.yml").read_text(
        encoding="utf-8"
    )
    assert "scripts/build_site.py" in text
    assert "path: site" in text
    assert "actions/deploy-pages" in text
