import re
import sys
from html.parser import HTMLParser
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
    for rel in (
        "index.html",
        "gallery.html",
        "assets/site.css",
        "assets/site.js",
        "404.html",
        ".nojekyll",
        "demos/the-neuron/index.html",
    ):
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

    class Resources(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if value and (key == "src" or (key == "href" and tag == "link")):
                    assert not value.startswith(("http:", "https:", "//")), value

    pattern = re.compile(r"""url\(\s*["']?https?://|@import\s+["']?https?://""")
    for path in tmp_path.rglob("*.html"):
        content = path.read_text(encoding="utf-8")
        Resources().feed(content)
        assert not pattern.search(content), path
    assets = (tmp_path / "assets" / "site.js").read_text(encoding="utf-8")
    assert not re.search(r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\(", assets)


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


def test_landing_links_resolve_and_template_is_complete(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "{{" not in page
    ids = set(re.findall(r'id="([^"]+)"', page))
    for href in re.findall(r'href="([^"]+)"', page):
        if href.startswith("#"):
            assert href == "#" or href[1:] in ids
        elif not href.startswith("https://"):
            assert (tmp_path / href.split("#")[0]).is_file(), href
    assert 'data-theme="light"' in page
    assert "not yet in a tagged release" in page
    assert "not a reconstruction" in page


def test_shared_shell_and_scroll_story(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    index = (tmp_path / "index.html").read_text()
    assert index.count('<section class="chapter') == 6
    assert 'aria-hidden="true"' in index
    for rel in ("index.html", "gallery.html", "plates/N01.html", "plates/Y01.html"):
        page = (tmp_path / rel).read_text()
        assert 'class="site-header"' in page
        assert 'class="site-footer"' in page
        assert "assets/site.css" in page
        assert "assets/site.js" in page
    script = (tmp_path / "assets/site.js").read_text()
    assert "webgl2" in script
    assert "visibilitychange" in script
    assert "prefers-reduced-motion" in script


def test_neuron_module_is_shipped(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    page = (tmp_path / "index.html").read_text()
    assert page.index("assets/neuron.js") < page.index("assets/site.js")
    module = (tmp_path / "assets/neuron.js").read_text()
    assert "NeuroShapes.neuron" in module
    assert "function branch" in module
    script = (tmp_path / "assets/site.js").read_text()
    assert "advanceLIF(dt)" in script
    assert "uniform.spikeAge" in script
