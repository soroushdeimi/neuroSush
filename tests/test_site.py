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
        "results.html",
        "research.html",
        "tem.html",
        "data/tem-run-v2-s0.json",
        "data/tem-run-v1-s0.json",
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
    assert "Results report" in page
    assert "Research story" in page
    assert "not a reconstruction" in page


def test_shared_shell_and_scroll_story(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    index = (tmp_path / "index.html").read_text()
    assert index.count('<section class="chapter') == 6
    assert 'aria-hidden="true"' in index
    for rel in (
        "index.html",
        "gallery.html",
        "results.html",
        "research.html",
        "tem.html",
        "plates/N01.html",
        "plates/Y01.html",
    ):
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


def test_foundation_and_instrument_assets(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    assets = tmp_path / "assets"
    for name in ("tokens.css", "components.css", "instruments.js"):
        assert (assets / name).is_file()
    tokens = (assets / "tokens.css").read_text()
    assert "#F4F1EA" in tokens
    assert "#0D0E0E" in tokens
    assert "font-display:swap" in tokens
    for font in (assets / "fonts").glob("*.woff2"):
        assert font.stat().st_size < 32768
    assert (assets / "fonts/SourceSerif4-OFL.txt").is_file()
    page = (tmp_path / "index.html").read_text()
    assert "stage-number" not in page
    assert 'id="motion"' in page


def test_synapse_geometry_is_shipped(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    page = (tmp_path / "index.html").read_text()
    assert page.index("assets/shape-kit.js") < page.index("assets/synapse.js")
    module = (tmp_path / "assets/synapse.js").read_text()
    assert "NeuroShapes.synapse" in module
    assert "Presynaptic bouton" in module
    assert "Spine head" in module


def test_microcircuit_has_layers_and_cell_types(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    module = (tmp_path / "assets/microcircuit.js").read_text()
    assert "inhibitory=i%5===0" in module
    assert "triangle" in module
    for layer in range(1, 7):
        assert f"'L{layer}'" in module


def test_sparse_geometry_uses_1024_minicolumns(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    module = (tmp_path / "assets/sparse.js").read_text()
    assert "row<32" in module
    assert "col<32" in module
    assert "Lateral voting" in module


def test_hierarchy_has_directional_flows(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    module = (tmp_path / "assets/hierarchy.js").read_text()
    assert "Predictions ↓" in module
    assert "Errors ↑" in module
    assert "layer<3" in module


def test_brain_has_folded_cortex_and_hindbrain(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    module = (tmp_path / "assets/brain.js").read_text()
    for anatomy in ("sulcus", "sylvian", "central", "Cerebellum", "Brainstem"):
        assert anatomy in module
    assert "Hippocampal formation" in module


def test_grid_lattice_is_explicit_and_separate(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    module = (tmp_path / "assets/grid.js").read_text()
    assert "Math.sqrt(3)/2" in module
    assert "Math.pow(1.4,module)" in module
    assert "not learned" in module
    assert "Illustration. This lattice is imposed by the scene code, not learned." in module


def test_tem_teaching_plate_is_published(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    page = (tmp_path / "plates/T01.html").read_text(encoding="utf-8")
    assert "TEACHING" in page
    assert "GIVEN / imposed" in page
    assert 'plates/T01.html"' in (tmp_path / "gallery.html").read_text(encoding="utf-8")


def test_results_report_page(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    results_path = tmp_path / "results.html"
    assert results_path.exists()
    page = results_path.read_text(encoding="utf-8")
    assert "0.1027" in page
    assert "structure imposed" in page
    assert "not a paper reproduction" in page.lower()
    assert "grid-cell emergence not shown" in page.lower()
    ids = set(re.findall(r'id="([^"]+)"', page))
    for href in re.findall(r'href="([^"]+)"', page):
        if href.startswith("#"):
            assert href == "#" or href[1:] in ids
        elif not href.startswith(("https://", "http://", "mailto:")):
            assert (tmp_path / href.split("#")[0]).is_file(), href


def test_research_report_page(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch)
    research_path = tmp_path / "research.html"
    assert research_path.exists()
    page = research_path.read_text(encoding="utf-8")
    assert "structure imposed" in page
    assert "not a paper reproduction" in page.lower()
    assert "grid-cell emergence not shown" in page.lower()
    ids = set(re.findall(r'id="([^"]+)"', page))
    for href in re.findall(r'href="([^"]+)"', page):
        if href.startswith("#"):
            assert href == "#" or href[1:] in ids
        elif not href.startswith(("https://", "http://", "mailto:")):
            assert (tmp_path / href.split("#")[0]).is_file(), href
