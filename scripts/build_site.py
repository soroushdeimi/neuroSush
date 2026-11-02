import argparse
import html
import shutil
from pathlib import Path

from neurosush import __version__
from neurosush.viz.catalog import PLATES
from neurosush.viz.export import available_plates, render_plate


def favicon_links(prefix: str = "") -> str:
    """Use browser colour preference independently of the page theme."""
    links = []
    for theme in ("light", "dark"):
        base = f"{prefix}assets/logo/{theme}/"
        media = f"(prefers-color-scheme: {theme})"
        links.append(f'<link rel="icon" href="{base}favicon.ico" sizes="any" media="{media}">')
        for size in (32, 16):
            links.append(
                f'<link rel="icon" href="{base}mark-{size}.png" type="image/png" '
                f'sizes="{size}x{size}" media="{media}">'
            )
        links.append(
            f'<link rel="apple-touch-icon" href="{base}mark-180.png" '
            f'sizes="180x180" media="{media}">'
        )
    return "".join(links)


def brand_mark(prefix: str = "") -> str:
    return "".join(
        f'<img class="brand-mark brand-mark-{theme}" '
        f'src="{prefix}assets/logo/{theme}/transparent-512.png" '
        'width="28" height="28" alt="">'
        for theme in ("dark", "light")
    )


def chrome(prefix: str = "") -> tuple[str, str]:
    repo = "https://github.com/soroushdeimi/neuroSush"
    home = prefix + "index.html"
    brand = brand_mark(prefix) + "<span>neuroSush</span>"
    header = (
        '<a class="skip" href="#main">Skip to content</a>'
        '<header class="site-header"><a class="brand" href="' + home + '">' + brand + "</a>"
        '<a class="version" href="' + repo + '/blob/main/CHANGELOG.md">v' + __version__ + "</a>"
        '<nav id="main-nav" aria-label="Main">'
        '<a href="' + home + '#workflow">Library</a>'
        '<a href="' + prefix + 'gallery.html">Atlas</a>'
        '<a href="' + prefix + 'research.html">Research</a>'
        '<a href="' + prefix + 'results.html">Results</a>'
        '<a href="' + prefix + 'studio-live.html">Studio Live</a>'
        '<a href="' + prefix + 'tem.html">Live TEM</a>'
        '<a href="' + home + '#studio">Studio <span class="chip">preview</span></a>'
        '<a href="' + home + '#evidence">Benchmarks</a>'
        '<a href="' + repo + '/tree/main/docs">Docs ↗</a></nav>'
        '<div class="header-tools"><a class="icon-button" href="' + repo + '" '
        'aria-label="GitHub repository" title="GitHub repository">'
        '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 19c-5 1-5-3-7-3m'
        "14 6v-4c0-1-.3-2-1-2 4 0 6-2 6-6 0-2-1-3-2-4 0-1 0-2-1-3-2 0-3 1-4 2h-4C"
        '9 4 8 3 6 3 5 4 5 5 5 6c-1 1-2 2-2 4 0 4 2 6 6 6-.7.5-1 1-1 2v4"/></svg>'
        "</a>"
        '<button id="theme" class="icon-button" aria-label="Switch to light theme"'
        ' title="Change theme">'
        '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 15A9 9 0 0 1 9 '
        '4a9 9 0 1 0 11 11Z"/></svg></button>'
        '<button id="motion" class="icon-button" aria-label="Reduce motion" aria-'
        'pressed="false" title="Reduce motion">'
        '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14"/></svg></button>'
        '<a class="button" href="' + home + '#install">Install</a>'
        '<button id="menu" class="icon-button" aria-label="Open navigation" aria-'
        'expanded="false" aria-controls="main-nav">'
        '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18M3 12h18M3 1'
        '8h18"/></svg></button></div></header>'
    )
    footer = (
        '<footer class="site-footer"><div class="footer-grid"><div><a class="brand" href="'
        + home
        + '">'
        + brand
        + "</a>"
        "<p>Spiking networks, HTM / Thousand Brains and predictive coding on PyTo"
        "rch.</p><small>MIT licence</small></div>"
        '<nav aria-label="Library"><strong>Library</strong><a href="'
        + home
        + '#install">Install</a>'
        '<a href="'
        + repo
        + '/tree/main/docs">API docs ↗</a><a href="'
        + repo
        + '/tree/main/examples">Examples ↗</a>'
        '<a href="' + repo + '/blob/main/CHANGELOG.md">Changelog ↗</a></nav>'
        '<nav aria-label="Atlas"><strong>Atlas</strong><a href="'
        + prefix
        + 'gallery.html">All plates</a>'
        '<a href="' + prefix + 'research.html">Research</a>'
        '<a href="' + prefix + 'results.html">Results</a>'
        '<a href="' + prefix + 'tem.html">Live TEM</a>'
        '<a href="'
        + prefix
        + 'gallery.html#neurons">Neurons</a><a href="'
        + prefix
        + 'gallery.html#synapses">Synapses</a>'
        '<a href="'
        + prefix
        + 'gallery.html#htm">HTM</a><a href="'
        + prefix
        + 'gallery.html#maths">Maths</a></nav>'
        '<nav aria-label="Project"><strong>Project</strong><a href="' + repo + '">GitHub ↗</a>'
        '<a href="'
        + repo
        + '/blob/main/CITATION.cff">Citation ↗</a><a href="'
        + repo
        + '/blob/main/docs/BENCHMARKS.md">Benchmark method ↗</a>'
        '<a href="' + repo + '/issues">Report an issue ↗</a></nav></div>'
        '<div class="footer-meta">MIT licence · Cite via CITATION.cff: neuroSush:'
        " brain-inspired computing on PyTorch, v" + __version__ + "</div></footer>"
    )
    return header, footer


def build_gallery(plates: list[tuple[str, str]], version: str) -> str:
    page = build_index(plates, version)
    head = page[: page.index("<body>")]
    header, footer = chrome()
    catalog_start = page.index('<section class="catalog ')
    catalog = page[catalog_start : page.index("</section>", catalog_start) + 10]
    return (
        head.replace("neuroSush — Neural circuits, written in PyTorch.", "neuroSush Atlas")
        + '<body class="atlas-page">'
        + header
        + '<main id="main">'
        + catalog.replace("Teaching plates you can perturb.", "Atlas")
        + "</main>"
        + footer
        + "</body></html>"
    )


_ASSETS = Path(__file__).resolve().parent / "site_assets"


def build_index(plates: list[tuple[str, str]], version: str) -> str:
    """Render the landing template with links to every shipped plate."""
    template = (_ASSETS / "index.html").read_text(encoding="utf-8")
    cards = "\n".join(
        f'<li><a href="plates/{html.escape(pid, quote=True)}.html">'
        f'<span class="id">{html.escape(pid)}</span>{html.escape(title)}</a></li>'
        for pid, title in plates
    )
    featured = {"N01": "neurons", "N05": "wave", "Y01": "stdp", "H01": "sdr"}
    previews = "\n".join(
        f'<a class="preview" href="plates/{html.escape(pid, quote=True)}.html">'
        f'<canvas data-art="{featured[pid]}" aria-label="Illustrative plate preview"></canvas>'
        f"<div><small>{html.escape(pid)} / LIVE PLATE ↗</small>"
        f"<h3>{html.escape(title)}</h3></div></a>"
        for pid, title in plates
        if pid in featured
    )
    return (
        template.replace("{{ICONS}}", favicon_links())
        .replace("{{VERSION}}", html.escape(version))
        .replace("{{HEADER}}", chrome()[0])
        .replace("{{FOOTER}}", chrome()[1])
        .replace("{{PREVIEWS}}", previews)
        .replace("{{PLATES}}", cards)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="site", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plates
    plates_dir = out_dir / "plates"
    plates_dir.mkdir(parents=True, exist_ok=True)

    all_plates = available_plates()
    for plate_id in all_plates:
        if plate_id == "demo":
            continue

        content = render_plate(plate_id, None, source="live", theme="dark")
        header, footer = chrome("../")
        content = (
            content.replace(
                "</head>",
                favicon_links("../") + '<link rel="stylesheet" href="../assets/site.css">'
                '<script defer src="../assets/site.js"></script></head>',
            )
            .replace("<body>", '<body class="atlas-page">' + header + '<main id="main">')
            .replace("</body>", "</main>" + footer + "</body>")
        )
        (plates_dir / f"{plate_id}.html").write_text(content, encoding="utf-8")

    # 2. Demo copy
    demo_dir = out_dir / "demos" / "the-neuron"
    demo_dir.mkdir(parents=True, exist_ok=True)
    demo = Path("demos/the-neuron/index.html").read_text(encoding="utf-8")
    (demo_dir / "index.html").write_text(
        demo.replace("</head>", favicon_links("../../") + "</head>"), encoding="utf-8"
    )

    # 3. .nojekyll
    (out_dir / ".nojekyll").touch()

    # 4. 404.html
    (out_dir / "404.html").write_text(
        '<!DOCTYPE html><html data-theme="dark"><head>'
        + favicon_links()
        + "</head><body><h1>404</h1>"
        '<a href="index.html">Back to home</a></body></html>',
        encoding="utf-8",
    )

    # 5. Landing assets and gallery
    shutil.copytree(_ASSETS, out_dir / "assets", dirs_exist_ok=True)
    (out_dir / "assets" / "index.html").unlink(missing_ok=True)
    (out_dir / "assets" / "results.html").unlink(missing_ok=True)
    (out_dir / "assets" / "research.html").unlink(missing_ok=True)
    (out_dir / "assets" / "studio-live.html").unlink(missing_ok=True)

    # 6. Index, Results, Research & Studio Live
    (out_dir / "assets" / "tem.html").unlink(missing_ok=True)

    # 6. Data copy
    site_data_dir = _ASSETS / "data"
    if site_data_dir.exists():
        data_dir = out_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for json_file in site_data_dir.glob("*.json"):
            if json_file.is_file():
                shutil.copy(json_file, data_dir / json_file.name)

    # 7. Index, Results, Research & Live TEM
    index_content = build_index(
        [(p, PLATES[p].title_en if p in PLATES else p) for p in all_plates if p != "demo"],
        __version__,
    )
    (out_dir / "index.html").write_text(index_content, encoding="utf-8")
    (out_dir / "gallery.html").write_text(
        build_gallery(
            [(p, PLATES[p].title_en if p in PLATES else p) for p in all_plates if p != "demo"],
            __version__,
        ),
        encoding="utf-8",
    )
    results_template = (_ASSETS / "results.html").read_text(encoding="utf-8")
    r_header, r_footer = chrome("")
    results_content = (
        results_template.replace("{{ICONS}}", favicon_links())
        .replace("{{VERSION}}", html.escape(__version__))
        .replace("{{HEADER}}", r_header)
        .replace("{{FOOTER}}", r_footer)
    )
    (out_dir / "results.html").write_text(results_content, encoding="utf-8")

    research_template = (_ASSETS / "research.html").read_text(encoding="utf-8")
    rs_header, rs_footer = chrome("")
    research_content = (
        research_template.replace("{{ICONS}}", favicon_links())
        .replace("{{VERSION}}", html.escape(__version__))
        .replace("{{HEADER}}", rs_header)
        .replace("{{FOOTER}}", rs_footer)
    )
    (out_dir / "research.html").write_text(research_content, encoding="utf-8")

    studio_live_template = (_ASSETS / "studio-live.html").read_text(encoding="utf-8")
    sl_header, sl_footer = chrome("")
    studio_live_content = (
        studio_live_template.replace("{{ICONS}}", favicon_links())
        .replace("{{VERSION}}", html.escape(__version__))
        .replace("{{HEADER}}", sl_header)
        .replace("{{FOOTER}}", sl_footer)
    )
    (out_dir / "studio-live.html").write_text(studio_live_content, encoding="utf-8")
    tem_template = (_ASSETS / "tem.html").read_text(encoding="utf-8")
    t_header, t_footer = chrome("")
    tem_content = (
        tem_template.replace("{{ICONS}}", favicon_links())
        .replace("{{VERSION}}", html.escape(__version__))
        .replace("{{HEADER}}", t_header)
        .replace("{{FOOTER}}", t_footer)
    )
    (out_dir / "tem.html").write_text(tem_content, encoding="utf-8")


if __name__ == "__main__":
    main()
