import argparse
import html
import shutil
from pathlib import Path

from neurosush import __version__
from neurosush.viz.catalog import PLATES
from neurosush.viz.export import available_plates, render_plate


def chrome(prefix: str = "") -> tuple[str, str]:
    header = (
        '<a class="skip" href="#main">Skip to content</a>'
        '<header class="site-header"><a class="brand" href="'
        + prefix
        + 'index.html">neuroSush<small>PYTORCH / ATLAS</small></a>'
        '<nav aria-label="Main"><a href="' + prefix + 'gallery.html">Atlas</a>'
        '<a href="' + prefix + 'research.html">Research</a>'
        '<a href="' + prefix + 'results.html">Results</a>'
        '<a class="repo" href="https://github.com/soroushdeimi/neuroSush">GitHub ↗</a>'
        '<button id="theme" aria-label="Switch to light theme">◐ Theme</button></nav></header>'
    )
    footer = (
        '<footer class="site-footer"><span>neuroSush / Brain-inspired. Tensor-native. / MIT</span>'
        '<nav aria-label="Footer"><a href="'
        + prefix
        + 'index.html">Home</a><a href="'
        + prefix
        + 'gallery.html">Atlas</a><a href="'
        + prefix
        + 'research.html">Research</a><a href="'
        + prefix
        + 'results.html">Results</a>'
        '<a href="https://github.com/soroushdeimi/neuroSush/tree/main/docs">Docs ↗</a>'
        "</nav></footer>"
    )
    return header, footer


def build_gallery(plates: list[tuple[str, str]], version: str) -> str:
    page = build_index(plates, version)
    head = page[: page.index("<body>")]
    header, footer = chrome()
    catalog = page[page.index('<section class="catalog"') : page.index("</main>")]
    return (
        head.replace("neuroSush — Give your ideas a nervous system.", "neuroSush Atlas")
        + '<body class="atlas-page">'
        + header
        + '<main id="main">'
        + catalog.replace("A question worth running.", "The living Atlas.")
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
        template.replace("{{VERSION}}", html.escape(version))
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
    # available_plates() returns sorted list.
    for plate_id in all_plates:
        if plate_id == "demo":
            continue

        content = render_plate(plate_id, None, source="live")
        header, footer = chrome("../")
        content = (
            content.replace(
                "</head>",
                '<link rel="icon" href="../assets/mark.svg">'
                '<link rel="stylesheet" href="../assets/site.css">'
                '<script defer src="../assets/site.js"></script></head>',
            )
            .replace("<body>", '<body class="atlas-page">' + header + '<main id="main">')
            .replace("</body>", "</main>" + footer + "</body>")
        )
        (plates_dir / f"{plate_id}.html").write_text(content, encoding="utf-8")

    # 2. Demo copy
    demo_dir = out_dir / "demos" / "the-neuron"
    demo_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("demos/the-neuron/index.html", demo_dir / "index.html")

    # 3. .nojekyll
    (out_dir / ".nojekyll").touch()

    # 4. 404.html
    (out_dir / "404.html").write_text(
        "<!DOCTYPE html><html><body><h1>404</h1>"
        '<a href="index.html">Back to home</a></body></html>',
        encoding="utf-8",
    )

    # 5. Landing assets and the original gallery
    shutil.copytree(_ASSETS, out_dir / "assets", dirs_exist_ok=True)
    (out_dir / "assets" / "index.html").unlink(missing_ok=True)
    (out_dir / "assets" / "results.html").unlink(missing_ok=True)
    (out_dir / "assets" / "research.html").unlink(missing_ok=True)

    # 6. Index, Results & Research
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
        results_template.replace("{{VERSION}}", html.escape(__version__))
        .replace("{{HEADER}}", r_header)
        .replace("{{FOOTER}}", r_footer)
    )
    (out_dir / "results.html").write_text(results_content, encoding="utf-8")

    research_template = (_ASSETS / "research.html").read_text(encoding="utf-8")
    rs_header, rs_footer = chrome("")
    research_content = (
        research_template.replace("{{VERSION}}", html.escape(__version__))
        .replace("{{HEADER}}", rs_header)
        .replace("{{FOOTER}}", rs_footer)
    )
    (out_dir / "research.html").write_text(research_content, encoding="utf-8")


if __name__ == "__main__":
    main()
