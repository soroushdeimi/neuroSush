import argparse
import html
import json
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
        '<a href="' + prefix + 'story/story.html">The Neuron, zoomed</a>'
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
        '<a href="' + prefix + 'gallery.html#foundations">Foundations</a>'
        '<a href="' + prefix + 'gallery.html#neurons">Neurons</a>'
        '<a href="' + prefix + 'gallery.html#synapses">Synapses</a>'
        '<a href="' + prefix + 'gallery.html#memory">Memory</a>'
        '<a href="' + prefix + 'gallery.html#research">Research</a>'
        '<a href="' + prefix + 'gallery.html#tools">Tools</a></nav>'
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


def build_gallery(version: str) -> str:
    atlas_data = json.loads((_ASSETS / "atlas.json").read_text(encoding="utf-8"))
    header, footer = chrome()

    step_html = []
    for step_item in atlas_data["path"]:
        pid = step_item["id"]
        why = step_item["why"]
        title = PLATES[pid].title_en if pid in PLATES else pid
        url = f"plates/{pid}.html"
        step_html.append(
            f'<a class="step-card" href="{url}">'
            f'<span class="step-num">Step {step_item["step"]}</span>'
            f'<span class="step-id">{html.escape(pid)}</span>'
            f"<strong>{html.escape(title)}</strong>"
            f"<p>{html.escape(why)}</p>"
            f"</a>"
        )

    group_chips = (
        '<button class="atlas-group-chip button" data-group="all" '
        'aria-pressed="true">All groups</button>'
    )
    for g in atlas_data["groups"]:
        group_chips += (
            f'<button class="atlas-group-chip button" data-group="{g["id"]}" '
            f'aria-pressed="false">{html.escape(g["title"])}</button>'
        )

    level_chips = (
        '<button class="atlas-level-chip button" data-level="all" '
        'aria-pressed="true">All levels</button>'
        '<button class="atlas-level-chip button" data-level="Beginner" '
        'aria-pressed="false">Beginner</button>'
        '<button class="atlas-level-chip button" data-level="Intermediate" '
        'aria-pressed="false">Intermediate</button>'
        '<button class="atlas-level-chip button" data-level="Advanced" '
        'aria-pressed="false">Advanced</button>'
    )

    group_sections = []
    items_by_group = {}
    for item in atlas_data["items"]:
        items_by_group.setdefault(item["group"], []).append(item)

    for g in atlas_data["groups"]:
        g_id = g["id"]
        g_items = items_by_group.get(g_id, [])
        if not g_items:
            continue
        cards_html = []
        for item in g_items:
            pid = item["id"]
            title = item.get("title") or (PLATES[pid].title_en if pid in PLATES else pid)
            url = item.get("url") or f"plates/{pid}.html"
            level = item["level"]
            minutes = item["minutes"]
            status = item["status"]
            learn = item["learn"]
            prereqs = item.get("prerequisite_ids", [])
            prereq_links = " · ".join(f'<a href="plates/{p}.html">{p}</a>' for p in prereqs)
            prereq_html = (
                f'<div class="prereq"><small>Prerequisite: {prereq_links}</small></div>'
                if prereqs
                else ""
            )
            tags_html = "".join(
                f'<span class="chip tag">{html.escape(t)}</span>' for t in item.get("tags", [])
            )

            cards_html.append(
                f'<div class="atlas-card" data-level="{html.escape(level)}" '
                f'data-group="{html.escape(g_id)}">'
                f'<span class="id">'
                f"{html.escape(pid if not item.get('url') else title[:3].upper())}"
                f"</span>"
                f'<h3><a href="{url}">{html.escape(title)}</a></h3>'
                f'<p class="learn">“{html.escape(learn)}”</p>'
                f'<div class="chips">'
                f'<span class="chip level">{html.escape(level)}</span>'
                f'<span class="chip time">{minutes} min</span>'
                f'<span class="chip status">{html.escape(status)}</span>'
                f"{tags_html}"
                f"</div>"
                f"{prereq_html}"
                f"</div>"
            )

        group_sections.append(
            f'<section class="atlas-group-section" id="{g["anchor"]}" data-group-id="{g_id}">'
            f"<h2>{html.escape(g['title'])}</h2>"
            f'<p class="group-subtitle">{html.escape(g["subtitle"])}</p>'
            f'<div class="grid">{"".join(cards_html)}</div>'
            f"</section>"
        )

    hero = (
        f'<section class="atlas-hero">\n'
        f'<p class="eyebrow">Atlas · v{html.escape(version)}</p>\n'
        f"<h1>Atlas of Neural Circuits & Models</h1>\n"
        f'<p class="lead">The neuroSush Atlas is a curated catalog of interactive '
        f"teaching plates and simulation models. Explore core equations, perturb "
        f"parameters, and trace neural dynamics from foundational biophysics to "
        f"network memory.</p>\n"
        f"</section>"
    )

    return (
        f"<!DOCTYPE html>\n"
        f'<html lang="en" data-theme="dark">\n'
        f"<head>\n"
        f'<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'<meta name="description" content="Atlas of neural circuits, simulation '
        f'plates and educational models in neuroSush.">\n'
        f"<title>neuroSush Atlas</title>\n"
        f"{favicon_links()}\n"
        f'<link rel="preload" href="assets/fonts/SourceSerif4-Display-Regular.woff2" '
        f'as="font" type="font/woff2" crossorigin>\n'
        f'<link rel="preload" href="assets/fonts/IBMPlexSans-Regular.woff2" '
        f'as="font" type="font/woff2" crossorigin>\n'
        f'<link rel="stylesheet" href="assets/site.css">\n'
        f'<script defer src="assets/site.js"></script>\n'
        f"</head>\n"
        f'<body class="atlas-page">\n'
        f"{header}\n"
        f'<main id="main">\n'
        f'<div class="atlas-container">\n'
        f"{hero}\n"
        f"\n"
        f'<section class="atlas-path-section">\n'
        f"<h2>Start here: a 4-step path</h2>\n"
        f'<p class="lead">Follow this guided sequence from membrane '
        f"biophysics to synaptic plasticity.</p>\n"
        f'<div class="stepper">\n'
        f"{''.join(step_html)}\n"
        f"</div>\n"
        f"</section>\n"
        f"\n"
        f'<section class="atlas-filter-bar">\n'
        f'<div class="search-box">\n'
        f'<label for="atlas-search">Search Atlas</label>\n'
        f'<input id="atlas-search" type="search" '
        f'placeholder="Search plates, equations, tags, models…">\n'
        f"</div>\n"
        f'<div class="filter-group-row">\n'
        f'<span class="filter-label">Group:</span>\n'
        f'<div class="chip-row">{group_chips}</div>\n'
        f"</div>\n"
        f'<div class="filter-level-row">\n'
        f'<span class="filter-label">Level:</span>\n'
        f'<div class="chip-row">{level_chips}</div>\n'
        f"</div>\n"
        f'<div id="result-count" class="result-count" role="status" '
        f'aria-live="polite">Showing {len(atlas_data["items"])} items</div>\n'
        f"</section>\n"
        f"\n"
        f'<div class="atlas-content">\n'
        f"{''.join(group_sections)}\n"
        f'<div id="empty-state" class="empty-state" hidden>\n'
        f"<h3>No matching plates or models found</h3>\n"
        f"<p>Try adjusting your search terms or clearing active "
        f"group and level filters.</p>\n"
        f"</div>\n"
        f"</div>\n"
        f"\n"
        f"</div>\n"
        f"</main>\n"
        f"{footer}\n"
        f"</body>\n"
        f"</html>\n"
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
    shutil.copy(_ASSETS / "atlas.json", out_dir / "assets" / "atlas.json")
    shutil.copytree(_ASSETS / "story", out_dir / "story", dirs_exist_ok=True)
    (out_dir / "assets" / "index.html").unlink(missing_ok=True)
    (out_dir / "assets" / "results.html").unlink(missing_ok=True)
    (out_dir / "assets" / "research.html").unlink(missing_ok=True)
    (out_dir / "assets" / "studio-live.html").unlink(missing_ok=True)
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
        build_gallery(__version__),
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
