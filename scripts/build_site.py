import argparse
import html
import shutil
from pathlib import Path

from neurosush import __version__
from neurosush.viz.catalog import PLATES
from neurosush.viz.export import available_plates, render_plate

_CSS = """
:root{
--bg:#fff;
--fg:#1b1f24;
--muted:#57606a;
--card:#f6f8fa;
--line:#d0d7de;
--accent:#0b6bcb}
@media (prefers-color-scheme:dark){:root{
--bg:#0d1117;
--fg:#e6edf3;
--muted:#9da7b3;
--card:#161b22;
--line:#30363d;
--accent:#58a6ff}
}
*{box-sizing:border-box}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);
font:16px/1.5 system-ui,sans-serif}
main{max-width:960px;margin:0 auto}
h1{margin:0 0 4px}
p.lead{color:var(--muted);margin:0 0 24px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;padding:0;list-style:none}
.grid a{display:block;height:100%;padding:12px;background:var(--card);
border:1px solid var(--line);border-radius:8px;color:var(--fg);text-decoration:none}
.grid a:hover,.grid a:focus-visible{border-color:var(--accent)}
.id{font:600 13px ui-monospace,monospace;color:var(--accent)}
h2{margin-top:32px}
"""


def build_index(plates: list[tuple[str, str]], version: str) -> str:
    """Return the gallery page; ``plates`` is a list of (id, title) pairs."""
    cards = "\n".join(
        f'<li><a href="plates/{html.escape(pid)}.html"><span class="id">{html.escape(pid)}</span>'
        f"<br>{html.escape(title)}</a></li>"
        for pid, title in plates
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>neuroSush Atlas</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>neuroSush Atlas</h1>
<p class="lead">Animated plates for spiking neurons, built from the neuroSush library
(version {html.escape(version)}).</p>
<h2>Talk demo</h2>
<ul class="grid"><li><a href="demos/the-neuron/index.html"><span class="id">LIVE</span><br>
The neuron: bucket model and Hodgkin-Huxley gates</a></li></ul>
<h2>Plates</h2>
<ul class="grid">
{cards}
</ul>
</main>
</body>
</html>
"""


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

    # 5. Index
    index_content = build_index(
        [(p, PLATES[p].title_en if p in PLATES else p) for p in all_plates if p != "demo"],
        __version__,
    )
    (out_dir / "index.html").write_text(index_content, encoding="utf-8")


if __name__ == "__main__":
    main()
