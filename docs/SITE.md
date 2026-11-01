# Public site

The Pages workflow still runs `scripts/build_site.py` and publishes `site/`. The builder
renders `scripts/site_assets/index.html`, copies its local CSS, JavaScript and SVG mark,
and exports the existing plates and talk demo. `gallery.html` preserves the original index.
No npm build or additional Python dependency is required.

Preview from the repository root, without installing the package:

```bash
PYTHONPATH=src python3 scripts/build_site.py --out /tmp/neurosush-site-preview
python3 -m http.server 8765 --bind 127.0.0.1 --directory /tmp/neurosush-site-preview
```

Open `http://127.0.0.1:8765/`. The `PYTHONPATH` prefix makes the checkout take precedence
over any older installed version. CI already installs the checkout and uses the unchanged
`python scripts/build_site.py` command.

Validation:

```bash
python3 scripts/check_site.py /tmp/neurosush-site-preview
node --check scripts/site_assets/site.js
node scripts/check_site_browser.mjs /tmp/neurosush-site-shots
```

The browser check uses Node's built-in WebSocket client (Node 22+) and local Chrome via
CDP, with no npm dependencies. Set `CHROME` to a different Chrome/Chromium executable when
needed. Keep the preview server running. It blocks nonlocal page requests, fails on console
errors, tests search, themes and the LIF tuner, checks 390/1440/1920px layouts, emulates reduced
motion, and captures dark sections and light heroes. Normal project tests remain in
`tests/test_site.py` and run with `scripts/check.sh` when development tools are available.

The landing art is original and illustrative. The LIF tuner computes the displayed Euler
model locally; the shipped Atlas plates retain their live equations. The GUI is a design
concept with synthetic activity. Grid cells ship today; TEM is explicitly marked as planned.
Benchmark context and its source are included beside the number and in an HTML comment.
GitHub and PyPI anchors navigate out only when followed; all automatically loaded resources
are local. Canvas work is limited to visible elements, capped at 30 fps and 1.5 device pixels,
and suspended when the document is hidden. Reduced motion renders static artwork, disables
CSS and SVG motion, and avoids animated scrolling and theme transitions.
