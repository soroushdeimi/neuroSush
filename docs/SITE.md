# Public site

The Pages workflow runs `python scripts/build_site.py` and publishes `site/`.
The deterministic builder exports every shipped teaching plate and the talk demo,
and adds shared navigation, tokens, theme control and footer to the exported plates.
Standalone library exports remain self-contained and unchanged.

Preview from the repository root:

```bash
PYTHONPATH=src python3 scripts/build_site.py --out /tmp/neurosush-site-preview
python3 -m http.server 8765 --bind 127.0.0.1 --directory /tmp/neurosush-site-preview
```

Open http://127.0.0.1:8765/. Validate with:

```bash
python3 scripts/check_site.py /tmp/neurosush-site-preview
node --check scripts/site_assets/site.js
node scripts/check_site_browser.mjs /tmp/neurosush-site-shots
```

The browser check uses Node 22+ and local Chrome through CDP without npm packages.
It checks themes, search, local requests, console errors, reduced motion and overflow
at 390, 1440 and 1920 pixels. It captures six story positions in both themes on desktop
and mobile, plus the gallery and two teaching plates.

The seeded `neuron.js` module samples a pyramidal soma, recursive dendritic tubes
with spines, apical tuft, basal arbor, myelinated axon and terminal boutons.
Its action-potential wave is triggered by the same live Euler LIF update as the trace.
Original WebGL2 art morphs between six seeded point clouds, with native scrolling,
a sticky viewport and eased scene interpolation. Desktop starts with 32,000 points;
mobile starts with 9,000. Sustained slow frames reduce the draw count. Device pixels
are capped at two; off-screen and hidden-tab animation pauses. Reduced motion or
missing WebGL renders a static Canvas2D neuron. No third-party assets or libraries load.
`window.neurosushScene.stats` reports recent animation frame cadence and point count.
This measures browser frame callbacks, not GPU execution time.

The scene is illustrative. Its LIF readout computes the documented Euler update,
threshold and reset locally. Atlas numerical code is preserved. Spatial structure is
not described as learned emergence; TEM is planned for this published snapshot.
Benchmark figures are quoted from docs/BENCHMARKS.md with hardware and workload.
