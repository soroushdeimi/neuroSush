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

Shared `tokens.css` and `components.css` implement warm paper (default) and graphite,
semantic data colours, the spacing scale and shared landing/gallery/plate chrome.
Fonts are static WOFF2 subsets of Source Serif 4, IBM Plex Sans/Mono and STIX Two Math
from google/fonts, under the adjacent verified SIL OFL licences. The complete set is
174,432 bytes; non-math fonts total 154,332 bytes. Math loads only when used.
Chapter order is neuron, synapse, circuit, columns, hierarchy, space. Mobile uses
separate figures above prose; final scene modules are being integrated individually.

`instruments.js` advances the LIF model at 1 ms simulated per 1 ms displayed,
with fixed Euler dt 0.5 ms and a 200 ms labelled window. Neuron centre-lines supplement
the sampled tube surfaces. The spike propagation speed is a visual encoding.

`synapse.js` uses the seeded surface sampler in `shape-kit.js` for bouton,
vesicles, receptors, cleft and dendritic spine. Transmitter motion and spine size
are schematic; the inset STDP curve computes the signed exponential window
with tau 20 ms and soft-bound weight 0.5, not a measurement from the morphology.

`microcircuit.js` draws six cortical density bands with 3,200 cells (1,200 mobile),
a 4:1 ratio of pyramidal triangles to inhibitory circles, apical dendrites, local
connections and a seeded generated raster. Activity waves illustrate E/I populations;
this is explicitly schematic rather than a balanced-network measurement.
