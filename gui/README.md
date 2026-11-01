# neuroSush Studio

M0–M1: a local Compose shell, a registry catalog and v1 spec validation. The studio
is a separate Python package; installing the library adds no web dependencies.

From the repository root:

```sh
docker compose up --build
# http://127.0.0.1:8765

docker compose -f compose.yml -f compose.dev.yml up --build
# Vite: http://127.0.0.1:5173; API: http://127.0.0.1:8765

docker compose -f compose.yml -f compose.gpu.yml up --build
```

The default image uses CPU PyTorch. The GPU override selects CUDA 13.0 wheels and
reserves one NVIDIA GPU; the host needs NVIDIA Container Toolkit. Both run with
uid 10001, a read-only root filesystem and a persistent `/data` volume. Images
have not been built locally because the Docker daemon was unavailable. Compose
configuration was checked for all three combinations. Dockerfile-specific ignore
files keep virtual environments, browser downloads and node_modules out of builds.

For development without Docker, install uv in a local environment, then:

```sh
uv sync --project gui/server --frozen
npm --prefix gui/web ci
npm --prefix gui/web run build
gui/run-server --port 8000
# In a second terminal: npm --prefix gui/web run dev
```

`gui/run-server` uses `gui/.venv/bin/uv` when available, otherwise `uv` on PATH.
`NS_PYTHON` can select an existing Python environment with the studio dependencies.
`NS_WEB_ROOT` overrides the built frontend directory; `NS_API_PROXY` overrides the
Vite proxy target. `make -C gui dev` starts the dev Compose stack.

The form discovers all LIF parameter names, types, required flags and defaults
from `/api/schema`. Required constructor values have no invented defaults. The
summary contains native v1 ColumnSpec JSON. The six stages indicate the intended
flow; only the Model chapter is implemented. There is no runner or probe yet.

Endpoints:

- `GET /api/health`: readiness.
- `GET /api/schema`: every currently registered behavior, docstrings, parameter
  descriptors, execution metadata and non-JSON limitations.
- `POST /api/validate`: v1 ColumnSpec JSON, parsed with the library's `from_json`.
  Returns `{valid, errors: [{path, message, fix}]}`. It checks document structure,
  parameter types, dimensions and constructors; it does not initialize or run a
  network. Malformed HTTP JSON returns FastAPI's 422 response.

`PYTHONPATH=src python -m neurosush.introspect --check` exits nonzero when a
registration has no usable JSON constructor schema. Scalar alternatives of
`float | Tensor` remain usable. Unsupported tensor/callable alternatives and
non-finite defaults are reported in each descriptor's `limitations`; a missing
tensor decoder does not hide the entire LIF constructor. No spec or registry
behavior is changed.

Motion uses shared duration/easing variables, finite staggered entrances,
button feedback and a summary reveal. Both the system reduced-motion setting and
the persistent in-app toggle disable animations. No animation presents fabricated
simulation activity. Typography currently uses system fallbacks; licensed font
bundles are left for the later design-system milestone.

Versions resolved and pinned on 2026-11-02:

| Tool | Version |
| --- | --- |
| React / React DOM | 19.3.1 |
| Vite / React plugin | 8.3.5 / 6.1.2 |
| TypeScript | 6.0.3 |
| ESLint / typescript-eslint | 10.14.0 / 8.72.1 |
| Vitest / jsdom | 5.0.5 / 30.1.3 |
| Playwright | 1.64.1 |
| FastAPI / uvicorn / pydantic | 0.162.0 / 0.54.0 / 2.14.2 |
| PyTorch | 2.14.1 |
| hatchling / uv | 1.33.0 / 0.13.6 |

All direct npm dependencies use exact versions in `package.json`; transitive
versions are in `package-lock.json`. Python dependencies are in
`gui/server/pyproject.toml` and `uv.lock`; Linux uv installs use the CPU wheel index.
The Node 24 and Python 3.12 base images are pinned by registry digest in Dockerfiles.

Checks, from the repository root after `uv sync` and `npm ci`:

```sh
export OMP_NUM_THREADS=1 PYTHONPATH=src:gui/server
uv run --project gui/server ruff check src gui/server tests/introspect
uv run --project gui/server ruff format src/neurosush/introspect.py gui/server tests/introspect
uv run --project gui/server mypy
uv run --project gui/server mypy src gui/server/studio
uv run --project gui/server pytest tests/introspect tests/structure/test_spec.py gui/server/tests -q
npm --prefix gui/web run typecheck
npm --prefix gui/web run lint
npm --prefix gui/web test
npm --prefix gui/web run build
cd gui/web
PLAYWRIGHT_BROWSERS_PATH=../.browsers npx playwright install chromium
PLAYWRIGHT_BROWSERS_PATH=../.browsers npm run e2e
```

Browser tests use the full Chromium binary. The matching headless shell stalled
frame callbacks on the local macOS host; full Chromium passed the same actionability
checks without forced clicks. Tests run with one worker.
