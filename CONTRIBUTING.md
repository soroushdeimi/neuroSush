# Contributing

## Development setup

Use Python 3.10 or newer. Create and activate a virtual environment, install CPU
PyTorch, then install the package with development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

On macOS, use `pip install torch` instead of the CPU index command.
Install the git hooks once; they fix whitespace and run ruff on every commit, as the CI
lint job does:

```bash
pre-commit install
```

Run the full check before committing:

```bash
bash scripts/check.sh
```

Tests marked `gpu` compare CUDA runs with the CPU. They are skipped on machines without a
CUDA device (including CI), so run them locally after changing tensor code:
`pytest -m gpu`.

## Code and commits

Keep math in pure functions and behaviors thin. Test equations with hand-computed
values, and never mutate inputs in place. Add type hints to public functions and
use `from __future__ import annotations` in Python modules. Follow
`docs/ARCHITECTURE.md` for the simulation model and conventions.

Use Conventional Commits with a scope: `feat`, `fix`, `docs`, `test`, `build`,
`ci`, or `refactor`, for example `fix(neurons): validate time constants`.

## Benchmarks

`python benchmarks/suite.py` times the main workloads (dense STDP, batched simulation,
spatial pooler and temporal memory learning). Each rate is also divided by a calibration
workload measured on the same machine, which removes most of the difference between
machines. The Benchmarks workflow runs the suite every Monday (or on demand) and compares
it with `benchmarks/baseline.json`: a normalized drop of more than 30% fails the run.

To record or move the baseline after an intended change, download the `benchmark`
artifact of a Benchmarks run on GitHub and commit it as `benchmarks/baseline.json`; a
baseline from another machine is not comparable.

## Releasing

Releases are automatic:

1. Set `__version__` in `src/neurosush/__init__.py` to the release version.
2. Move the Unreleased notes under `## [X.Y.Z] - YYYY-MM-DD`, keeping an
   Unreleased section for future changes.
3. Run `bash scripts/check.sh`, commit, and push to `main`.

On that push the Release workflow sees a release version without a GitHub
release, checks the version and changelog, runs the full CI, and creates the tag
`vX.Y.Z` and a GitHub release with the wheel, the sdist and the changelog notes.
Development versions (`X.Y.Z.devN`) are never released. Pushing a tag `vX.Y.Z`
by hand does the same.

PyPI publishing is a separate job that is off until it is set up once: add this
repository as a trusted publisher on PyPI (workflow `release.yml`, environment
`pypi`), create the `pypi` environment in the repository settings, and set the
repository variable `PYPI_PUBLISH` to `true`. After that every release is also
published to PyPI.
