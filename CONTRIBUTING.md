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
Run the full check before committing:

```bash
bash scripts/check.sh
```

## Code and commits

Keep math in pure functions and behaviors thin. Test equations with hand-computed
values, and never mutate inputs in place. Add type hints to public functions and
use `from __future__ import annotations` in Python modules. Follow
`docs/ARCHITECTURE.md` for the simulation model and conventions.

Use Conventional Commits with a scope: `feat`, `fix`, `docs`, `test`, `build`,
`ci`, or `refactor`, for example `fix(neurons): validate time constants`.

## Releasing

1. Bump `__version__` in `src/neurosush/__init__.py` to the release version.
2. Move the Unreleased notes under `## [X.Y.Z] - YYYY-MM-DD`, keeping an
   Unreleased section for future changes.
3. Run `bash scripts/check.sh` and commit the version and changelog changes.
4. Create the tag with `git tag -a vX.Y.Z -m vX.Y.Z` and push the commit and tag
   (`git push origin main`, then `git push origin vX.Y.Z`).

The Release workflow verifies the tag, version and changelog, runs CI, publishes
to PyPI via trusted publishing, and creates a GitHub release with the changelog
notes and distribution files. One-time setup: add this GitHub repository as a
trusted publisher on PyPI, selecting the `release.yml` workflow and environment
`pypi`, and create that environment in the repository settings.
