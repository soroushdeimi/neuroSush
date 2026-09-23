#!/usr/bin/env bash
# Lint, format check and the full test suite; extra arguments go to pytest.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi
"$PY" -m ruff check .
"$PY" -m ruff format --check .
"$PY" -m pytest "$@"
