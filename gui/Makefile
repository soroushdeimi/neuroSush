.PHONY: dev up check web-check
up:
	cd .. && docker compose up --build

dev:
	cd .. && docker compose -f compose.yml -f compose.dev.yml up --build

web-check:
	npm --prefix web run typecheck
	npm --prefix web run lint
	npm --prefix web test
	npm --prefix web run build

check: web-check
	cd .. && OMP_NUM_THREADS=1 PYTHONPATH=src:gui/server gui/.venv/bin/uv run --project gui/server pytest tests/introspect gui/server/tests -q
