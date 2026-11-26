# CI Workflows

neuroSush uses two main GitHub Actions workflows:

1. **CI** (`.github/workflows/ci.yml`):
   - **`lint`**, **`typecheck`**, and **`build`**: Run on all pushes, PRs, workflow dispatch, schedule, and workflow call.
   - **Test Matrix (`test` & `test-extra`)**:
     - **PRs**: Runs only when library code, tests, examples, benchmarks, `pyproject.toml`, or `ci.yml` change (via `dorny/paths-filter`). Runs the light `test` matrix (`ubuntu` py3.10, 3.12, 3.13 latest + py3.10 torch 2.1.0).
     - **Push / Dispatch / Schedule / Reusable Workflow Call**: Runs the light `test` matrix plus `test-extra` (`ubuntu` py3.11 latest + `macos-latest` py3.12).

2. **Site** (`.github/workflows/site.yml`):
   - **`site`**: Runs when site assets, build script, check script, web visualization code, site tests, or `docs/SITE.md` change. Runs Node tests and builds/validates the site.
