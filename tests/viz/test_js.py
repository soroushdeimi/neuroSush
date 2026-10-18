"""Pytest runner for Atlas JavaScript unit tests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_atlas_js_suite() -> None:
    """Run node --test tests/viz/js and assert all tests pass."""
    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js is not installed on PATH")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [node_bin, "--test", "tests/viz/js"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        pytest.fail(
            f"node --test failed with exit code {result.returncode}:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
