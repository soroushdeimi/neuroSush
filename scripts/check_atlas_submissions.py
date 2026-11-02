#!/usr/bin/env python3
"""Atlas community submission validator for NeuroSush.

CLI:
    python scripts/check_atlas_submissions.py [--changed-since REF] [--write-parity SLUG]

Validates community plates and ideas under atlas/community/ against rules in WORK_PACKAGES.md.
Exits 1 on any violation, printing one line per problem: <path>: <rule>: <fix>.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SLUG_RE = re.compile(r"^(_?[a-z0-9]+(-[a-z0-9]+)*)$")
FORBIDDEN_TOKENS = [
    "fetch(",
    "XMLHttpRequest",
    "WebSocket",
    "import(",
    "importScripts",
    "eval(",
    "new Function",
    "localStorage",
    "document.cookie",
    "http://",
    "https://",
]


def validate_schema(data: dict[str, Any]) -> list[str]:
    """Lightweight pure-python schema validator for plate.json."""
    errors = []
    if not isinstance(data, dict):
        return ["manifest must be an object"]

    if data.get("schema") != 1:
        errors.append("schema must be integer 1")

    kind = data.get("kind")
    if kind not in ("plate", "idea"):
        errors.append("kind must be 'plate' or 'idea'")

    plate_id = data.get("id")
    if not isinstance(plate_id, str) or not re.match(r"^c-[a-z0-9]+(-[a-z0-9]+)*$", plate_id):
        errors.append("id must match pattern ^c-[a-z0-9]+(-[a-z0-9]+)*$")

    title = data.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append("title is required and must be non-empty string")

    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("summary is required and must be non-empty string")

    authors = data.get("authors")
    if not isinstance(authors, list) or len(authors) == 0:
        errors.append("authors must be a non-empty list")
    else:
        for a in authors:
            if not isinstance(a, dict) or not a.get("name") or not a.get("github"):
                errors.append("each author must have name and github")

    if data.get("license") != "MIT":
        errors.append("license must be 'MIT'")

    data_type = data.get("data")
    if data_type not in ("teaching-js", "recorded-run", "illustrative"):
        errors.append("data must be 'teaching-js', 'recorded-run', or 'illustrative'")

    covers = data.get("covers")
    if not isinstance(covers, list) or len(covers) == 0:
        errors.append("covers must be a non-empty list of module/class references")

    return errors


def check_submission(sub_dir: Path, write_parity: bool = False) -> list[str]:
    issues = []
    slug = sub_dir.name
    manifest_path = sub_dir / "plate.json"
    readme_path = sub_dir / "README.md"

    # 1. Slug check
    if not SLUG_RE.match(slug) or len(slug) > 40:
        issues.append(
            f"{sub_dir}: slug: Use lowercase alphanumeric and hyphens only (max 40 chars)"
        )

    # 2. Manifest and README existence
    if not manifest_path.is_file():
        issues.append(f"{manifest_path}: manifest: Missing plate.json")
        return issues

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        issues.append(f"{manifest_path}: json: Invalid JSON in plate.json: {e}")
        return issues

    schema_errors = validate_schema(manifest)
    for err in schema_errors:
        issues.append(f"{manifest_path}: schema: {err}")

    expected_id = f"c-{slug.lstrip('_')}"
    if manifest.get("id") != expected_id:
        issues.append(f"{manifest_path}: id: plate.json id must be {expected_id}")

    if not readme_path.is_file():
        issues.append(f"{readme_path}: readme: Missing README.md")

    kind = manifest.get("kind")
    data_type = manifest.get("data")

    # 3. Allowed files check
    allowed_idea = {"plate.json", "README.md"}
    allowed_plate = {
        "plate.json",
        "README.md",
        "plate.js",
        "plate.test.mjs",
        "parity.json",
        "trace.json",
        "preview.png",
    }
    actual_files = {p.name for p in sub_dir.iterdir() if p.is_file()}

    if kind == "idea":
        unauthorized = actual_files - allowed_idea
        if unauthorized:
            issues.append(
                f"{sub_dir}: allowed-files: Idea submissions must contain "
                f"only plate.json and README.md (found {unauthorized})"
            )
        return issues
    unauthorized = actual_files - allowed_plate
    if unauthorized:
        issues.append(f"{sub_dir}: allowed-files: Unauthorized files found: {unauthorized}")

    # 4. Size limits
    total_size = 0
    plate_js_path = sub_dir / "plate.js"
    trace_json_path = sub_dir / "trace.json"

    for p in sub_dir.iterdir():
        if p.is_file():
            size = p.stat().st_size
            total_size += size
            if p.name == "plate.js" and size > 150 * 1024:
                issues.append(f"{p}: size-limit: plate.js exceeds 150 kB limit")
            if p.name == "trace.json" and size > 2 * 1024 * 1024:
                issues.append(f"{p}: size-limit: trace.json exceeds 2 MB limit")
            if p.name == "preview.png" and size > 300 * 1024:
                issues.append(f"{p}: size-limit: preview.png exceeds 300 kB limit")

    if total_size > 3 * 1024 * 1024:
        issues.append(f"{sub_dir}: size-limit: Total size exceeds 3 MB limit")

    # 5. Covers importlib check
    for cover in manifest.get("covers", []):
        try:
            parts = cover.split(".")
            mod_name = ".".join(parts[:-1])
            attr_name = parts[-1]
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, attr_name):
                issues.append(
                    f"{manifest_path}: covers: Attribute {attr_name} not found in {mod_name}"
                )
        except Exception as e:
            issues.append(f"{manifest_path}: covers: Cover {cover} failed to resolve: {e}")

    # 6. Data-specific requirements
    if data_type == "teaching-js":
        if not (sub_dir / "plate.js").is_file():
            issues.append(f"{sub_dir}: teaching-js: missing plate.js")
        if not (sub_dir / "plate.test.mjs").is_file():
            issues.append(f"{sub_dir}: teaching-js: missing plate.test.mjs")
        equations = manifest.get("equations", [])
        if not isinstance(equations, list) or len(equations) == 0:
            issues.append(f"{manifest_path}: equations: teaching-js requires equations")
    elif data_type == "recorded-run":
        if not trace_json_path.is_file():
            issues.append(f"{sub_dir}: recorded-run: missing trace.json")
        else:
            try:
                with open(trace_json_path, encoding="utf-8") as f:
                    trace_data = json.load(f)
                meta = trace_data.get("meta", {})
                for field in ("library", "seed", "dt"):
                    if field not in meta:
                        issues.append(
                            f"{trace_json_path}: recorded-run: "
                            f"trace.json meta missing required field '{field}'"
                        )
            except Exception as e:
                issues.append(f"{trace_json_path}: json: invalid trace.json: {e}")
        if not (sub_dir / "renderer").is_file() and not (sub_dir / "plate.js").is_file():
            renderer = manifest.get("renderer")
            if not renderer or not renderer.get("core_plate"):
                issues.append(
                    f"{manifest_path}: recorded-run: "
                    "recorded-run requires renderer.core_plate or plate.js"
                )

    # 7 & 8. plate.js static checks (forbidden tokens and reduced-motion)
    if plate_js_path.is_file():
        try:
            with open(plate_js_path, encoding="utf-8", errors="ignore") as f:
                js_content = f.read()
            for token in FORBIDDEN_TOKENS:
                if token in js_content:
                    issues.append(
                        f"{plate_js_path}: forbidden-tokens: "
                        f"plate.js contains forbidden token '{token}'"
                    )
            if "prefers-reduced-motion" not in js_content and "matchMedia" not in js_content:
                issues.append(
                    f"{plate_js_path}: reduced-motion: "
                    "plate.js must check prefers-reduced-motion via matchMedia"
                )
        except Exception as e:
            issues.append(f"{plate_js_path}: read: failed to read plate.js: {e}")

    # 9. Parity check / generation
    if data_type == "teaching-js":
        equations = manifest.get("equations", [])
        parity_path = sub_dir / "parity.json"
        computed_parity = []

        for eq in equations:
            py_ref = eq.get("python_ref")
            cases = eq.get("cases", [])
            eq_results = []

            # Allowlist check for python_ref
            if not py_ref.startswith("neurosush.neurons.") and not py_ref.startswith(
                "neurosush.synapses."
            ):
                issues.append(
                    f"{manifest_path}: python_ref: "
                    f"python_ref {py_ref} must be in neurosush.neurons or synapses"
                )
                continue

            try:
                parts = py_ref.split(".")
                mod_name = ".".join(parts[:-1])
                attr_name = parts[-1]
                mod = importlib.import_module(mod_name)
                func = getattr(mod, attr_name)

                for case in cases:
                    args = case.get("args", [])
                    kwargs = case.get("kwargs", {})
                    res = func(*args, **kwargs)
                    if hasattr(res, "tolist"):
                        res = res.tolist()
                    eq_results.append(res)
            except Exception as e:
                issues.append(f"{manifest_path}: parity: failed to evaluate {py_ref}: {e}")

            computed_parity.append({"js": eq.get("js"), "results": eq_results})

        if write_parity:
            try:
                with open(parity_path, "w", encoding="utf-8") as f:
                    json.dump(computed_parity, f, indent=2)
                    f.write("\n")
            except Exception as e:
                issues.append(f"{parity_path}: parity: failed to write parity.json: {e}")
        else:
            if not parity_path.is_file():
                issues.append(
                    f"{parity_path}: parity: missing parity.json (run with --write-parity {slug})"
                )
            else:
                try:
                    with open(parity_path, encoding="utf-8") as f:
                        committed_parity = json.load(f)
                    if committed_parity != computed_parity:
                        issues.append(
                            f"{parity_path}: parity: parity.json mismatch. "
                            f"Run with --write-parity {slug}"
                        )
                except Exception as e:
                    issues.append(f"{parity_path}: json: invalid parity.json: {e}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Atlas community submissions.")
    parser.add_argument("--changed-since", help="Git ref to check changed submissions against.")
    parser.add_argument("--write-parity", help="Regenerate parity.json for the given slug.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    community_dir = repo_root / "atlas" / "community"

    if not community_dir.is_dir():
        print(f"Community directory not found: {community_dir}", file=sys.stderr)
        return 0

    target_subdirs = []
    if args.write_parity:
        target_dir = community_dir / args.write_parity
        if not target_dir.is_dir():
            print(
                f"Error: submission directory not found for slug: {args.write_parity}",
                file=sys.stderr,
            )
            return 1
        target_subdirs = [target_dir]
    elif args.changed_since:
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", args.changed_since],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            changed_files = result.stdout.splitlines()
            changed_slugs = set()
            for cf in changed_files:
                parts = Path(cf).parts
                if len(parts) >= 3 and parts[0] == "atlas" and parts[1] == "community":
                    changed_slugs.add(parts[2])
            target_subdirs = [
                community_dir / slug for slug in changed_slugs if (community_dir / slug).is_dir()
            ]
        except Exception as e:
            print(f"Error running git diff: {e}", file=sys.stderr)
            target_subdirs = [d for d in community_dir.iterdir() if d.is_dir()]
    else:
        target_subdirs = [d for d in community_dir.iterdir() if d.is_dir()]

    all_issues = []
    for sub_dir in sorted(target_subdirs):
        if not sub_dir.is_dir():
            continue
        write_p = args.write_parity == sub_dir.name
        issues = check_submission(sub_dir, write_parity=write_p)
        all_issues.extend(issues)

    for issue in all_issues:
        print(issue)

    return 1 if all_issues else 0


if __name__ == "__main__":
    sys.exit(main())
