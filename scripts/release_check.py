"""Validate release metadata and extract changelog notes."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def read_version(init_file: Path) -> str:
    """Read the package version without importing the package."""
    match = re.search(
        r"^__version__[ \t]*=[ \t]*(['\"])([^'\"\r\n]+)\1",
        init_file.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"{init_file} has no __version__ assignment")
    return match.group(2)


def changelog_section(text: str, version: str) -> str | None:
    """Return the stripped body of a release section, if present."""
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*(?:\n|$)(.*?)(?=^## \[|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    return None if match is None else match.group(1).strip()


def check(tag: str, version: str, changelog: str) -> list[str]:
    """Return problems with the release tag, version and changelog."""
    errors = []
    if tag != f"v{version}":
        errors.append(f"tag {tag!r} does not match version {version!r}")
    if re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?", version) is None:
        errors.append(f"cannot release development version {version!r}")
    section = changelog_section(changelog, version)
    if section is None:
        errors.append(f"CHANGELOG.md has no section for {version}")
    elif not section:
        errors.append(f"CHANGELOG.md section for {version} is empty")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Check release metadata and optionally write release notes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--notes-out", type=Path)
    args = parser.parse_args(argv)
    try:
        version = read_version(args.root / "src" / "neurosush" / "__init__.py")
        changelog = (args.root / "CHANGELOG.md").read_text(encoding="utf-8")
        errors = check(args.tag, version, changelog)
        if not errors and args.notes_out is not None:
            section = changelog_section(changelog, version)
            args.notes_out.write_text(f"{section}\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        errors = [str(exc)]
    for error in errors:
        sys.stderr.write(f"release check: {error}\n")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
