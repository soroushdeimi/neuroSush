import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_check.py"
spec = importlib.util.spec_from_file_location("release_check", SCRIPT)
release_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_check)

CHANGELOG = """# Changelog

## [Unreleased]

## [0.2.0] - 2026-10-01

### Added
- Dopamine modulation.

## [0.1.0] - 2026-09-24

### Added
- First release.
"""


def test_read_version(tmp_path):
    init = tmp_path / "__init__.py"
    init.write_text('"""Doc."""\n\n__version__ = "1.2.3rc1"\n')
    assert release_check.read_version(init) == "1.2.3rc1"


def test_read_version_missing(tmp_path):
    init = tmp_path / "__init__.py"
    init.write_text('"""Doc."""\n')
    with pytest.raises(ValueError, match="__version__"):
        release_check.read_version(init)


def test_changelog_section_returns_the_body():
    body = release_check.changelog_section(CHANGELOG, "0.2.0")
    assert body == "### Added\n- Dopamine modulation."


def test_changelog_section_of_the_last_release():
    assert release_check.changelog_section(CHANGELOG, "0.1.0") == "### Added\n- First release."


def test_changelog_section_missing():
    assert release_check.changelog_section(CHANGELOG, "0.3.0") is None


def test_check_accepts_a_consistent_release():
    assert release_check.check("v0.2.0", "0.2.0", CHANGELOG) == []


@pytest.mark.parametrize(
    ("tag", "version", "fragment"),
    [
        ("v0.2.1", "0.2.0", "does not match"),
        ("0.2.0", "0.2.0", "does not match"),
        ("v0.3.0.dev0", "0.3.0.dev0", "development"),
        ("v0.3.0", "0.3.0", "CHANGELOG"),
    ],
)
def test_check_reports_problems(tag, version, fragment):
    errors = release_check.check(tag, version, CHANGELOG)
    assert any(fragment in error for error in errors)


def test_empty_changelog_section_is_an_error():
    text = "# Changelog\n\n## [0.4.0] - 2026-10-10\n\n## [0.3.0] - 2026-10-02\n- x\n"
    assert any("empty" in e for e in release_check.check("v0.4.0", "0.4.0", text))


def test_main_writes_release_notes(tmp_path, capsys):
    root = tmp_path
    (root / "src" / "neurosush").mkdir(parents=True)
    (root / "src" / "neurosush" / "__init__.py").write_text('__version__ = "0.2.0"\n')
    (root / "CHANGELOG.md").write_text(CHANGELOG)
    notes = tmp_path / "notes.md"
    code = release_check.main(["--tag", "v0.2.0", "--root", str(root), "--notes-out", str(notes)])
    assert code == 0
    assert notes.read_text() == "### Added\n- Dopamine modulation.\n"


def test_main_fails_with_messages(tmp_path, capsys):
    (tmp_path / "src" / "neurosush").mkdir(parents=True)
    (tmp_path / "src" / "neurosush" / "__init__.py").write_text('__version__ = "0.2.0"\n')
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    code = release_check.main(["--tag", "v9.9.9", "--root", str(tmp_path)])
    assert code == 1
    assert "does not match" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("version", "expected"),
    [("0.1.0", True), ("1.2.3rc1", True), ("0.2.0.dev0", False), ("1.0", False)],
)
def test_is_release_version(version, expected):
    assert release_check.is_release_version(version) is expected
    assert release_check.main(["--is-release", version]) == (0 if expected else 1)


def test_print_version(tmp_path, capsys):
    (tmp_path / "src" / "neurosush").mkdir(parents=True)
    (tmp_path / "src" / "neurosush" / "__init__.py").write_text('__version__ = "0.3.0"\n')
    assert release_check.main(["--print-version", "--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == "0.3.0\n"
