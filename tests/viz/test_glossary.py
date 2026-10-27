from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

GLOSSARY_DIR = Path(__file__).resolve().parents[2] / "src" / "neurosush" / "viz" / "i18n"
GLOSSARY = GLOSSARY_DIR / "glossary_en.json"
FIELDS = ("term", "short", "long", "plates", "symbol")
MAX_SHORT = 140
MIN_ENTRIES = 70
KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
COMMON_PUNCTUATION = frozenset(
    "\u2018\u2019\u201c\u201d\u201e\u2013\u2014\u2026\u00b7\u2022"
    "\u00b0\u00b1\u00d7\u00f7\u2248\u2260\u2264\u2265\u2190\u2192"
    "\u2191\u2193\u2020\u2021\u00a7\u00b6"
)

VALID_PLATES = frozenset(
    [f"N{i:02d}" for i in range(1, 10)]
    + [f"D{i:02d}" for i in range(1, 4)]
    + [f"Y{i:02d}" for i in range(1, 7)]
    + [f"C{i:02d}" for i in range(1, 4)]
    + [f"W{i:02d}" for i in range(1, 6)]
    + [f"E{i:02d}" for i in range(1, 5)]
    + [f"H{i:02d}" for i in range(1, 9)]
    + [f"R{i:02d}" for i in range(1, 3)]
    + [f"X{i:02d}" for i in range(1, 4)]
)


@pytest.fixture(scope="module")
def glossary() -> dict[str, dict]:
    return json.loads(GLOSSARY.read_text(encoding="utf-8"))


def test_only_the_english_glossary_exists():
    names = {path.name for path in GLOSSARY_DIR.glob("glossary_*.json")}
    assert names == {"glossary_en.json"}, sorted(names)


def test_glossary_is_a_json_object(glossary):
    assert isinstance(glossary, dict)


def test_every_entry_has_the_five_fields_with_correct_types(glossary):
    for key, entry in glossary.items():
        assert set(entry) == set(FIELDS), key
        assert isinstance(entry["term"], str), key
        assert entry["term"].strip(), key
        assert isinstance(entry["short"], str), key
        assert entry["short"].strip(), key
        assert isinstance(entry["long"], str), key
        assert entry["long"].strip(), key
        assert isinstance(entry["plates"], list), key
        assert all(isinstance(plate, str) for plate in entry["plates"]), key
        assert entry["symbol"] is None or (
            isinstance(entry["symbol"], str) and entry["symbol"].strip()
        ), key


def test_keys_are_lowercase_snake_case(glossary):
    for key in glossary:
        assert KEY_PATTERN.fullmatch(key), key


def test_at_least_seventy_entries(glossary):
    assert len(glossary) >= MIN_ENTRIES


def test_short_within_length_limit(glossary):
    for key, entry in glossary.items():
        assert len(entry["short"]) <= MAX_SHORT, f"{key}: {len(entry['short'])} chars"


def test_long_is_non_empty(glossary):
    for key, entry in glossary.items():
        assert entry["long"].strip(), key


def test_long_is_printable_ascii(glossary):
    for key, entry in glossary.items():
        offenders = sorted(
            {ch for ch in entry["long"] if not (" " <= ch <= "~" or ch in COMMON_PUNCTUATION)}
        )
        assert not offenders, f"{key}: {offenders}"


def test_plates_are_valid_and_non_empty(glossary):
    for key, entry in glossary.items():
        assert entry["plates"], key
        unknown = set(entry["plates"]) - VALID_PLATES
        assert not unknown, f"{key}: {sorted(unknown)}"


def test_no_duplicate_terms(glossary):
    terms = [entry["term"] for entry in glossary.values()]
    duplicates = {term for term in terms if terms.count(term) > 1}
    assert not duplicates, sorted(duplicates)


def test_short_differs_from_long(glossary):
    for key, entry in glossary.items():
        assert entry["short"] != entry["long"], key
