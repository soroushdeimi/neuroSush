"""The catalog carries every planned plate and covers the library's public models."""

import importlib
import inspect

import pytest

from neurosush.core.behavior import Behavior
from neurosush.viz.catalog import EXEMPT, PLANNED, PLATES, public_models, uncovered

# The contract: every plate id of PLAN.md section 3, with its wave and flagship flag.
EXPECTED_IDS = (
    "N01",
    "N02",
    "N03",
    "N04",
    "N05",
    "N06",
    "N07",
    "N08",
    "N09",
    "D01",
    "D02",
    "D03",
    "Y01",
    "Y02",
    "Y03",
    "Y04",
    "Y05",
    "Y06",
    "C01",
    "C02",
    "C03",
    "W01",
    "W02",
    "W03",
    "W04",
    "W05",
    "E01",
    "E02",
    "E03",
    "E04",
    "H01",
    "H02",
    "H03",
    "H04",
    "H05",
    "H06",
    "H07",
    "H08",
    "R01",
    "R02",
    "X01",
    "X02",
    "X03",
)
EXPECTED_WAVES = {
    "N01": 2,
    "N02": 2,
    "N03": 2,
    "N04": 2,
    "N05": 2,
    "N06": 2,
    "N07": 2,
    "N08": 2,
    "N09": 3,
    "D01": 3,
    "D02": 3,
    "D03": 3,
    "Y01": 2,
    "Y02": 3,
    "Y03": 3,
    "Y04": 3,
    "Y05": 3,
    "Y06": 3,
    "C01": 3,
    "C02": 3,
    "C03": 3,
    "W01": 3,
    "W02": 3,
    "W03": 3,
    "W04": 4,
    "W05": 4,
    "E01": 2,
    "E02": 4,
    "E03": 4,
    "E04": 4,
    "H01": 3,
    "H02": 3,
    "H03": 4,
    "H04": 4,
    "H05": 4,
    "H06": 4,
    "H07": 4,
    "H08": 4,
    "R01": 4,
    "R02": 4,
    "X01": 3,
    "X02": 3,
    "X03": 4,
}
EXPECTED_FLAGSHIPS = ("H04", "N05", "Y05")

# Catalog names that resolve to a real class yet stay out of public_models(); none so far.
EXTRA: dict[str, str] = {}


def has_arabic_script(text):
    return any(
        "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f" or "\u08a0" <= ch <= "\u08ff"
        for ch in text
    )


def resolves(name):
    module_name, _, attr = name.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def test_every_planned_plate_id_exists():
    assert set(PLATES) == set(EXPECTED_IDS)
    assert len(PLATES) == len(EXPECTED_IDS)


def test_waves_match_the_plan():
    assert all(1 <= PLATES[plate_id].wave <= 4 for plate_id in PLATES)
    assert {plate_id: spec.wave for plate_id, spec in PLATES.items()} == EXPECTED_WAVES


def test_flagship_flags_match_the_plan():
    assert tuple(sorted(plate_id for plate_id, s in PLATES.items() if s.flagship)) == (
        EXPECTED_FLAGSHIPS
    )


def test_titles_are_non_empty():
    for spec in PLATES.values():
        assert spec.title_en
        assert spec.title_fa


def test_persian_titles_use_arabic_script():
    for spec in PLATES.values():
        assert has_arabic_script(spec.title_fa), spec.id


def test_planned_names_are_qualified_class_paths():
    assert PLANNED  # nothing planned means the plan changed; tell the manager
    for name in PLANNED:
        parts = name.split(".")
        assert len(parts) >= 3
        assert all(parts)


def test_every_covered_name_resolves_and_is_a_public_model():
    for spec in PLATES.values():
        for name in spec.covers:
            obj = resolves(name)
            assert inspect.isclass(obj), f"{name} of {spec.id} is not a class"
            assert name in public_models() or name in EXTRA, (
                f"{name} of {spec.id} is not a public model"
            )


def test_covers_the_whole_library():
    assert uncovered() == []


def test_exempt_reasons_are_documented():
    for name, reason in EXEMPT.items():
        assert len(reason) >= 20, name


def test_exempt_names_exist():
    for name in EXEMPT:
        obj = resolves(name)
        assert inspect.isclass(obj), name


@pytest.mark.parametrize("name", sorted(EXEMPT))
def test_exempt_names_are_not_required_by_the_coverage_check(name):
    assert name not in uncovered()


def test_uncovered_reports_a_class_no_plate_covers(monkeypatch):
    import neurosush.neurons.models as models

    class FakeNeuron(Behavior):
        """A public behavior no plate covers."""

        order = 999

    FakeNeuron.__module__ = models.__name__
    monkeypatch.setattr(models, "FakeNeuron", FakeNeuron, raising=False)
    assert "neurosush.neurons.models.FakeNeuron" in uncovered()


def test_public_models_reports_name_of_defined_class_only(monkeypatch):
    import neurosush.neurons.models as models

    class Reexported(Behavior):
        """A class re-exported into models from elsewhere stays out."""

        order = 999

    Reexported.__module__ = "neurosush.elsewhere"
    monkeypatch.setattr(models, "Reexported", Reexported, raising=False)
    assert "neurosush.neurons.models.Reexported" not in public_models()


def test_public_models_skips_viz():
    assert all(not name.startswith("neurosush.viz") for name in public_models())
