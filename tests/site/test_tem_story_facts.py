import json
import re
from pathlib import Path


def test_tem_story_facts_and_story():
    root = Path(__file__).resolve().parents[2]
    story_dir = root / "scripts" / "site_assets" / "story-tem"
    facts_path = story_dir / "facts.json"
    story_path = story_dir / "story.md"

    assert facts_path.is_file(), "facts.json missing"
    assert story_path.is_file(), "story.md missing"

    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    assert isinstance(facts, list)
    assert len(facts) > 0

    fact_values = set()
    for fact in facts:
        assert "id" in fact
        assert "value" in fact
        assert "source" in fact
        assert "type" in fact
        assert "status" in fact

        assert fact["source"], f"Fact {fact['id']} lacks a source"
        if fact["status"] != "VERIFIED":
            assert fact["value"] is None or fact["value"] == ""
        if fact["value"] is not None:
            fact_values.add(str(fact["value"]))

    # Check all numbers in story.md exist in facts.json
    story_text = story_path.read_text(encoding="utf-8")
    numbers_in_story = re.findall(r"-?\d+(?:\.\d+)?", story_text)

    permitted_extra = {"1", "2", "6", "7", "21", "23", "95", "183", "1249", "1263", "5", "0", "1.0"}
    for num in numbers_in_story:
        matched = False
        if num in permitted_extra:
            matched = True
        else:
            try:
                f_num = float(num)
                for fv in fact_values:
                    try:
                        if abs(float(fv) - f_num) < 1e-9:
                            matched = True
                            break
                    except ValueError:
                        if num in fv:
                            matched = True
                            break
            except ValueError:
                if num in fact_values:
                    matched = True
        assert matched, f"Number '{num}' found in story.md but not present in facts.json"

    # Check forbidden claims in story.md
    story_lower = story_text.lower()
    assert "grid cells emerged" not in story_lower
    assert "better than the paper" not in story_lower


def test_tem_ledger_generator_is_reproducible():
    import runpy

    root = Path(__file__).resolve().parents[2]
    directory = root / "scripts/site_assets/story-tem"
    generated = runpy.run_path(str(directory / "make_facts.py"))["FACTS"]
    assert generated == json.loads((directory / "facts.json").read_text())
    assert len({row["id"] for row in generated}) == len(generated)


def test_tem_panel_sources_and_word_budget():
    import subprocess

    root = Path(__file__).resolve().parents[2]
    script = (
        "import {panels} from './scripts/site_assets/story-tem/panels.mjs';"
        "console.log(JSON.stringify(panels));"
    )
    panels = json.loads(
        subprocess.check_output(["node", "--input-type=module", "-e", script], cwd=root)
    )
    facts = {
        row["id"]: row
        for row in json.loads((root / "scripts/site_assets/story-tem/facts.json").read_text())
    }
    for panel in panels:
        assert 45 <= len(panel["copy"].split()) <= 60
        assert panel["note"]
        for fact_id in panel["facts"]:
            assert facts[fact_id]["status"] == "VERIFIED"


def test_tem_derived_arena_size():
    root = Path(__file__).resolve().parents[2]
    facts = {
        row["id"]: row["value"]
        for row in json.loads((root / "scripts/site_assets/story-tem/facts.json").read_text())
    }
    assert facts["arena_positions"] == facts["arena_side"] ** 2
    assert facts["sensory_neurons"] < facts["arena_positions"]


def test_tem_ideal_lattice_scores_recompute_from_stored_maps():
    import math

    import pytest
    import torch

    from neurosush.htm.grid_cells import hexagonal_rate
    from neurosush.spatial import gridness

    root = Path(__file__).resolve().parents[2]
    facts = {
        row["id"]: row["value"]
        for row in json.loads((root / "scripts/site_assets/story-tem/facts.json").read_text())
    }
    axis = torch.arange(facts["ideal_map_bins"], dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    scale = facts["ideal_scale"]
    expected = {
        "square": 2 + torch.cos(2 * math.pi * x / scale) + torch.cos(2 * math.pi * y / scale),
        "hex": hexagonal_rate(torch.stack((x, y), -1), scale=scale, orientation=0),
    }
    for name in ("square", "hex"):
        rate = torch.tensor(facts["ideal_maps"][name], dtype=torch.float64)
        torch.testing.assert_close(rate, expected[name], rtol=0, atol=0)
        score = gridness(rate, annulus=(facts["ideal_annulus_inner"], facts["ideal_annulus_outer"]))
        c = score.correlations
        recomputed = min(c[60], c[120]) - max(c[30], c[90], c[150])
        assert facts[f"ideal_{name}_score_display"] == f"{score.hexagonal:.3f}"
        assert score.resolvable
        assert facts[f"ideal_{name}_score"] == pytest.approx(recomputed, rel=0, abs=1e-12)
    assert facts["ideal_square_score"] < 0 < facts["ideal_hex_score"]
