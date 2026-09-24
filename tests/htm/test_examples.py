"""The Thousand Brains examples run and show what their docstrings claim."""

import importlib.util
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


def test_sequence_prediction_learns_both_endings():
    result = load("sequence_prediction").run(seed=0, epochs=8)
    assert result.anomaly[0] > 0.5
    assert result.anomaly[-1] == 0.0
    assert result.correct


def test_more_voting_columns_need_fewer_touches():
    touches = load("object_recognition").run(objects=40, columns=(1, 4), seed=1)
    assert touches[1] > touches[4] >= 1.0
