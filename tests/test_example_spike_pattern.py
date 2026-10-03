"""The spike pattern detection example generates its input and runs a short simulation."""

import importlib.util
import itertools
import sys
from pathlib import Path

import torch

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "spike_pattern_detection.py"


def load():
    spec = importlib.util.spec_from_file_location("spike_pattern_detection", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["spike_pattern_detection"] = module
    spec.loader.exec_module(module)
    return module


def test_input_has_rates_in_range_and_repeating_pattern():
    spd = load()
    source = spd.PatternInput(200, seed=1)
    stream = iter(source)
    frames = torch.stack([next(stream) for _ in range(3000)])
    assert frames.shape == (3000, 200)
    assert frames.dtype == torch.bool
    assert 0.0 < frames.float().mean() < (spd.MAX_HZ + spd.NOISE_HZ) / 1000
    onsets = source.onsets
    assert len(onsets) >= 5
    assert all(b - a >= spd.PATTERN_MS for a, b in itertools.pairwise(onsets))
    # the pattern afferents repeat the same spikes (plus 10 Hz noise) at every onset
    first = frames[onsets[0] : onsets[0] + spd.PATTERN_MS, : source.n_pat]
    second = frames[onsets[1] : onsets[1] + spd.PATTERN_MS, : source.n_pat]
    assert (first & second).sum() >= 0.8 * source.pattern.sum()


def test_short_run_updates_weights_and_analysis_runs():
    spd = load()
    result = spd.run(3.0, seed=0, n=200, threshold=11.5)
    w = result["weights"]
    assert w.shape == (200,)
    assert w.min() >= 0
    assert w.max() <= 1
    assert not torch.allclose(w, torch.full_like(w, 0.475))
    stats = spd.analyse(result)
    assert 0 <= stats["hit_rate"] <= 1
    assert stats["high"] + stats["low"] + stats["middle"] == 200
