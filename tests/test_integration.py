"""End-to-end: the two-pattern example learns, and a seeded run is reproducible."""

import importlib.util
import sys
from pathlib import Path

import torch

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "two_patterns.py"
spec = importlib.util.spec_from_file_location("two_patterns", EXAMPLE)
two_patterns = importlib.util.module_from_spec(spec)
sys.modules["two_patterns"] = two_patterns  # dataclasses resolve annotations through it
spec.loader.exec_module(two_patterns)


def test_each_output_neuron_learns_one_pattern():
    result = two_patterns.run(seed=0, rounds=20)
    preferred = result.preferred()
    assert sorted(preferred) == ["A", "B"]
    assert min(result.selectivity()) >= 0.8


def test_seeded_runs_are_identical():
    first = two_patterns.build(seed=3, rounds=2)
    second = two_patterns.build(seed=3, rounds=2)
    first.net.run(150)
    second.net.run(150)
    assert torch.equal(first.synapse.weights, second.synapse.weights)
    assert torch.equal(first.output.threshold, second.output.threshold)


def test_different_seeds_differ():
    first = two_patterns.build(seed=1, rounds=2)
    second = two_patterns.build(seed=2, rounds=2)
    first.net.run(150)
    second.net.run(150)
    assert not torch.equal(first.synapse.weights, second.synapse.weights)
