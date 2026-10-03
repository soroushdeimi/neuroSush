"""Smoke test of the Brunel (2000) example on a tiny network."""

import importlib.util
import sys
from pathlib import Path

import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_brunel_runs_and_reports_statistics():
    ex = load("brunel_network")
    result = ex.run("AI", n_exc=200, seconds=0.3, transient=0.1)
    assert result["raster"].shape == (600, 200)
    assert 0 < result["rate"] < 500  # refractory period caps the rate at 500 Hz
    assert result["chi"] >= 0


def test_brunel_connectivity_and_delay():
    ex = load("brunel_network")
    net, exc, inh = ex.build(n_exc=200, g=5.0, eta=0.0)  # no external drive
    syn = next(s for s in net.synapses if s.name == "inh_exc")
    assert torch.bincount(syn.dst_idx, minlength=200).eq(5).all()  # epsilon * N_I
    assert float(syn.weights[0]) == 5.0
    # a spike emitted in step 1 moves the target in step 4: D = 1.5 ms = 3 steps
    exc.v.fill_(ex.THETA + 1)
    seen = []
    for _ in range(5):  # the excitatory neurons spike in step 1
        net.run(1)
        seen.append(float(inh.v.max() > 0))
    assert seen == [0.0, 0.0, 0.0, 1.0, 1.0]
