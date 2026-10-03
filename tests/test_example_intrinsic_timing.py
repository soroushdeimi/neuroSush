"""Smoke test of the intrinsic-timing ramps example on a tiny network."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_schedules_follow_the_protocols():
    ex = load("intrinsic_timing_ramps")
    rng = np.random.default_rng(0)
    on, trials = ex.make_schedule("fixed_jitter", 4000, rng, p_deviant=0.2)
    assert on.sum() == ex.STIM_BINS * len(trials)
    fixed = [t["isi"] for t in trials[1:] if t["context"] == "fixed" and not t["deviant"]]
    assert set(fixed) == {30}
    assert {t["isi"] for t in trials if t["deviant"]} == {15}
    _, trials = ex.make_schedule("short_long", 1500, rng)
    assert {t["isi"] for t in trials[1:]} <= {20, 40}


def test_exponential_fit_recovers_tau():
    ex = load("intrinsic_timing_ramps")
    t = (np.arange(30) + 0.5) * ex.BIN / 1000
    trace = (1 + 4 * np.exp(-t / 0.5))[:, None]
    tau, amp, r2 = ex.fit_exponential(trace, np.geomspace(0.05, 3, 40))
    assert abs(tau[0] - 0.5) < 0.06
    assert amp[0] > 0
    assert r2[0] > 0.99


def test_stimulus_ramps_in_a_tiny_network():
    ex = load("intrinsic_timing_ramps")
    out = ex.simulate(["random"], 12.0, n_a=20, n_b=20, n_i=8, calcium_tau=0.0)
    assert out["a"].shape == (1, 240, 20)
    assert out["b"].shape == (1, 240, 20)
    x = ex._activity(out)
    assert np.isfinite(x).all()
    trials = out["trials"][0]
    pre = np.mean([x[0, t["onset"] - 4 : t["onset"], :20].mean() for t in trials])
    during = np.mean([x[0, t["onset"] + 1 : t["offset"], :20].mean() for t in trials])
    assert during > pre  # the stimulus drives the A cells
