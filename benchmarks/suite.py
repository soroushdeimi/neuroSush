"""Benchmark suite: throughput of the main workloads, normalized by machine speed.

Every case is timed ``--repeats`` times and the median is kept. Raw speed depends on the
machine, so each case is also divided by a calibration workload measured on the same
machine (a loop of small tensor operations: at these sizes a simulation step is dominated
by per-operation overhead). ``compare.py`` checks the normalized numbers against a
baseline recorded on the same kind of machine.

Run: ``python benchmarks/suite.py --out results.json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch
from dense_stdp import build

from neurosush.core.network import Network
from neurosush.htm.sdr import random_sdr
from neurosush.htm.spatial_pooler import SpatialPooler
from neurosush.htm.temporal_memory import TemporalMemory

Case = tuple[Callable[[], Any], Callable[[Any], int]]


def timed(case: Case, repeats: int) -> float:
    """Median rate (units of work per second) over ``repeats`` runs, after one warm-up.

    Every run starts from a fresh state built by the case's setup, which is not timed: a
    learning workload changes as it learns, so repeating it on one state is not a repeat.
    """
    setup, work = case
    work(setup())
    rates = []
    for _ in range(repeats):
        state = setup()
        start = time.perf_counter()
        units = work(state)
        rates.append(units / (time.perf_counter() - start))
    return statistics.median(rates)


def _calibrate(_: None) -> int:
    """Small tensor operations, like those of one simulation step."""
    x = torch.zeros(400)
    for _ in range(2000):
        x = (x * 0.9 + 0.1).clamp(max=1.0)
    return 2000


def _run_network(net: Network) -> int:
    net.run(100)
    return 100 * (net.batch_size or 1)  # sample-steps


def _learn_pooler(state: tuple[SpatialPooler, torch.Tensor]) -> int:
    sp, data = state
    sp.compute(data)
    return len(data)  # learned samples


def _learn_sequence(state: tuple[TemporalMemory, torch.Tensor]) -> int:
    tm, data = state
    for x in data:
        tm.compute(x)
    return len(data)  # steps


def _sdrs(n: int, w: int, count: int) -> torch.Tensor:
    return random_sdr(n, w, batch=(count,), generator=torch.Generator().manual_seed(0))


CALIBRATION: Case = (lambda: None, _calibrate)
CASES: dict[str, Case] = {
    "dense_stdp": (lambda: build("cpu"), _run_network),
    "dense_stdp_batch32": (lambda: build("cpu", 32), _run_network),
    "spatial_pooler_learn": (
        lambda: (
            SpatialPooler(784, 1024, potential_radius=784, density=0.04, boost_strength=1.0),
            _sdrs(784, 40, 20),
        ),
        _learn_pooler,
    ),
    "temporal_memory_learn": (
        lambda: (
            TemporalMemory(1024, 16, activation_threshold=10, min_threshold=8),
            _sdrs(1024, 20, 25),
        ),
        _learn_sequence,
    ),
}


def run(repeats: int) -> dict[str, object]:
    """Rates of every case, raw and divided by the calibration rate."""
    torch.set_num_threads(1)  # thread counts differ between machines; one is comparable
    reference = timed(CALIBRATION, repeats)
    cases = {}
    for name, case in CASES.items():
        rate = timed(case, repeats)
        cases[name] = {"rate": rate, "normalized": rate / reference}
    return {
        "calibration": reference,
        "cases": cases,
        "machine": {"python": platform.python_version(), "torch": torch.__version__},
    }


def main() -> None:
    """Run the suite and write the results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", default="benchmark.json")
    args = parser.parse_args()
    results = run(args.repeats)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    for name, case in results["cases"].items():  # type: ignore[attr-defined]
        print(f"{name:24s} {case['rate']:10.1f}/s  normalized {case['normalized']:.4f}")


if __name__ == "__main__":
    main()
