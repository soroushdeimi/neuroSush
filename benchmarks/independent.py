"""Throughput of independent networks in one batch (``Network(independent=True)``).

Times the Diehl and Cook network (784 Poisson inputs, N excitatory and N inhibitory
conductance LIF neurons, dense STDP, adaptive thresholds) with ``--members`` independent
copies per batch, eager and with a CUDA graph. ``steps/s`` counts batch steps;
``member-steps/s`` is ``steps/s`` times the number of members. The unbatched network is
listed as ``members = 1 (single)``. Writes JSON and prints a Markdown table.

Run: ``python benchmarks/independent.py --out independent.json`` (``--help`` lists the options).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "diehl_cook_mnist.py"


def load_example():
    """Import ``examples/diehl_cook_mnist.py`` for its ``build_network``."""
    spec = importlib.util.spec_from_file_location("diehl_cook_mnist", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up
    spec.loader.exec_module(module)
    return module


def steps_per_second(
    dc, device: str, members: int | None, neurons: int, steps: int, *, graph: bool
) -> float:
    """Steps/s of a fresh network: ``members`` is ``None`` for one unbatched network."""
    model = dc.build_network(neurons, device, members=members, graph=graph, seed=0)
    model.input.rates.fill_(0.03)  # about 30 Hz at dt = 1 ms on every pixel
    stepper = model.stepper
    stepper.run(20)  # warm up (and capture the graph)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    stepper.run(steps)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return steps / (time.perf_counter() - start)


def main() -> None:
    """Run every combination and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--members", default="1,4,16,64")
    parser.add_argument("--neurons", type=int, default=100)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="independent.json")
    args = parser.parse_args()
    dc = load_example()
    modes = ("eager", "graph") if args.device.startswith("cuda") else ("eager",)
    cases: list[tuple[str, int | None]] = []
    for mode in modes:
        cases.append((mode, None))
        cases.extend((mode, int(m)) for m in args.members.split(","))
    rows = []
    for mode, members in cases:
        rates = [
            steps_per_second(
                dc, args.device, members, args.neurons, args.steps, graph=mode == "graph"
            )
            for _ in range(args.repeats)
        ]
        rate = statistics.median(rates)
        rows.append(
            {
                "mode": mode,
                "members": members or 1,
                "single": members is None,
                "steps_per_s": rate,
                "member_steps_per_s": rate * (members or 1),
                "runs": rates,
            }
        )
    machine = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {"machine": machine, "neurons": args.neurons, "steps": args.steps, "results": rows},
            f,
            indent=2,
        )
        f.write("\n")
    print("| device | mode | members | steps/s | member-steps/s |")
    print("|---|---|---:|---:|---:|")
    for row in rows:
        label = f"{row['members']} (single)" if row["single"] else row["members"]
        print(
            f"| {args.device} | {row['mode']} | {label} | {row['steps_per_s']:,.0f} | "
            f"{row['member_steps_per_s']:,.0f} |"
        )


if __name__ == "__main__":
    main()
