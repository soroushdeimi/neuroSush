"""Eager, ``GraphStepper`` and ``CompiledStepper`` on the Diehl and Cook network.

For each network size and batch mode (unbatched, or ``--members`` independent copies in one
batch) it times ``steps/s`` of the three steppers (median of ``--repeats``) and reports the
compile time of the compiled one, measured from a cold ``torch.compile`` cache, separately.
Needs a CUDA device. Writes JSON and prints a Markdown table.

Run: ``python benchmarks/compiled.py --out compiled.json`` (``--help`` lists the options).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent import load_example

from neurosush.core.compiled import CompiledStepper
from neurosush.core.graph import GraphStepper


def make(dc, mode: str, neurons: int, members: int | None):
    """A fresh network and its stepper; ``compile_s`` is the first-steps time of ``compiled``."""
    model = dc.build_network(neurons, "cuda", members=members, seed=0)
    model.input.rates.fill_(0.03)
    compile_s = 0.0
    if mode == "eager":
        stepper = model.net
    elif mode == "graph":
        stepper = GraphStepper(model.net)
    else:
        torch._dynamo.reset()  # a cold cache: the compile time is part of what is reported
        stepper = CompiledStepper(model.net)
        start = time.perf_counter()
        stepper.run(10)  # compile, warm up and capture
        torch.cuda.synchronize()
        compile_s = time.perf_counter() - start
    stepper.run(30)
    torch.cuda.synchronize()
    return stepper, compile_s


def timed(stepper, steps: int) -> float:
    """Steps/s over ``steps`` steps."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    stepper.run(steps)
    torch.cuda.synchronize()
    return steps / (time.perf_counter() - start)


def main() -> None:
    """Run every combination and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--neurons", default="100,400")
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="compiled.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("needs a CUDA device")
    dc = load_example()
    rows = []
    for neurons in (int(n) for n in args.neurons.split(",")):
        for members in (None, args.members):
            row = {"neurons": neurons, "members": members or 1}
            for mode in ("eager", "graph", "compiled"):
                stepper, compile_s = make(dc, mode, neurons, members)
                rates = [timed(stepper, args.steps) for _ in range(args.repeats)]
                row[mode] = statistics.median(rates)
                if mode == "compiled":
                    row["compile_s"] = compile_s
            rows.append(row)
    machine = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"machine": machine, "steps": args.steps, "results": rows}, f, indent=2)
        f.write("\n")
    print("| neurons | members | eager | graph | compiled | compiled / graph | compile (s) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['neurons']} | {r['members']} | {r['eager']:,.0f} | {r['graph']:,.0f} | "
            f"{r['compiled']:,.0f} | {r['compiled'] / r['graph']:.1f}x | {r['compile_s']:.1f} |"
        )


if __name__ == "__main__":
    main()
