"""How throughput scales with batch size, device, CUDA graphs and CPU threads.

Times the 784 -> 400 dense STDP network of ``dense_stdp.py`` for every combination of
device, batch size and (on the CPU) thread count. Every run builds a fresh network, warms it
up for 20 steps and times ``--steps`` steps; the median of ``--repeats`` runs is reported.
Writes JSON and prints a Markdown table.

Run: ``python benchmarks/scaling.py --out scaling.json`` (``--help`` lists the options).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics

import torch
from dense_stdp import steps_per_second


def parse_batches(text: str) -> list[int | None]:
    """``"none,8,32"`` -> ``[None, 8, 32]`` (``none`` is an unbatched network)."""
    return [None if item == "none" else int(item) for item in text.split(",")]


def measure(
    device: str, batch: int | None, steps: int, repeats: int, *, graph: bool = False
) -> dict[str, object]:
    """Median steps/s of ``repeats`` fresh runs, or the error that stopped them."""
    try:
        rates = [steps_per_second(device, steps, batch, graph=graph) for _ in range(repeats)]
    except RuntimeError as error:  # for example CUDA out of memory
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return {"error": str(error)[:300]}
    rate = statistics.median(rates)
    return {
        "steps_per_s": rate,
        "sample_steps_per_s": rate * (batch or 1),
        "runs": rates,
    }


def main() -> None:
    """Run every combination and write the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu-batches", default="none,8,32,128,512,2048")
    parser.add_argument("--cpu-batches", default="none,8,32,128")
    parser.add_argument("--cpu-threads", default=f"1,{torch.get_num_threads()}")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="scaling.json")
    args = parser.parse_args()
    rows = []
    if torch.cuda.is_available():
        for mode in ("eager", "graph"):
            for batch in parse_batches(args.gpu_batches):
                result = measure("cuda", batch, args.steps, args.repeats, graph=mode == "graph")
                rows.append(
                    {
                        "device": "cuda",
                        "mode": mode,
                        "threads": None,
                        "batch": batch,
                        **result,
                    }
                )
    for threads in (int(t) for t in args.cpu_threads.split(",")):
        torch.set_num_threads(threads)
        for batch in parse_batches(args.cpu_batches):
            result = measure("cpu", batch, args.steps, args.repeats)
            rows.append(
                {
                    "device": "cpu",
                    "mode": "eager",
                    "threads": threads,
                    "batch": batch,
                    **result,
                }
            )
    machine = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"machine": machine, "steps": args.steps, "results": rows}, f, indent=2)
        f.write("\n")
    print("| device | mode | threads | batch | steps/s | sample-steps/s |")
    print("|---|---|---:|---:|---:|---:|")
    for row in rows:
        batch = "unbatched" if row["batch"] is None else row["batch"]
        head = f"| {row['device']} | {row['mode']} | {row['threads'] or ''} | {batch} |"
        if "error" in row:
            print(f"{head} error | |")
        else:
            print(f"{head} {row['steps_per_s']:,.0f} | {row['sample_steps_per_s']:,.0f} |")


if __name__ == "__main__":
    main()
