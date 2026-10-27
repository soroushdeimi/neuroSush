"""Profile one simulation step by workload, device, and stepper.

The report separates wall time from operator and CUDA-kernel activity. Run, for example,
``python benchmarks/profile_step.py --workload dense_stdp --device cpu --neurons 20
--steps 5 --warmup 2 --stepper eager --out profile.json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_stdp import build as build_dense_stdp  # noqa: E402
from independent import load_example  # noqa: E402

from neurosush.core.compiled import CompiledStepper  # noqa: E402
from neurosush.core.graph import GraphStepper  # noqa: E402

REPEATS = 5
TOP_OPS = 15


def _device_time_us(event: Any) -> float:
    # torch before 2.4 names it self_cuda_time_total
    if hasattr(event, "self_device_time_total"):
        return float(event.self_device_time_total)
    return float(event.self_cuda_time_total)


def _positive(name: str) -> Callable[[str], int]:
    def parse(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"{name} must be positive, got {parsed}")
        return parsed

    return parse


def _non_negative(name: str) -> Callable[[str], int]:
    def parse(value: str) -> int:
        parsed = int(value)
        if parsed < 0:
            raise argparse.ArgumentTypeError(f"{name} must be non-negative, got {parsed}")
        return parsed

    return parse


def _build(workload: str, device: str, neurons: int) -> Any:
    if workload == "dense_stdp":
        return build_dense_stdp(device, inputs=784, outputs=neurons)
    dc = load_example()
    model = dc.build_network(neurons, device, seed=0)
    model.input.rates.fill_(0.03)
    return model.net


def _make_stepper(net: Any, stepper: str) -> Any:
    if stepper == "eager":
        return net
    if stepper == "graph":
        return GraphStepper(net)
    return CompiledStepper(net)


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _timed_repeat(stepper: Any, steps: int, device: str) -> float:
    _synchronize(device)
    start = time.perf_counter()
    stepper.run(steps)
    _synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / steps


def _operator_profile(stepper: Any, steps: int, device: str) -> dict[str, Any]:
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as recorded:
        stepper.run(steps)
        _synchronize(device)

    events = recorded.events()
    cuda_events = [
        event
        for event in events
        if str(event.device_type).endswith("CUDA")
        and not event.name.casefold().startswith(("memcpy", "memset"))
    ]
    kernel_total_us = sum(_device_time_us(event) for event in cuda_events)
    averages = list(recorded.key_averages())
    aten = [event for event in averages if event.key.startswith("aten::")]
    total_cpu_us = sum(event.self_cpu_time_total for event in averages)
    total_cuda_us = sum(_device_time_us(event) for event in averages)
    key = (
        (lambda event: _device_time_us(event))
        if device == "cuda"
        else (lambda event: event.self_cpu_time_total)
    )
    top_ops = []
    for event in sorted(aten, key=key, reverse=True)[:TOP_OPS]:
        top_ops.append(
            {
                "name": event.key,
                "calls": event.count,
                "self_cpu_us": event.self_cpu_time_total,
                "self_cpu_percent": (
                    100.0 * event.self_cpu_time_total / total_cpu_us if total_cpu_us else 0.0
                ),
                "self_cuda_us": _device_time_us(event) if device == "cuda" else 0.0,
                "self_cuda_percent": (
                    100.0 * _device_time_us(event) / total_cuda_us
                    if device == "cuda" and total_cuda_us
                    else 0.0
                ),
            }
        )
    return {
        "aten_ops": sum(event.count for event in aten),
        "aten_ops_per_step": sum(event.count for event in aten) / steps,
        "cuda_kernels": len(cuda_events),
        "cuda_kernels_per_step": len(cuda_events) / steps,
        "cuda_kernel_time_us_per_step": kernel_total_us / steps,
        "top_ops": top_ops,
    }


def profile_workload(
    workload: str,
    device: str,
    neurons: int,
    steps: int,
    warmup: int,
    stepper_name: str,
    *,
    repeats: int = REPEATS,
) -> dict[str, Any]:
    """Profile one workload and return JSON-serializable measurements.

    Args:
        workload: ``dense_stdp`` or ``diehl_cook``.
        device: ``cpu`` or ``cuda``.
        neurons: Number of output or excitatory neurons.
        steps: Number of steps in each measured window.
        warmup: Number of unmeasured warm-up steps.
        stepper_name: ``eager``, ``graph``, or ``compiled``.
        repeats: Number of independent wall-time windows.

    Returns:
        Workload configuration, repeat timings, profiler counts, and top operators.
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    net = _build(workload, device, neurons)
    stepper = _make_stepper(net, stepper_name)
    stepper.run(warmup)
    _synchronize(device)
    repeat_ms = [_timed_repeat(stepper, steps, device) for _ in range(repeats)]
    wall_median = statistics.median(repeat_ms)
    operations = _operator_profile(stepper, steps, device)
    kernel_ms = operations["cuda_kernel_time_us_per_step"] / 1000.0
    launch_bound_ratio = (wall_median - kernel_ms) / wall_median if device == "cuda" else None
    return {
        "workload": workload,
        "device": device,
        "stepper": stepper_name,
        "neurons": neurons,
        "steps": steps,
        "warmup": warmup,
        "repeats": repeat_ms,
        "wall_time_ms_per_step": {
            "median": wall_median,
            "min": min(repeat_ms),
            "max": max(repeat_ms),
            "spread": max(repeat_ms) - min(repeat_ms),
        },
        "launch_bound_ratio": launch_bound_ratio,
        **operations,
    }


def _cpu_info() -> tuple[str | None, float | None]:
    model = None
    mhz = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name") and model is None:
                model = line.split(":", 1)[1].strip()
            elif line.startswith("cpu MHz") and mhz is None:
                mhz = float(line.split(":", 1)[1])
    except (OSError, ValueError, IndexError):
        pass
    return model, mhz


def _driver_version() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.splitlines()[0].strip()


def machine_info() -> dict[str, Any]:
    """Return runtime, CPU, and available CUDA device information.

    Returns:
        JSON-serializable machine metadata and frequency warnings.
    """
    cpu_model, cpu_mhz = _cpu_info()
    warnings = []
    if cpu_mhz is not None and cpu_mhz < 1500:
        warnings.append(f"cpu MHz below 1500 under load: {cpu_mhz:.1f}")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_mhz": cpu_mhz,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
        "driver": _driver_version(),
        "warnings": warnings,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", choices=("dense_stdp", "diehl_cook", "all"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--neurons", type=_positive("neurons"), default=400)
    parser.add_argument("--steps", type=_positive("steps"), default=100)
    parser.add_argument("--warmup", type=_non_negative("warmup"), default=20)
    parser.add_argument("--stepper", choices=("eager", "graph", "compiled"), default="eager")
    parser.add_argument("--repeats", type=_positive("repeats"), default=REPEATS)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    """Parse arguments, run the requested profiles, and write JSON.

    Returns:
        Zero after writing a report; argument errors exit non-zero.
    """
    parser = _parser()
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(f"device cuda requested but CUDA is unavailable (got {args.device!r})")
    if args.stepper == "graph" and args.device != "cuda":
        parser.error(f"stepper graph requires device cuda, got {args.device!r}")
    workloads = ("dense_stdp", "diehl_cook") if args.workload == "all" else (args.workload,)
    results = [
        profile_workload(
            workload,
            args.device,
            args.neurons,
            args.steps,
            args.warmup,
            args.stepper,
            repeats=args.repeats,
        )
        for workload in workloads
    ]
    report: dict[str, Any]
    if len(results) == 1:
        report = results[0]
        report["machine"] = machine_info()
    else:
        report = {
            "machine": machine_info(),
            "workload": "all",
            "device": args.device,
            "stepper": args.stepper,
            "results": results,
        }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for warning in report["machine"]["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
