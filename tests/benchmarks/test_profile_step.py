from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks" / "profile_step.py"
SPEC = importlib.util.spec_from_file_location("profile_step", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
profile_step = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile_step)


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_report(report: dict[str, object], *, cuda: bool) -> None:
    wall = report["wall_time_ms_per_step"]
    assert isinstance(wall, dict)
    assert wall["median"] > 0
    assert wall["spread"] >= 0
    assert report["aten_ops"] > 0
    assert report["aten_ops_per_step"] > 0
    top_ops = report["top_ops"]
    assert isinstance(top_ops, list)
    assert top_ops
    sort_key = "self_cuda_us" if cuda else "self_cpu_us"
    times = [op[sort_key] for op in top_ops]
    assert times == sorted(times, reverse=True)
    assert report["cuda_kernels_per_step"] > 0 if cuda else report["cuda_kernels_per_step"] == 0


def test_cpu_smoke_writes_documented_profile(tmp_path: Path) -> None:
    output = tmp_path / "profile.json"
    result = _run(
        tmp_path,
        "--workload",
        "dense_stdp",
        "--device",
        "cpu",
        "--neurons",
        "20",
        "--steps",
        "5",
        "--warmup",
        "1",
        "--stepper",
        "eager",
        "--repeats",
        "1",
        "--out",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["workload"] == "dense_stdp"
    assert report["stepper"] == "eager"
    assert report["machine"]["python"]
    assert len(report["repeats"]) == 1
    _assert_report(report, cuda=False)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--workload", "unknown"), "unknown"),
        (("--workload", "dense_stdp", "--steps", "0"), "steps must be positive, got 0"),
    ],
)
def test_bad_arguments_name_value(tmp_path: Path, arguments: tuple[str, ...], message: str) -> None:
    result = _run(
        tmp_path,
        *arguments,
        "--device",
        "cpu",
        "--out",
        str(tmp_path / "bad.json"),
    )
    assert result.returncode != 0
    assert message in result.stderr


def test_cuda_unavailable_names_requested_device(tmp_path: Path) -> None:
    if torch.cuda.is_available():
        pytest.skip("requires a host without CUDA")
    result = _run(
        tmp_path,
        "--workload",
        "dense_stdp",
        "--device",
        "cuda",
        "--out",
        str(tmp_path / "bad.json"),
    )
    assert result.returncode != 0
    assert "device cuda requested" in result.stderr


@pytest.mark.gpu
def test_cuda_smoke_counts_kernels(tmp_path: Path) -> None:
    output = tmp_path / "profile.json"
    result = _run(
        tmp_path,
        "--workload",
        "dense_stdp",
        "--device",
        "cuda",
        "--neurons",
        "20",
        "--steps",
        "5",
        "--warmup",
        "4",
        "--stepper",
        "eager",
        "--repeats",
        "1",
        "--out",
        str(output),
    )
    assert result.returncode == 0, result.stderr
    _assert_report(json.loads(output.read_text(encoding="utf-8")), cuda=True)
