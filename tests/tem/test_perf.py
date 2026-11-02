"""Regression fixtures and hot-path operation counts.

Golden generated on macOS arm64; CPU kernels differ across platforms.
Measured split-reduction noise: outputs 1.20e-7, gradients 3.82e-6 max absolute.
History 1.87e-9 / parameters 2.99e-8; tolerances allow at least 10x measured noise.
"""

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from neurosush.tem import TEM, TEMConfig
from neurosush.tem.train import train

GOLDEN = Path(__file__).parent / "golden" / "perf.pt"


def snapshot(case: int) -> dict[str, Any]:
    flags = case >= 2
    config = TEMConfig(
        sensory=3,
        compressed=3,
        modules=(4, 2),
        hidden=4,
        batch=2,
        iterations=2,
        transition="rotation" if case % 2 else "matrix",
        inference_memory_loss=flags,
        calibrated_reductions=flags,
        bounded_attractor_activation=flags,
        reliable_fusion=flags,
    )
    model = TEM(config, generator=torch.Generator().manual_seed(31))
    state = model.reset(2)
    outputs = []
    for t in range(3):
        override = tuple(torch.full_like(g, 0.25) for g in state.g) if case == 4 else None
        output, state = model(
            torch.tensor([t, 2 - t]), torch.tensor([t, 4]), state, prior_override=override
        )
        outputs.extend(getattr(output, f.name) for f in fields(output))
    torch.autograd.backward(sum(x.square().sum() for x in outputs if x is not None))
    return {
        "outputs": [x.detach() for x in outputs],
        "gradients": {
            n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()
        },
    }


@pytest.mark.parametrize("case", range(5))
def test_golden(case: int) -> None:
    expected = torch.load(GOLDEN, weights_only=True)[case]
    actual = snapshot(case)
    for a, b in zip(actual["outputs"], expected["outputs"], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=2e-6, equal_nan=False)
    for name, gradient in actual["gradients"].items():
        reference = expected["gradients"][name]
        if reference is None:
            assert gradient is None
        else:
            assert gradient is not None
            torch.testing.assert_close(gradient, reference, rtol=5e-4, atol=4e-5, equal_nan=False)


class ScalarProbe(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.equal_count = 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if str(func) == "aten.equal.default":
            self.equal_count += 1
        if "_local_scalar_dense" in str(func):
            self.count += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("rotation", [False, True])
def test_forward_has_no_scalar_extractions(rotation: bool) -> None:
    model = TEM(
        TEMConfig(transition="rotation" if rotation else "matrix"),
        generator=torch.Generator().manual_seed(3),
    )
    state = model.reset(2)
    with ScalarProbe() as probe:
        model(torch.tensor([0, 19]), torch.tensor([0, 4]), state)
    assert probe.count == 0


def training_snapshot() -> dict[str, Any]:
    result = train(
        TEMConfig(
            sensory=3,
            compressed=3,
            modules=(4, 2),
            hidden=4,
            batch=2,
            iterations=1,
            widths=(2,),
            walk_len=5,
            bptt=2,
        ),
        steps=7,
        seed=12,
    )
    return {"history": result.history, "parameters": result.model.state_dict()}


def test_training_golden() -> None:
    expected = torch.load(GOLDEN, weights_only=True)["training"]
    actual = training_snapshot()
    assert len(actual["history"]) == len(expected["history"])
    for row, reference in zip(actual["history"], expected["history"], strict=True):
        assert row.keys() == reference.keys()
        for name, value in row.items():
            assert type(value) is type(reference[name])
            torch.testing.assert_close(
                torch.tensor(value, dtype=torch.float64),
                torch.tensor(reference[name], dtype=torch.float64),
                rtol=7e-4,
                atol=1e-6,
                equal_nan=False,
            )
    for name, value in actual["parameters"].items():
        torch.testing.assert_close(
            value, expected["parameters"][name], rtol=1e-4, atol=1e-6, equal_nan=False
        )


def test_staged_walk_uses_trusted_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    training = importlib.import_module("neurosush.tem.train")
    config = TEMConfig(sensory=3, compressed=3, modules=(4,), batch=2, widths=(2,))
    model = TEM(config, generator=torch.Generator().manual_seed(5))
    walk = training.batched_random_walks(training._worlds(2, config, 4), 3)
    staged = training._stage_walk(model, walk)
    assert torch.equal(staged.observations, walk.observations)
    assert torch.equal(staged.executed_actions, walk.executed_actions)

    def reject(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("per-step validation")

    monkeypatch.setattr(model, "_validate_ranges", reject)
    with ScalarProbe() as probe:
        training._step(model, staged, 0, model.reset(2), decay=0.9, rate=0.2, oracle=False)
    assert probe.count == 0
    assert probe.equal_count == 0


def test_benchmark_advances_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "benchmarks"))
    speed = importlib.import_module("tem_speed")
    config = TEMConfig(
        sensory=3, compressed=3, modules=(4,), hidden=3, batch=2, bptt=2, widths=(2,), iterations=1
    )
    monkeypatch.setattr(speed, "TEMConfig", lambda **kwargs: config)
    walk = speed.batched_random_walks(speed._worlds(2, config, 42), 7, stay=False, valid_only=True)
    monkeypatch.setattr(speed, "batched_random_walks", lambda *args, **kwargs: walk)
    observed = []
    original = TEM.forward

    def capture(self: TEM, observation: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        observed.append(observation.clone())
        return original(self, observation, *args, **kwargs)

    monkeypatch.setattr(TEM, "forward", capture)
    assert speed._benchmark_tem("cpu", 2, 2, 2, 1, 0) > 0
    assert len(observed) == 6
    assert torch.equal(torch.stack(observed, 1), walk.observations[:, 1:7])


def test_profiler_event_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "benchmarks"))
    profiler = importlib.import_module("tem_profile")
    events = [
        SimpleNamespace(key=k, count=n)
        for k, n in [
            ("aten::add", 7),
            ("aten::mul", 3),
            ("autograd::engine", 4),
            ("Optimizer.step#Adam.step", 2),
            ("other", 99),
        ]
    ]
    assert profiler._event_counts(events) == (10, 4, 2)


def test_bounded_suite_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "benchmarks"))
    suite = importlib.import_module("suite")
    before = torch.random.get_rng_state().clone()
    for name, units in [("tem_forward", 4), ("spatial_autocorrelogram", 1)]:
        setup, work = suite.CASES[name]
        assert work(setup()) == units
        assert suite.CASE_METADATA[name]["device"] == "cpu"
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("diagnostics", [False, True])
def test_metrics_transfer_once_per_update(
    monkeypatch: pytest.MonkeyPatch, diagnostics: bool
) -> None:
    transfers = []
    original = torch.Tensor.cpu

    def capture(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        transfers.append(tensor.shape)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", capture)
    result = train(
        TEMConfig(
            sensory=3,
            compressed=3,
            modules=(2,),
            hidden=3,
            batch=2,
            widths=(2,),
            walk_len=3,
            bptt=2,
            iterations=1,
        ),
        steps=4,
        seed=0,
        diagnostics=diagnostics,
        log_grads=diagnostics,
    )
    assert len(result.history) == 3
    assert len(transfers) == 3
    assert all(len(shape) == 1 for shape in transfers)
