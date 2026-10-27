from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from neurosush.core.buffers import ArrivalBuffer, HistoryBuffer

_ROOT = Path(__file__).resolve().parents[1]
# No exceptions: Tensor.bool() is a device-side cast, unlike the builtin bool(tensor).
_EXCEPTIONS: dict[str, str] = {}


def _behaviors() -> dict[str, tuple[Path, ast.ClassDef]]:
    classes = {
        node.name: (path, node)
        for path in (_ROOT / "src/neurosush").rglob("*.py")
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef)
    }
    found = {"Behavior": classes["Behavior"]}
    while True:
        added = {
            name: entry
            for name, entry in classes.items()
            if name not in found
            and any(isinstance(base, ast.Name) and base.id in found for base in entry[1].bases)
        }
        if not added:
            return found
        found.update(added)


def _inherited(
    name: str, member: str, classes: dict[str, tuple[Path, ast.ClassDef]]
) -> ast.AST | None:
    node = classes[name][1]
    for child in node.body:
        if isinstance(child, ast.FunctionDef) and child.name == member:
            return child
        if isinstance(child, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == member for t in child.targets
        ):
            return child.value
        if (
            isinstance(child, ast.AnnAssign)
            and isinstance(child.target, ast.Name)
            and child.target.id == member
        ):
            return child.value
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in classes:
            value = _inherited(base.id, member, classes)
            if value is not None:
                return value
    return None


def _host_extractions(node: ast.AST) -> list[int]:
    return [
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and (
            (isinstance(call.func, ast.Attribute) and call.func.attr in {"item", "tolist"})
            or (isinstance(call.func, ast.Name) and call.func.id == "bool")
        )
    ]


def test_graph_safe_forwards_do_not_extract_host_values():
    classes = _behaviors()
    failures = []
    for name in classes:
        safe = _inherited(name, "graph_safe", classes)
        if not isinstance(safe, ast.Constant) or safe.value is not True:
            continue
        forward = _inherited(name, "forward", classes)
        assert forward is not None, name
        if (lines := _host_extractions(forward)) and name not in _EXCEPTIONS:
            failures.append(f"{name}: forward contains host extraction at lines {lines}")
    assert not failures, "\n".join(failures)
    assert all(_EXCEPTIONS.values())


@pytest.mark.parametrize("expression", ["x.item()", "x.tolist()", "bool(x)"])
def test_host_extraction_scanner_detects_forbidden_calls(expression):
    assert _host_extractions(ast.parse(f"def forward(self, x):\n    return {expression}")) == [2]


def test_device_bool_cast_is_allowed():
    assert _host_extractions(ast.parse("x.bool()")) == []


def test_readiness_table_covers_every_behavior():
    document = (_ROOT / "docs/GRAPH_READINESS.md").read_text()
    for name in _behaviors():
        assert f"| `{name}` |" in document, name


class _LegacyValidation:
    def _no_delay(self, delay):
        if isinstance(delay, int):
            return delay == 0
        if self.depth == 1:
            if delay.shape != (self.size,):
                raise ValueError(f"delay shape must be ({self.size},), got {tuple(delay.shape)}")
            return True
        cached = self._checked.get(id(delay))
        if cached is None or cached[0] != delay._version:
            if delay.shape != (self.size,):
                raise ValueError(f"delay shape must be ({self.size},), got {tuple(delay.shape)}")
            if bool(((delay < 0) | (delay >= self.depth)).any()):
                raise ValueError(f"delay must be in [0, {self.depth}), got {delay.tolist()}")
            cached = (delay._version, not bool(delay.any()))
            self._checked[id(delay)] = cached
        return cached[1]


class _LegacyArrival(_LegacyValidation, ArrivalBuffer):
    pass


class _LegacyHistory(_LegacyValidation, HistoryBuffer):
    pass


@pytest.mark.parametrize("batch", [None, 3])
@pytest.mark.parametrize("depth", [1, 2, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_delay_validation_is_bit_exact_for_200_steps(batch, depth, dtype):
    generator = torch.Generator().manual_seed(473)
    pairs = [
        (
            ArrivalBuffer(depth, 5, batch=batch, dtype=dtype),
            _LegacyArrival(depth, 5, batch=batch, dtype=dtype),
        ),
        (HistoryBuffer(depth, 5, batch=batch), _LegacyHistory(depth, 5, batch=batch)),
    ]
    delays = [torch.zeros(5, dtype=torch.long), torch.arange(5) % depth]
    shape = (5,) if batch is None else (batch, 5)
    for step in range(200):
        if step % 11 == 0:
            delays[1].copy_(torch.randint(depth, (5,), generator=generator))
        if step == 97:
            for pair in pairs:
                for buffer in pair:
                    buffer.reset()
        for actual, reference in pairs:
            for delay in (*delays, step % depth):
                value = torch.randn(shape, generator=generator, dtype=dtype)
                before = value.clone()
                if isinstance(actual, ArrivalBuffer):
                    actual.add(value, delay)
                    reference.add(value, delay)
                    assert torch.equal(actual.current(), reference.current())
                else:
                    spikes = value > 0
                    actual.push(spikes)
                    reference.push(spikes)
                    assert torch.equal(actual.read(delay), reference.read(delay))
                assert torch.equal(value, before)
            if isinstance(actual, ArrivalBuffer):
                actual.advance()
                reference.advance()
            left, right = actual.state_dict(), reference.state_dict()
            assert left.keys() == right.keys()
            for key in left:
                if isinstance(left[key], torch.Tensor):
                    assert torch.equal(left[key], right[key]), (step, key)
                else:
                    assert left[key] == right[key], (step, key)
            assert actual._checked == reference._checked


@pytest.mark.parametrize("buffer_type", [ArrivalBuffer, HistoryBuffer])
@pytest.mark.parametrize("bad", [torch.tensor([-1, 0]), torch.tensor([0, 3]), torch.zeros(3)])
def test_delay_validation_keeps_failures_and_revalidates_changes(buffer_type, bad):
    buffer = buffer_type(3, 2)
    delay = torch.zeros(2, dtype=torch.long)
    assert buffer._no_delay(delay)
    delay[1] = 2
    assert not buffer._no_delay(delay)
    delay[1] = 3
    with pytest.raises(ValueError, match=r"delay must be in \[0, 3\), got \[0, 3\]"):
        buffer._no_delay(delay)
    with pytest.raises(ValueError, match="delay"):
        buffer._no_delay(bad)


def test_delay_validation_extracts_both_flags_once_per_version(monkeypatch):
    original = torch.Tensor.tolist
    transfers = []

    def tracked(tensor):
        transfers.append(tuple(tensor.shape))
        return original(tensor)

    def forbidden(tensor):
        raise AssertionError("separate bool(tensor) extraction")

    monkeypatch.setattr(torch.Tensor, "tolist", tracked)
    monkeypatch.setattr(torch.Tensor, "__bool__", forbidden)
    buffer = ArrivalBuffer(3, 2)
    delay = torch.tensor([0, 1])
    assert not buffer._no_delay(delay)
    assert not buffer._no_delay(delay)
    assert transfers == [(2,)]
    delay.zero_()
    assert buffer._no_delay(delay)
    assert transfers == [(2,), (2,)]


def test_float32_dopamine_replacement_changes_eager_bits():
    generator = torch.Generator().manual_seed(731)
    dopamine = 0.1
    candidate = torch.tensor(dopamine, dtype=torch.float32)
    different = False
    for payoff in torch.randn(200, generator=generator, dtype=torch.float64).tolist():
        dopamine += 0.3 * (-dopamine / 7.0 + payoff)
        candidate = candidate + 0.3 * (-candidate / 7.0 + payoff)
        different |= not torch.equal(candidate, torch.tensor(dopamine, dtype=torch.float32))
    assert different
