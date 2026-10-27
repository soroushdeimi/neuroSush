"""Regression tests for recording and exporting replay data."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import torch

from neurosush.core.network import Network, NeuronGroup
from neurosush.neurons.models import Izhikevich
from neurosush.viz import Trace, TraceRecorder, render_plate, schedule_lanes, schedule_table


@pytest.mark.parametrize(("interval", "steps"), [(1, 500), (3, 9), (3, 0)])
def test_recorded_html_decodes_in_node(interval: int, steps: int) -> None:
    rec = TraceRecorder("v", "spikes", interval=interval, max_steps=500)
    net = Network(seed=42)
    group = NeuronGroup(net, 5, [Izhikevich(a=0.02, b=0.2, c=-65, d=8), rec])
    net.initialize()
    group.spikes[4] = True
    net.run(steps)
    trace = rec.trace(**Trace.make_meta(net))
    html = render_plate("demo", trace.to_dict())
    match = re.search(r'id="atlas-trace">(.*?)</script>', html)
    assert match is not None
    decoder = Path(__file__).resolve().parents[2] / "src/neurosush/viz/web/atlas.js"
    script = """
const assert = require('node:assert/strict');
const Atlas = require(process.argv[1]);
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const trace = new Atlas.Trace(input);
assert.deepEqual(trace.indices, input.steps);
assert.equal(trace.steps, input.steps.length);
assert.equal(trace.meta.library, input.meta.library);
assert.deepEqual(trace.events, input.steps.map((_, i) => i));
const clock = new Atlas.Clock({ steps: trace.steps, events: trace.events });
clock.seek(Infinity);
assert.equal(clock.t, Math.max(0, input.steps.length - 1));
if (trace.steps) {
  assert.equal(trace.at('v', clock.t).length, 5);
  assert.equal(trace.at('spikes', clock.t)[4], 1);
}
"""
    subprocess.run(["node", "-e", script, str(decoder)], input=match[1], text=True, check=True)
    assert json.loads(match[1])["steps"] == list(range(interval, steps + 1, interval))


@pytest.mark.parametrize("bad", [torch.zeros(2, 3), torch.zeros(5, dtype=torch.float64), 1.0])
def test_recorder_invalid_sample_is_recoverable(bad: torch.Tensor | float) -> None:
    rec = TraceRecorder("v", "u", max_steps=2)
    net = Network()
    group = NeuronGroup(net, 5, [Izhikevich(a=0.02, b=0.2, c=-65, d=8), rec], name="cells")
    net.run(1)
    before = rec.trace()
    correct = group.u
    group.u = bad
    net.iteration = 2
    with pytest.raises(RuntimeError) as exc:
        rec.forward(group)
    message = str(exc.value)
    for token in ("cells", "u", "expected shape (5,)", "dtype torch.float32", "got shape"):
        assert token in message
    assert str(tuple(torch.as_tensor(bad).shape)) in message
    assert str(torch.as_tensor(bad).dtype) in message
    assert rec.steps == [1]
    assert rec._count == 1
    assert torch.equal(rec.trace().series["v"], before.series["v"])
    group.u = correct
    rec.forward(group)
    assert rec.steps == [1, 2]
    assert rec.trace().series["u"].shape == (2, 5)
    net.iteration = 3
    with pytest.raises(RuntimeError, match=r"cells.*max_steps=2.*iteration 3.*increase max_steps"):
        rec.forward(group)
    assert rec.steps == [1, 2]


@pytest.mark.parametrize(
    ("keyword", "value", "allowed"),
    [
        ("theme", "sepia", ("auto", "light", "dark")),
        ("source", "unknown", ("live", "replay", "sweep")),
    ],
)
def test_export_invalid_options(keyword: str, value: str, allowed: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match=keyword) as exc:
        render_plate("demo", {}, **{keyword: value})
    assert value in str(exc.value)
    for option in allowed:
        assert option in str(exc.value)


@pytest.mark.parametrize("theme", ["auto", "light", "dark"])
def test_export_allowed_themes(theme: str) -> None:
    html = render_plate("demo", {"steps": [], "series": {}}, theme=theme)
    assert '<html lang="en" dir="ltr"' in html
    if theme != "auto":
        assert f'data-theme="{theme}"' in html


def test_schedule_recorder_dependencies_and_dynamics() -> None:
    net = Network()
    NeuronGroup(
        net, 2, [Izhikevich(a=0.02, b=0.2, c=-65, d=8), TraceRecorder("v", "u", max_steps=2)]
    )
    NeuronGroup(
        net, 3, [Izhikevich(a=0.02, b=0.2, c=-65, d=8), TraceRecorder("spikes", max_steps=2)]
    )
    lanes = {lane["name"]: lane["order"] for lane in schedule_lanes()}
    rows = schedule_table(net)
    for row in rows:
        assert row["order"] == lanes[row["lane"]]
    models = [row for row in rows if row["behavior"] == "Izhikevich"]
    assert len(models) == 2
    for row in models:
        assert row["reads"] == ["v", "u", "I"]
        assert row["writes"] == ["v", "u"]
    recorders = [row for row in rows if row["behavior"] == "TraceRecorder"]
    assert [row["reads"] for row in recorders] == [["v", "u"], ["spikes"]]
    assert all(row["writes"] == [] for row in recorders)
