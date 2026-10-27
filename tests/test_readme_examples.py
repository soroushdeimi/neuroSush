"""Executable tests for the README examples."""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from neurosush.neurons import dynamics

README = Path(__file__).parents[1] / "README.md"
SINGLE_NEURON_HEADING = "## Single neuron without a network"


def test_single_neuron_example_prints_a_rate_close_to_lif_rate() -> None:
    """Run the standalone LIF example and compare its five-second rate window."""
    readme = README.read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(SINGLE_NEURON_HEADING)}\n.*?^```python\n(.*?)^```$",
        readme,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    assert len(match.group(1).splitlines()) == 10

    stdout = io.StringIO()
    namespace: dict[str, object] = {}
    with redirect_stdout(stdout):
        exec(compile(match.group(1), "README single-neuron example", "exec"), namespace)

    printed_rate, lif_rate = map(float, stdout.getvalue().split())
    # One decimal place rounds each rate by at most 0.05 Hz.
    rounding_tolerance = 0.05
    assert printed_rate == pytest.approx(49.4, abs=rounding_tolerance, rel=0)
    comment = re.search(r"# Prints: ([0-9.]+) ([0-9.]+)$", match.group(1).strip())
    assert comment is not None
    comment_rates = tuple(map(float, comment.groups()))
    assert comment_rates == pytest.approx((printed_rate, lif_rate), abs=rounding_tolerance, rel=0)
    analytic_rate = dynamics.lif_rate(
        tau=20.0,
        v_rest=-70.0,
        v_reset=-75.0,
        threshold=-40.0,
        resistance=10.0,
        current=5.0,
    )
    assert lif_rate == pytest.approx(analytic_rate, abs=rounding_tolerance, rel=0)
    # The 50,000 steps of 0.1 ms span five seconds.
    window_seconds = 50_000 * 0.1 / 1000
    spikes = namespace["spikes"]
    assert isinstance(spikes, int)
    counted_rate = spikes / window_seconds
    assert printed_rate == pytest.approx(counted_rate, abs=rounding_tolerance, rel=0)
    assert abs(counted_rate - analytic_rate) / analytic_rate < 0.05
