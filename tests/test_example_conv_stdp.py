"""Smoke test of examples/conv_stdp_mnist.py on synthetic images with tiny layers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

SPEC = importlib.util.spec_from_file_location(
    "conv_stdp_mnist", Path(__file__).resolve().parents[1] / "examples" / "conv_stdp_mnist.py"
)
assert SPEC is not None
assert SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def _images(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bars: label 0 is a horizontal bar, label 1 a vertical one, at random offsets."""
    gen = torch.Generator().manual_seed(0)
    x = torch.zeros(n, 28, 28, dtype=torch.uint8)
    y = torch.arange(n) % 2
    for i in range(n):
        o = int(torch.randint(8, 18, (1,), generator=gen))
        if y[i] == 0:
            x[i, o : o + 3, 4:24] = 255
        else:
            x[i, 4:24, o : o + 3] = 255
    return x, y


def test_pipeline_learns_bounded_weights_and_features() -> None:
    x, y = _images(24)
    times = example.encode(x, threshold=0.1)
    assert times.shape == (24, 2, 28, 28)
    assert int(times.max()) == example.STEPS
    assert int((times < example.STEPS).sum()) > 0

    gen = torch.Generator().manual_seed(0)
    w1 = (0.8 + 0.05 * torch.randn(4, 2, 5, 5, generator=gen)).clamp(0, 1)
    w2 = (0.8 + 0.05 * torch.randn(6, 4, 5, 5, generator=gen)).clamp(0, 1)
    before = w1.clone()
    kw = {"a_plus": 0.004, "a_minus": 0.003, "radius": 2}
    assert example.train_layer(times, w1, 15.0, log="c1", **kw) == 24
    assert not torch.equal(w1, before)
    assert float(w1.min()) >= 0.0
    assert float(w1.max()) <= 1.0

    pooled = example.layer1(times, w1, 15.0)
    assert pooled.shape == (24, 4, 12, 12)
    example.train_layer(pooled, w2, 10.0, log="c2", **kw)
    feats = example.features(times, w1, w2, 15.0)
    assert feats.shape == (24, 6)
    assert bool(torch.isfinite(feats).all())

    model = example.fit_logreg((feats - feats.mean(0)) / feats.std(0).clamp_min(1e-6), y, epochs=5)
    assert model.weight.shape == (10, 6)
