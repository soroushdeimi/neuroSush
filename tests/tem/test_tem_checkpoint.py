"""Checkpoint round trips use real optimizer updates."""

import pytest
import torch

from neurosush.tem import TEMConfig, train
from neurosush.tem.checkpoint import load_checkpoint, save_checkpoint


def test_roundtrip(tmp_path, monkeypatch):
    config = TEMConfig(batch=1, bptt=2, walk_len=4, modules=(4,), widths=(2,))
    path = tmp_path / "tiny.pt"
    import importlib

    from neurosush.tem import checkpoint

    training = importlib.import_module("neurosush.tem.train")
    calls = []

    def record(path, model, *, seed, updates):
        calls.append(updates)
        checkpoint.save_checkpoint(path, model, seed=seed, updates=updates)

    monkeypatch.setattr(training, "save_checkpoint", record)
    result = train(config, steps=4, seed=7, checkpoint_path=str(path), checkpoint_every=1)
    assert calls == [1, 2, 2]
    loaded, seed, updates = load_checkpoint(path)
    assert (seed, updates) == (7, 2)
    observation, action = torch.tensor([1]), torch.tensor([4])
    with torch.no_grad():
        expected, _ = result.model(observation, action, result.model.reset(1))
        actual, _ = loaded(observation, action, loaded.reset(1))
    torch.testing.assert_close(actual.g_inf, expected.g_inf, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual.p_inf, expected.p_inf, atol=1e-7, rtol=1e-6)
    with pytest.raises(ValueError, match="nonnegative"):
        save_checkpoint(path, loaded, seed=0, updates=-1)
    with pytest.raises(ValueError, match="checkpoint_every"):
        train(config, steps=1, seed=0, checkpoint_every=0)
