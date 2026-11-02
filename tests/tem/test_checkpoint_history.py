"""Tests for snapshot checkpointing, metadata, and backward compatibility."""

from dataclasses import asdict
from pathlib import Path

import torch

from neurosush.tem import TEM, TEMConfig, train
from neurosush.tem.checkpoint import load_checkpoint, read_metadata


def test_checkpoint_snapshots_and_metadata(tmp_path: Path) -> None:
    config = TEMConfig(batch=1, bptt=2, walk_len=10, modules=(4,), widths=(2,))
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_path = ckpt_dir / "v2-s1.pt"

    # Tiny run with --updates 600 --snapshots 100,300 --checkpoint-every 100
    train(
        config,
        steps=600 * config.bptt,
        seed=1,
        checkpoint_path=str(ckpt_path),
        checkpoint_every=100,
        snapshots="100,300",
    )

    snap100 = ckpt_dir / "v2-s1-u100.pt"
    snap300 = ckpt_dir / "v2-s1-u300.pt"
    snap600 = ckpt_dir / "v2-s1-u600.pt"
    final_pt = ckpt_dir / "v2-s1.pt"

    assert snap100.exists()
    assert snap300.exists()
    assert snap600.exists()
    assert final_pt.exists()

    for path in (snap100, snap300, snap600, final_pt):
        model, seed, _updates = load_checkpoint(path)
        assert seed == 1
        assert model is not None

        meta = read_metadata(path)
        assert "config" in meta
        assert "seed" in meta
        assert "updates" in meta or "update" in meta
        assert "git_sha" in meta
        assert "library_version" in meta
        assert "torch_version" in meta
        assert meta["seed"] == 1


def test_old_format_checkpoint_backward_compatibility(tmp_path: Path) -> None:
    config = TEMConfig(batch=1, bptt=2, walk_len=4, modules=(4,), widths=(2,))
    model = TEM(config, generator=torch.Generator().manual_seed(0))
    path = tmp_path / "old.pt"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(model.config),
            "seed": 42,
            "updates": 50,
        },
        path,
    )

    loaded_model, seed, updates = load_checkpoint(path)
    assert seed == 42
    assert updates == 50
    assert loaded_model is not None

    meta = read_metadata(path)
    assert meta["seed"] == 42
    assert meta["updates"] == 50
    assert meta["git_sha"] == "unknown"


def test_proving_test_can_fail(tmp_path: Path) -> None:
    config = TEMConfig(batch=1, bptt=2, walk_len=10, modules=(4,), widths=(2,))
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_path = ckpt_dir / "v2-s1.pt"

    # Red phase: disable snapshots, assert snapshot file missing
    train(
        config,
        steps=200 * config.bptt,
        seed=1,
        checkpoint_path=str(ckpt_path),
        checkpoint_every=100,
        snapshots="",
    )
    snap100 = ckpt_dir / "v2-s1-u100.pt"
    assert not snap100.exists()

    # Green phase: restore snapshots, assert snapshot file exists
    train(
        config,
        steps=200 * config.bptt,
        seed=1,
        checkpoint_path=str(ckpt_path),
        checkpoint_every=100,
        snapshots="100",
    )
    assert snap100.exists()
