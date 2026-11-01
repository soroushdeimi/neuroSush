"""Tests for Hebbian memory and attractor dynamics."""

from __future__ import annotations

import pytest
import torch

from neurosush.memory import HebbianMemory, attractor, hebbian_update


def test_hebbian_update_hand_computed() -> None:
    # n = 2
    # memory = [[0.1, 0.2], [0.3, 0.4]]
    # pattern = [1.0, 0.0], predicted = [0.0, 0.0]
    # decay = 0.9, rate = 0.5, clip = 1.0
    # outer = [1, 0]^T * [1, 0] = [[1, 0], [0, 0]]
    # rate * outer = [[0.5, 0], [0, 0]]
    # decay * memory = [[0.09, 0.18], [0.27, 0.36]]
    # sum = [[0.59, 0.18], [0.27, 0.36]]
    memory = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    pattern = torch.tensor([1.0, 0.0])
    predicted = torch.tensor([0.0, 0.0])
    updated = hebbian_update(memory, pattern, predicted, decay=0.9, rate=0.5, clip=1.0)
    expected = torch.tensor([[0.59, 0.18], [0.27, 0.36]])
    assert torch.allclose(updated, expected, atol=1e-5)


def test_hebbian_update_clamp_holds_under_repeated_updates() -> None:
    n = 4
    memory = torch.zeros(n, n)
    pattern = torch.tensor([1.0, 1.0, 0.0, 0.0])
    for _ in range(1000):
        memory = hebbian_update(memory, pattern, pattern * 0.5, decay=1.0, rate=0.1, clip=1.0)
    assert memory.max().item() <= 1.0
    assert memory.min().item() >= -1.0


def test_hebbian_update_decay_zero_forgets() -> None:
    memory = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    pattern = torch.tensor([1.0, 0.0])
    predicted = torch.tensor([0.0, 0.0])
    updated = hebbian_update(memory, pattern, predicted, decay=0.0, rate=0.5, clip=None)
    expected = torch.tensor([[0.5, 0.0], [0.0, 0.0]])
    assert torch.allclose(updated, expected)


def test_hebbian_update_no_input_mutation() -> None:
    memory = torch.zeros(2, 2)
    mem_orig = memory.clone()
    pattern = torch.tensor([1.0, 0.0])
    predicted = torch.tensor([0.0, 0.0])
    _ = hebbian_update(memory, pattern, predicted)
    assert torch.equal(memory, mem_orig)


def test_hebbian_update_validation_errors() -> None:
    mem = torch.zeros(2, 2)
    p = torch.tensor([1.0, 0.0])
    with pytest.raises(ValueError, match="decay"):
        hebbian_update(mem, p, p, decay=1.5)
    with pytest.raises(ValueError, match="rate"):
        hebbian_update(mem, p, p, rate=-0.1)
    with pytest.raises(ValueError, match="clip"):
        hebbian_update(mem, p, p, clip=0.0)
    with pytest.raises(ValueError, match="shapes must match"):
        hebbian_update(mem, p, torch.zeros(3))
    with pytest.raises(ValueError, match="memory shape"):
        hebbian_update(torch.zeros(2, 3), p, p)


def test_hebbian_update_batch_broadcasting() -> None:
    memory = torch.zeros(3, 4, 4)
    pattern = torch.zeros(3, 4)
    updated = hebbian_update(memory, pattern, pattern)
    assert updated.shape == (3, 4, 4)


def test_attractor_zero_iterations_returns_cue() -> None:
    cue = torch.tensor([0.5, 0.5])
    memory = torch.eye(2)
    out = attractor(cue, memory, iterations=0)
    assert torch.equal(out, cue)


def test_attractor_one_iteration_hand_computed() -> None:
    # cue = [1.0, 0.0], memory = [[0.5, 0], [0, 0.5]]
    # retention = 0.8, memory @ cue = [0.5, 0]
    # 0.8 * cue + memory @ cue = [0.8 + 0.5, 0] = [1.3, 0]
    # leaky_relu([1.3, 0]) = [1.3, 0]
    # cap = 1.0 -> [1.0, 0.0]
    cue = torch.tensor([1.0, 0.0])
    memory = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
    out = attractor(cue, memory, iterations=1, retention=0.8, cap=1.0)
    expected = torch.tensor([1.0, 0.0])
    assert torch.allclose(out, expected)


def test_attractor_cap_respected() -> None:
    cue = torch.tensor([2.0, 2.0])
    memory = torch.eye(2) * 2.0
    out = attractor(cue, memory, iterations=2, cap=1.0)
    assert out.max().item() <= 1.0


def test_attractor_large_negative_drive_not_clamped_to_minus_cap() -> None:
    cue = torch.tensor([-10.0, 0.0])
    memory = torch.zeros(2, 2)
    out = attractor(cue, memory, iterations=1, retention=1.0, negative_slope=0.01, cap=1.0)
    assert out[0].item() == pytest.approx(-0.1)
    assert out[0].item() > -1.0


def test_attractor_mask_keeps_masked_entries() -> None:
    cue = torch.tensor([1.0, 0.0])
    memory = torch.zeros(2, 2)
    mask = torch.tensor([False, True])
    # Where mask is False (index 0), entry should remain cue[0] = 1.0
    out = attractor(cue, memory, iterations=3, retention=0.8, mask=mask)
    assert out[0].item() == 1.0


def test_four_pattern_store_retrieve() -> None:
    n = 16
    patterns = []
    for i in range(4):
        p = torch.zeros(n)
        p[2 * i] = 1.0
        p[2 * i + 1] = 1.0
        patterns.append(p)

    mem = HebbianMemory(n, decay=1.0, rate=0.5)

    # Control: before storing, retrieval from half-corrupted cue does NOT recover pattern 0
    cue_0 = patterns[0].clone()
    cue_0[1] = 0.0
    retrieved_before = mem.retrieve(cue_0, iterations=5)
    sim_before = torch.dot(retrieved_before, patterns[0]) / (
        retrieved_before.norm() * patterns[0].norm() + 1e-8
    )
    assert sim_before.item() < 0.95

    # Store patterns multiple times
    for _ in range(5):
        for p in patterns:
            mem.store(p)

    # Retrieve each from half-corrupted cue
    for p in patterns:
        cue = p.clone()
        cue[1] = 0.0
        retrieved = mem.retrieve(cue, iterations=5)
        sim = torch.dot(retrieved, p) / (retrieved.norm() * p.norm() + 1e-8)
        assert sim.item() > 0.95


def test_hebbian_memory_reset_and_properties() -> None:
    mem = HebbianMemory(8, decay=0.9, rate=0.2)
    mem.store(torch.ones(8))
    assert mem.memory.abs().sum().item() > 0
    mem_id = id(mem.memory)
    mem.reset()
    assert mem.memory.abs().sum().item() == 0
    assert id(mem.memory) == mem_id

    # Property assignments & validation
    mem.decay = 0.95
    assert mem.decay == 0.95
    with pytest.raises(ValueError, match="decay"):
        mem.decay = 1.5

    mem.rate = 0.3
    assert mem.rate == 0.3
    with pytest.raises(ValueError, match="rate"):
        mem.rate = -0.1

    mem.clip = 2.0
    assert mem.clip == 2.0
    with pytest.raises(ValueError, match="clip"):
        mem.clip = 0.0
