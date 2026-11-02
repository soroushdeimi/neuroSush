"""Device support, training steps, and CPU/device parity tests for TEM models."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neurosush.spatial import GraphWorld, Walk
from neurosush.spatial.batch import batched_random_walks
from neurosush.spatial.controls import seen_targets
from neurosush.tem import TEM, TEMConfig
from neurosush.tem.loss import total_loss
from neurosush.tem.spiking import SpikingTEM, SpikingTEMConfig
from neurosush.tem.train import _row, _worlds


def _run_tem_step(device: str, batch: int = 2) -> None:
    config = TEMConfig(batch=batch, bptt=10)
    generator = torch.Generator(device=device).manual_seed(0)
    model = TEM(config, generator=generator).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_max)

    worlds = _worlds(batch, config, seed=0)
    walk = batched_random_walks(worlds, 15, stay=False, valid_only=True)
    mask = torch.stack([seen_targets(_row(walk, b)) for b in range(batch)]).to(device)
    state = model.reset(batch)

    optimizer.zero_grad()
    losses = []
    for t in range(1, 11):
        obs = walk.observations[:, t].to(device)
        action = walk.executed_actions[:, t - 1].to(device)
        output, state = model(obs, action, state, decay=0.9, rate=0.2)
        loss, _ = total_loss(output, obs, config, iteration=0, mask=mask[:, t - 1])
        losses.append(loss)
    loss = torch.stack(losses).mean()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)


def _run_spiking_step(device: str) -> None:
    config = SpikingTEMConfig(sensory=20, units=32)
    model = SpikingTEM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    world = GraphWorld(width=5, sensory=config.sensory, seed=0)
    cpu_walk = world.random_walk(10)
    walk = Walk(
        nodes=cpu_walk.nodes.to(device),
        observations=cpu_walk.observations.to(device),
        actions=cpu_walk.actions.to(device),
        novel_edge_to_seen=cpu_walk.novel_edge_to_seen.to(device),
        fraction_seen=cpu_walk.fraction_seen.to(device),
        executed_actions=cpu_walk.executed_actions.to(device)
        if cpu_walk.executed_actions is not None
        else None,
    )

    optimizer.zero_grad()
    state = model.initial_state(m_init_std=0.05)
    logits, rates_dict, _ = model(walk, state)
    targets = walk.observations[1:].to(device)
    loss = F.cross_entropy(logits.to(device), targets) + 1e-3 * (
        rates_dict["mecii"].mean() + rates_dict["mecv"].mean() + rates_dict["ca1"].mean()
    )
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)


def test_cpu_models_forward_backward() -> None:
    _run_tem_step("cpu", batch=2)
    _run_spiking_step("cpu")


@pytest.mark.gpu
def test_gpu_models_forward_backward() -> None:
    _run_tem_step("cuda", batch=2)
    _run_spiking_step("cuda")


def test_mps_models_forward_backward_finite() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS device not available")
    _run_tem_step("mps", batch=2)


def test_cpu_vs_device_parity() -> None:
    dev = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else None)
    )
    if dev is None:
        pytest.skip("No non-CPU device available for parity test")

    batch = 2
    bptt = 8
    config = TEMConfig(batch=batch, bptt=bptt)

    # CPU run
    gen_cpu = torch.Generator(device="cpu").manual_seed(123)
    model_cpu = TEM(config, generator=gen_cpu).to("cpu")
    worlds_cpu = _worlds(batch, config, seed=123)
    walk_cpu = batched_random_walks(worlds_cpu, bptt + 1, stay=False, valid_only=True)
    state_cpu = model_cpu.reset(batch)
    obs_cpu = walk_cpu.observations[:, 1].to("cpu")
    act_cpu = walk_cpu.executed_actions[:, 0].to("cpu")
    out_cpu, _ = model_cpu(obs_cpu, act_cpu, state_cpu)

    # Device run
    gen_dev = torch.Generator(device=dev).manual_seed(123)
    model_dev = TEM(config, generator=gen_dev).to(dev)
    model_dev.load_state_dict(model_cpu.state_dict())

    worlds_dev = _worlds(batch, config, seed=123)
    walk_dev = batched_random_walks(worlds_dev, bptt + 1, stay=False, valid_only=True)
    state_dev = model_dev.reset(batch)
    obs_dev = walk_dev.observations[:, 1].to(dev)
    act_dev = walk_dev.executed_actions[:, 0].to(dev)
    out_dev, _ = model_dev(obs_dev, act_dev, state_dev)

    torch.testing.assert_close(out_dev.logits_gen.cpu(), out_cpu.logits_gen, atol=1e-3, rtol=1e-3)
